"""
OpenWispr Desktop v2 — pywebview-based floating pill overlay.

A polished always-on-top dictation widget:
  - Click or Ctrl+Alt+L to toggle recording
  - Smooth waveform visualization while listening
  - Auto-stops after silence
  - Transcribes via ASR endpoint and pastes at cursor

Requirements: pip install -r requirements_v2.txt
Usage: pythonw client_v2.pyw (background) | python client_v2.pyw --debug (console)
"""

import io
import os
import sys
import json
import time
import wave
import math
import struct
import base64
import threading
import ctypes
from ctypes import wintypes
from pathlib import Path

import webview
import pyaudio
import requests
from pynput import keyboard

# --- Config ---
CONFIG_PATH = Path(__file__).parent / "config.json"
DEBUG = "--debug" in sys.argv

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONFIG = load_config()

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024


def log(msg):
    if DEBUG:
        print(f"[OpenWispr] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# --- Audio Recording ---
class Recorder:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False
        self.lock = threading.Lock()
        self.current_level = 0.0
        self.current_rms = 0.0
        self.silence_start = None
        self.start_time = 0

    def start(self):
        with self.lock:
            if self.recording:
                return False
            self.frames = []
            self.recording = True
            self.silence_start = None
            self.start_time = time.time()
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._callback,
            )
            self.stream.start_stream()
            log("Recording started")
            return True

    def _callback(self, in_data, frame_count, time_info, status):
        if self.recording:
            self.frames.append(in_data)
            samples = struct.unpack(f'<{frame_count}h', in_data)
            rms = math.sqrt(sum(s * s for s in samples) / len(samples))
            self.current_rms = rms
            self.current_level = min(1.0, rms / 6000.0)
        return (None, pyaudio.paContinue)

    def get_level(self):
        return self.current_level

    def get_duration(self):
        if self.start_time == 0:
            return 0
        return time.time() - self.start_time

    def should_auto_stop(self):
        if not CONFIG.get("auto_stop", True):
            return False
        threshold = CONFIG.get("silence_threshold", 500)
        silence_ms = CONFIG.get("silence_duration_ms", 1500)
        min_ms = CONFIG.get("min_recording_ms", 500)

        if self.current_rms < threshold:
            if self.silence_start is None:
                self.silence_start = time.time()
            elif (time.time() - self.silence_start > silence_ms / 1000.0
                  and self.get_duration() > min_ms / 1000.0):
                return True
        else:
            self.silence_start = None
        return False

    def stop(self):
        with self.lock:
            self.recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

        if not self.frames:
            return b''

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))

        dur = self.get_duration()
        log(f"Recorded {dur:.1f}s")
        self.start_time = 0
        return buf.getvalue()


# --- Clipboard + Paste ---
def do_paste(text: str):
    CF_UNICODETEXT = 13
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    user32.OpenClipboard(0)
    user32.EmptyClipboard()
    encoded = text.encode('utf-16-le') + b'\x00\x00'
    h = kernel32.GlobalAlloc(0x0042, len(encoded))
    p = kernel32.GlobalLock(h)
    ctypes.memmove(p, encoded, len(encoded))
    kernel32.GlobalUnlock(h)
    user32.SetClipboardData(CF_UNICODETEXT, h)
    user32.CloseClipboard()

    time.sleep(0.05)

    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    VK_CONTROL = 0x11
    VK_V = 0x56

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]
        _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]

    def send_key(vk, up=False):
        x = INPUT(type=INPUT_KEYBOARD)
        x._input.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP if up else 0,
                                  time=0, dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)))
        user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    send_key(VK_CONTROL)
    send_key(VK_V)
    time.sleep(0.02)
    send_key(VK_V, up=True)
    send_key(VK_CONTROL, up=True)
    log(f"Pasted: {text[:60]}...")


# --- API exposed to JS via pywebview ---
class Api:
    def __init__(self):
        self.recorder = Recorder()
        self._window = None

    def set_window(self, window):
        self._window = window

    def start_recording(self):
        result = self.recorder.start()
        log(f"start_recording → {result}")
        return result

    def stop_recording(self):
        wav_bytes = self.recorder.stop()
        if not wav_bytes:
            return ""
        encoded = base64.b64encode(wav_bytes).decode('ascii')
        log(f"stop_recording → {len(wav_bytes)} bytes")
        return encoded

    def get_audio_level(self):
        level = self.recorder.get_level()
        auto_stop = self.recorder.should_auto_stop()
        duration = self.recorder.get_duration()
        return {"level": level, "auto_stop": auto_stop, "duration": duration}

    def transcribe(self, audio_b64):
        wav_bytes = base64.b64decode(audio_b64)
        endpoint = CONFIG.get("asr_endpoint", "http://localhost:8200")
        url = f"{endpoint}/transcribe"
        language = CONFIG.get("language", "English")
        try:
            r = requests.post(
                url,
                files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
                data={"language": language},
                timeout=30
            )
            r.raise_for_status()
            result = r.json()
            text = result.get("text", "")
            log(f"Transcribed ({result.get('processing_ms', '?')}ms): {text[:80]}")
            return {"text": text, "error": None}
        except requests.ConnectionError:
            log("ERROR: Can't connect to ASR")
            return {"text": "", "error": "No connection — check tunnel"}
        except Exception as e:
            log(f"ERROR: {e}")
            return {"text": "", "error": str(e)}

    def paste_text(self, text):
        do_paste(text)
        return True

    def get_config(self):
        return CONFIG

    def check_health(self):
        endpoint = CONFIG.get("asr_endpoint", "http://localhost:8200")
        try:
            r = requests.get(f"{endpoint}/health", timeout=5)
            return r.ok
        except Exception:
            return False

    def resize_window(self, width, height):
        if self._window:
            self._window.resize(width, height)

    def toggle_from_hotkey(self):
        if self._window:
            self._window.evaluate_js("window.wisprToggle && window.wisprToggle()")


# --- Global Hotkey ---
class HotkeyListener:
    def __init__(self, api):
        self.api = api
        self._pressed = set()

    def start(self):
        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        listener.daemon = True
        listener.start()
        log("Hotkey listener started (Ctrl+Alt+L)")

    def _on_press(self, key):
        try:
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._pressed.add('ctrl')
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
                self._pressed.add('alt')
            else:
                vk = getattr(key, 'vk', None)
                char = getattr(key, 'char', None)
                is_l = (vk == 0x4C) or (char and char.lower() == 'l')
                if is_l and 'ctrl' in self._pressed and 'alt' in self._pressed:
                    log("Hotkey Ctrl+Alt+L triggered")
                    self.api.toggle_from_hotkey()
        except Exception as e:
            log(f"Hotkey error: {e}")

    def _on_release(self, key):
        try:
            if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
                self._pressed.discard('ctrl')
            elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
                self._pressed.discard('alt')
        except Exception:
            pass


# --- Main ---
def main():
    log("OpenWispr v2 starting...")
    log(f"ASR: {CONFIG.get('asr_endpoint')}")

    api = Api()

    # Calculate starting position (bottom-center)
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    win_w = 200
    win_h = 48
    x = (screen_w - win_w) // 2
    y = screen_h - win_h - 80

    html_path = Path(__file__).parent / "ui.html"

    window = webview.create_window(
        "OpenWispr",
        url=str(html_path),
        width=win_w,
        height=win_h,
        x=x,
        y=y,
        frameless=True,
        transparent=True,
        on_top=True,
        resizable=False,
        js_api=api,
    )

    api.set_window(window)

    # Start hotkey listener after window is created
    hotkey = HotkeyListener(api)
    hotkey.start()

    log("Window created, starting webview...")
    webview.start(debug=DEBUG)


if __name__ == "__main__":
    main()
