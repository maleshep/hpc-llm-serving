"""
OpenWispr Desktop — Toggle-Based Floating Overlay

A small always-on-top pill widget:
  - Click it or press Ctrl+Alt+L to toggle recording
  - Shows animated waveform while listening
  - Auto-stops after 1.5s of silence
  - Transcribes via HPC and pastes at cursor

Requirements: pip install --user pyaudio requests pynput
Usage: pythonw app.pyw (background) | python app.pyw --debug (console)
"""

import io
import sys
import time
import wave
import math
import struct
import threading
import ctypes
from ctypes import wintypes

# --- Config ---
ASR_URL = "http://localhost:8200/transcribe"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
DEBUG = "--debug" in sys.argv

SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5
MIN_RECORDING_DURATION = 0.5


def log(msg):
    if DEBUG:
        print(f"[OpenWispr] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# --- Imports ---
try:
    import pyaudio
    import requests
    from pynput import keyboard
except ImportError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"Missing dependency: {missing}")
    print("Install with: pip install --user pyaudio requests pynput")
    sys.exit(1)

import tkinter as tk


# --- Audio Recording ---
class Recorder:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False
        self.lock = threading.Lock()
        self.current_level = 0.0
        self._level_callback = None
        self.silence_start = None

    def start(self, level_callback=None):
        with self.lock:
            if self.recording:
                return
            self.frames = []
            self.recording = True
            self._level_callback = level_callback
            self.silence_start = None
            self.stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._cb,
            )
            self.stream.start_stream()
            log("Recording started")

    def _cb(self, in_data, frame_count, time_info, status):
        if self.recording:
            self.frames.append(in_data)
            samples = struct.unpack(f'<{frame_count}h', in_data)
            rms = math.sqrt(sum(s * s for s in samples) / len(samples))
            self.current_level = min(1.0, rms / 6000.0)
            if self._level_callback:
                self._level_callback(self.current_level, rms)
        return (None, pyaudio.paContinue)

    def get_duration(self):
        return len(self.frames) * CHUNK_SIZE / SAMPLE_RATE

    def stop(self) -> bytes:
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

        dur = len(self.frames) * CHUNK_SIZE / SAMPLE_RATE
        log(f"Recorded {dur:.1f}s")
        return buf.getvalue()


# --- Clipboard + Paste ---
def paste_text(text: str):
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


# --- ASR ---
def transcribe(wav_bytes: bytes) -> str:
    try:
        r = requests.post(ASR_URL, files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
                          data={"language": "English"}, timeout=30)
        r.raise_for_status()
        result = r.json()
        log(f"Transcribed ({result.get('processing_ms', '?')}ms): {result['text'][:80]}")
        return result["text"]
    except requests.ConnectionError:
        log("ERROR: Can't connect — is SSH tunnel running?")
        return ""
    except Exception as e:
        log(f"ERROR: {e}")
        return ""


# --- Overlay Widget ---
class OpenWisprOverlay:

    # Colors
    BG = "#1a1b2e"
    ACCENT = "#6366f1"
    RED = "#ef4444"
    AMBER = "#f59e0b"
    GREEN = "#22c55e"
    TEXT = "#e2e8f0"
    TEXT_DIM = "#64748b"

    WIDTH_IDLE = 180
    WIDTH_ACTIVE = 320
    HEIGHT = 54
    BAR_COUNT = 24

    def __init__(self):
        self.state = "idle"
        self.recorder = Recorder()
        self.root = None
        self.canvas = None
        self._bars = [0.0] * self.BAR_COUNT
        self._target_bars = [0.0] * self.BAR_COUNT
        self._anim_id = None
        self._pressed_keys = set()
        self._connected = False

    def run(self):
        self.root = tk.Tk()
        self.root.title("OpenWispr")
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.95)

        # Position bottom-center
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - self.WIDTH_IDLE) // 2
        y = screen_h - self.HEIGHT - 80
        self.root.geometry(f"{self.WIDTH_IDLE}x{self.HEIGHT}+{x}+{y}")

        # Use a Frame for proper event handling with overrideredirect
        self.frame = tk.Frame(self.root, bg=self.BG, cursor="hand2")
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.frame, width=self.WIDTH_IDLE, height=self.HEIGHT,
            bg=self.BG, highlightthickness=0, cursor="hand2"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind click to BOTH frame and canvas, using ButtonRelease
        self.canvas.bind("<ButtonRelease-1>", self._on_click)
        self.frame.bind("<ButtonRelease-1>", self._on_click)

        # Drag support via ButtonPress + Motion
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.frame.bind("<ButtonPress-1>", self._on_press)
        self.frame.bind("<B1-Motion>", self._on_motion)

        # Right-click quit
        self.canvas.bind("<Button-3>", self._right_click)
        self.frame.bind("<Button-3>", self._right_click)

        # Track drag state
        self._drag_x = 0
        self._drag_y = 0
        self._dragged = False

        self._draw_idle()

        # Focus hack for overrideredirect on Windows
        self.root.bind("<FocusIn>", lambda e: None)
        self.root.after(100, self._force_focus)

        # Global hotkey (Ctrl+Alt+L)
        listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        listener.daemon = True
        listener.start()
        log("Overlay ready — Ctrl+Alt+L or click to toggle")

        # Check ASR on startup
        self.root.after(500, self._check_connection)

        self.root.mainloop()

    def _force_focus(self):
        """Force the window to accept input on Windows with overrideredirect."""
        try:
            hwnd = int(self.root.frame(), 16)
        except Exception:
            try:
                hwnd = ctypes.windll.user32.GetForegroundWindow()
            except Exception:
                pass

    # --- Click and drag ---

    def _on_press(self, event):
        self._drag_x = event.x_root
        self._drag_y = event.y_root
        self._dragged = False

    def _on_motion(self, event):
        dx = event.x_root - self._drag_x
        dy = event.y_root - self._drag_y
        if abs(dx) > 3 or abs(dy) > 3:
            self._dragged = True
            new_x = self.root.winfo_x() + dx
            new_y = self.root.winfo_y() + dy
            self.root.geometry(f"+{new_x}+{new_y}")
            self._drag_x = event.x_root
            self._drag_y = event.y_root

    def _on_click(self, event):
        if self._dragged:
            self._dragged = False
            return
        log(f"CLICK! state={self.state}")
        self._toggle()

    def _right_click(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Quit OpenWispr", command=self._quit)
        menu.tk_popup(event.x_root, event.y_root)

    # --- Hotkey (Ctrl+Alt+L) ---

    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self._pressed_keys.add('ctrl')
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r or key == keyboard.Key.alt_gr:
                self._pressed_keys.add('alt')
            else:
                # 'L' key: check vk code (0x4C) or char
                vk = getattr(key, 'vk', None)
                char = getattr(key, 'char', None)
                is_L = (vk == 0x4C) or (char and char.lower() == 'l')
                if is_L and 'ctrl' in self._pressed_keys and 'alt' in self._pressed_keys:
                    log("Hotkey Ctrl+Alt+L!")
                    self.root.after(0, self._toggle)
        except Exception as e:
            log(f"Key error: {e}")

    def _on_key_release(self, key):
        try:
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self._pressed_keys.discard('ctrl')
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r or key == keyboard.Key.alt_gr:
                self._pressed_keys.discard('alt')
        except Exception:
            pass

    # --- State machine ---

    def _toggle(self):
        log(f"Toggle: {self.state}")
        if self.state == "idle":
            self._start_listening()
        elif self.state == "listening":
            self._stop_and_transcribe()

    def _start_listening(self):
        self.state = "listening"
        log("STATE → listening")
        self._resize(self.WIDTH_ACTIVE)
        self.recorder.start(level_callback=self._on_audio_level)
        self._animate_waveform()

    def _on_audio_level(self, level, rms):
        self._target_bars = self._target_bars[1:] + [level]
        # Auto-stop on silence
        if rms < SILENCE_THRESHOLD:
            if self.recorder.silence_start is None:
                self.recorder.silence_start = time.time()
            elif (time.time() - self.recorder.silence_start > SILENCE_DURATION
                  and self.recorder.get_duration() > MIN_RECORDING_DURATION):
                log("Silence → auto-stop")
                self.root.after(0, self._stop_and_transcribe)
                self.recorder.silence_start = None
        else:
            self.recorder.silence_start = None

    def _stop_and_transcribe(self):
        if self.state != "listening":
            return
        self.state = "processing"
        log("STATE → processing")
        wav = self.recorder.stop()
        self._draw_processing()

        def _process():
            text = transcribe(wav)
            if text:
                paste_text(text)
                self.root.after(0, lambda: self._show_result("success", text))
            else:
                self.root.after(0, lambda: self._show_result("error", ""))

        threading.Thread(target=_process, daemon=True).start()

    def _show_result(self, result, text):
        self.state = result
        log(f"STATE → {result}: {text[:50]}")
        if result == "success":
            self._draw_success(text)
        else:
            self._draw_error()
        self.root.after(2000, self._go_idle)

    def _go_idle(self):
        self.state = "idle"
        self._bars = [0.0] * self.BAR_COUNT
        self._target_bars = [0.0] * self.BAR_COUNT
        self._resize(self.WIDTH_IDLE)
        self._draw_idle()
        log("STATE → idle")

    def _resize(self, width):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        old_w = self.root.winfo_width()
        new_x = x - (width - old_w) // 2
        self.root.geometry(f"{width}x{self.HEIGHT}+{new_x}+{y}")
        self.canvas.configure(width=width)

    # --- Drawing ---

    def _draw_idle(self):
        if self._anim_id:
            self.root.after_cancel(self._anim_id)
            self._anim_id = None
        c = self.canvas
        c.delete("all")

        # Mic circle
        c.create_oval(10, 11, 42, 43, fill=self.ACCENT, outline="")
        # Mic icon
        c.create_rectangle(21, 17, 31, 32, fill="white", outline="")
        c.create_arc(18, 24, 34, 38, start=0, extent=-180, style=tk.ARC, outline="white", width=2)
        c.create_line(26, 38, 26, 42, fill="white", width=2)

        # Text
        c.create_text(105, self.HEIGHT // 2, text="OpenWispr",
                      fill=self.TEXT, font=("Segoe UI", 12, "bold"))

        # Status dot
        dot_color = self.GREEN if self._connected else self.RED
        c.create_oval(160, 23, 170, 33, fill=dot_color, outline="")

    def _draw_processing(self):
        if self._anim_id:
            self.root.after_cancel(self._anim_id)
            self._anim_id = None
        self._dots_frame = 0
        self._animate_dots()

    def _draw_success(self, text):
        if self._anim_id:
            self.root.after_cancel(self._anim_id)
            self._anim_id = None
        c = self.canvas
        c.delete("all")
        display = text[:35] + "..." if len(text) > 35 else text
        c.create_text(self.WIDTH_ACTIVE // 2, 17,
                      text="Pasted!", fill=self.GREEN, font=("Segoe UI", 11, "bold"))
        c.create_text(self.WIDTH_ACTIVE // 2, 37,
                      text=display, fill=self.TEXT_DIM, font=("Segoe UI", 9))

    def _draw_error(self):
        if self._anim_id:
            self.root.after_cancel(self._anim_id)
            self._anim_id = None
        c = self.canvas
        c.delete("all")
        c.create_text(self.WIDTH_ACTIVE // 2, self.HEIGHT // 2,
                      text="No transcription — check tunnel", fill=self.RED,
                      font=("Segoe UI", 10))

    # --- Animations ---

    def _animate_waveform(self):
        if self.state != "listening":
            return
        c = self.canvas
        c.delete("all")

        # Smooth bars
        for i in range(self.BAR_COUNT):
            self._bars[i] += (self._target_bars[i] - self._bars[i]) * 0.35

        bar_w = 7
        gap = 3
        total_w = self.BAR_COUNT * (bar_w + gap)
        start_x = (self.WIDTH_ACTIVE - total_w) // 2 + 20
        cy = self.HEIGHT // 2

        for i, level in enumerate(self._bars):
            x = start_x + i * (bar_w + gap)
            h = max(4, level * 34)
            y1 = cy - h / 2
            y2 = cy + h / 2
            c.create_rectangle(x, y1, x + bar_w, y2, fill=self.RED, outline="")

        # REC label
        c.create_oval(8, 20, 20, 32, fill=self.RED, outline="")
        c.create_text(35, self.HEIGHT // 2, text=f"{self.recorder.get_duration():.0f}s",
                      fill=self.TEXT_DIM, font=("Segoe UI", 9), anchor="w")

        self._anim_id = self.root.after(50, self._animate_waveform)

    def _animate_dots(self):
        if self.state != "processing":
            return
        self._dots_frame += 1
        dots = "." * ((self._dots_frame % 3) + 1)
        c = self.canvas
        c.delete("all")
        c.create_text(self.WIDTH_ACTIVE // 2, self.HEIGHT // 2,
                      text=f"Transcribing{dots}",
                      fill=self.AMBER, font=("Segoe UI", 12, "bold"))
        self._anim_id = self.root.after(400, self._animate_dots)

    # --- Connection check ---

    def _check_connection(self):
        def _check():
            try:
                r = requests.get(ASR_URL.replace("/transcribe", "/health"), timeout=5)
                self._connected = r.ok
                log(f"ASR: {'connected' if r.ok else 'error'}")
            except Exception:
                self._connected = False
                log("ASR not reachable")
            self.root.after(0, self._draw_idle)
        threading.Thread(target=_check, daemon=True).start()

    def _quit(self):
        log("Quitting")
        self.root.destroy()


if __name__ == "__main__":
    log("OpenWispr starting...")
    log(f"ASR: {ASR_URL}")
    log("Ctrl+Alt+L or click pill to toggle")
    overlay = OpenWisprOverlay()
    overlay.run()
