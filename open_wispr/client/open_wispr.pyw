"""
OpenWispr — Voice-to-Text Client for Windows

Press Ctrl+Shift+Space to start recording.
Release to stop and send audio to HPC for transcription.
Text is inserted at your cursor position.

A small animated overlay shows processing state.

Requirements:
    pip install pyaudio keyboard requests pyperclip pywin32

Usage:
    pythonw open_wispr.pyw          # Run in background (no console)
    python open_wispr.pyw --debug   # Run with console output
"""

import io
import sys
import time
import wave
import threading
import tempfile
from pathlib import Path

import keyboard
import pyaudio
import requests
import pyperclip
import win32api
import win32con
import win32gui
import win32ui
import ctypes

# --- Configuration ---
ASR_URL = "http://localhost:8200/transcribe"
HOTKEY = "ctrl+shift+space"
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024

DEBUG = "--debug" in sys.argv


def log(msg):
    if DEBUG:
        print(f"[OpenWispr] {msg}")


# --- Audio Recording ---
class AudioRecorder:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False

    def start(self):
        """Start recording audio from the microphone."""
        self.frames = []
        self.recording = True
        self.stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._callback,
        )
        self.stream.start_stream()
        log("Recording started")

    def _callback(self, in_data, frame_count, time_info, status):
        if self.recording:
            self.frames.append(in_data)
        return (None, pyaudio.paContinue)

    def stop(self) -> bytes:
        """Stop recording and return WAV bytes."""
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Convert frames to WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.pa.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self.frames))

        duration = len(self.frames) * CHUNK_SIZE / SAMPLE_RATE
        log(f"Recording stopped ({duration:.1f}s, {len(self.frames)} chunks)")
        return buf.getvalue()

    def cleanup(self):
        self.pa.terminate()


# --- Overlay Animation ---
class OverlayWindow:
    """Transparent overlay that shows recording/processing state."""

    def __init__(self):
        self.hwnd = None
        self.visible = False
        self._thread = None

    def show(self, state="recording"):
        """Show the overlay with the given state."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._create_window, args=(state,), daemon=True)
        self._thread.start()

    def update_state(self, state):
        """Update overlay text."""
        if self.hwnd:
            win32gui.PostMessage(self.hwnd, win32con.WM_USER + 1, 0, 0)

    def hide(self):
        """Hide the overlay."""
        if self.hwnd:
            try:
                win32gui.PostMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)
            except Exception:
                pass
        self.hwnd = None
        self.visible = False

    def _create_window(self, state):
        """Create a small transparent overlay window."""
        # Use ctypes for a simpler overlay approach
        try:
            # Get cursor position to place overlay nearby
            cursor = win32api.GetCursorPos()

            # Create a simple tooltip-style overlay using win32
            wc = win32gui.WNDCLASS()
            wc.lpszClassName = "OpenWisprOverlay"
            wc.hbrBackground = win32gui.GetStockObject(win32con.BLACK_BRUSH)
            wc.lpfnWndProc = {
                win32con.WM_CLOSE: lambda hwnd, msg, wp, lp: win32gui.DestroyWindow(hwnd),
                win32con.WM_DESTROY: lambda hwnd, msg, wp, lp: win32gui.PostQuitMessage(0),
            }

            try:
                win32gui.RegisterClass(wc)
            except Exception:
                pass  # Already registered

            # Small rounded indicator near cursor
            width, height = 180, 36
            x = cursor[0] + 20
            y = cursor[1] - 50

            style = win32con.WS_POPUP | win32con.WS_VISIBLE
            ex_style = (
                win32con.WS_EX_TOPMOST
                | win32con.WS_EX_LAYERED
                | win32con.WS_EX_TOOLWINDOW
                | win32con.WS_EX_TRANSPARENT
            )

            self.hwnd = win32gui.CreateWindowEx(
                ex_style,
                "OpenWisprOverlay",
                None,
                style,
                x, y, width, height,
                0, 0, 0, None,
            )

            # Set transparency (200/255 = ~78% opaque)
            ctypes.windll.user32.SetLayeredWindowAttributes(
                self.hwnd, 0, 200, 0x00000002  # LWA_ALPHA
            )

            # Draw text on the window
            hdc = win32gui.GetDC(self.hwnd)
            text = " Recording..." if state == "recording" else " Processing..."
            color = 0x0000FF if state == "recording" else 0x00AAFF  # Red / Orange

            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            win32gui.SetTextColor(hdc, color)
            rect = (8, 8, width - 8, height - 8)
            win32gui.DrawText(hdc, text, -1, rect, win32con.DT_LEFT | win32con.DT_VCENTER)
            win32gui.ReleaseDC(self.hwnd, hdc)

            self.visible = True

            # Message loop
            win32gui.PumpMessages()

        except Exception as e:
            log(f"Overlay error: {e}")


# --- Text Insertion ---
def insert_text_at_cursor(text: str):
    """Insert text at the current cursor position using clipboard + paste."""
    # Save current clipboard
    try:
        old_clip = pyperclip.paste()
    except Exception:
        old_clip = ""

    # Set text to clipboard and paste
    pyperclip.copy(text)
    time.sleep(0.05)

    # Simulate Ctrl+V
    keyboard.press_and_release("ctrl+v")
    time.sleep(0.1)

    # Restore clipboard
    pyperclip.copy(old_clip)
    log(f"Inserted: {text[:50]}...")


# --- ASR Request ---
def transcribe(audio_bytes: bytes) -> str:
    """Send audio to HPC ASR server and return transcribed text."""
    try:
        response = requests.post(
            ASR_URL,
            files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
            data={"language": "en", "prompt": "Transcribe this audio."},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        log(f"Transcription: {result['text']} ({result['processing_ms']:.0f}ms)")
        return result["text"]
    except requests.ConnectionError:
        log("ERROR: Cannot connect to ASR server. Is the tunnel running?")
        return ""
    except Exception as e:
        log(f"ERROR: Transcription failed: {e}")
        return ""


# --- Main Loop ---
class OpenWispr:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.overlay = OverlayWindow()
        self.is_recording = False

    def on_hotkey_press(self):
        """Called when hotkey is pressed — start recording."""
        if self.is_recording:
            return
        self.is_recording = True
        self.overlay.show("recording")
        self.recorder.start()

    def on_hotkey_release(self):
        """Called when hotkey is released — stop, transcribe, insert."""
        if not self.is_recording:
            return
        self.is_recording = False

        # Stop recording
        audio_bytes = self.recorder.stop()

        # Update overlay
        self.overlay.hide()
        self.overlay.show("processing")

        # Transcribe in background
        def _process():
            text = transcribe(audio_bytes)
            self.overlay.hide()
            if text:
                insert_text_at_cursor(text)

        threading.Thread(target=_process, daemon=True).start()

    def run(self):
        """Start the OpenWispr event loop."""
        log(f"OpenWispr started. Hotkey: {HOTKEY}")
        log(f"ASR server: {ASR_URL}")
        log("Press Ctrl+C to exit.")

        # Register hotkey
        keyboard.on_press_key(
            "space",
            lambda e: self.on_hotkey_press()
            if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift")
            else None,
        )
        keyboard.on_release_key(
            "space",
            lambda e: self.on_hotkey_release() if self.is_recording else None,
        )

        # Keep alive
        try:
            keyboard.wait("ctrl+shift+escape")  # Exit combo
        except KeyboardInterrupt:
            pass
        finally:
            self.recorder.cleanup()
            log("OpenWispr stopped.")


if __name__ == "__main__":
    app = OpenWispr()
    app.run()
