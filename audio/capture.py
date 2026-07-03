"""
Phase 3: Live microphone capture in fixed-size chunks, feeding a RingBuffer.

Captures at the device's native sample rate (no forced resampling here --
that happens later, right before the model, per project decision). Chunk
size is derived from config (chunk_ms), matching the plan's 20ms chunks.
"""

from __future__ import annotations

from pathlib import Path

import sounddevice as sd
import yaml

from audio.ringbuffer import RingBuffer
from audio.devices import list_devices

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def load_audio_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        settings = yaml.safe_load(f) or {}
    return settings.get("audio", {}) or {}


def resolve_device_index(name: str, kind: str) -> int:
    devices = list_devices(kind)
    for dev in devices:
        if dev.name == name:
            return dev.index
    raise RuntimeError(
        f"Configured {kind} device '{name}' not found. "
        f"Run `python -m audio.devices` to re-pick devices."
    )


class MicCapture:
    """Captures audio from the configured input device into a RingBuffer,
    in fixed-size chunks, running on its own callback thread (managed by
    sounddevice/PortAudio internally)."""

    def __init__(self, ring_buffer: RingBuffer | None = None):
        cfg = load_audio_config()
        self.device_name = cfg.get("input_device")
        if not self.device_name:
            raise RuntimeError("No input_device configured. Run `python -m audio.devices` first.")

        self.device_index = resolve_device_index(self.device_name, "input")
        device_info = sd.query_devices(self.device_index)
        self.samplerate = int(device_info["default_samplerate"])
        self.channels = 1  # mono capture -- voice isolation doesn't need stereo input

        chunk_ms = cfg.get("chunk_ms", 20)
        self.chunk_samples = int(self.samplerate * chunk_ms / 1000)

        buffer_chunks = cfg.get("buffer_chunks", 10)
        self.ring_buffer = ring_buffer if ring_buffer is not None else RingBuffer(buffer_chunks)

        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # xruns / overflows show up here -- worth routing to real
            # logging (loguru) once that's wired in; print is fine for now.
            print(f"[MicCapture] status: {status}")
        # indata is (frames, channels) float32; copy() because PortAudio
        # reuses the buffer after the callback returns.
        self.ring_buffer.push(indata[:, 0].copy())

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device_index,
            channels=self.channels,
            samplerate=self.samplerate,
            blocksize=self.chunk_samples,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        print(f"[MicCapture] started: '{self.device_name}' @ {self.samplerate} Hz, "
              f"{self.chunk_samples} samples/chunk")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
