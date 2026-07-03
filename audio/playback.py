"""
Phase 3: Audio playback from a RingBuffer.

Used for testing now (loopback to speakers). In Phase 7 this same class
points at 'CABLE Input' instead, which is how processed audio actually
reaches Discord/Zoom/games.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from audio.ringbuffer import RingBuffer
from audio.capture import load_audio_config, resolve_device_index


class Playback:
    def __init__(self, ring_buffer: RingBuffer, device_name: str | None = None,
                 samplerate: int | None = None, chunk_samples: int | None = None):
        cfg = load_audio_config()
        self.device_name = device_name or cfg.get("output_device")
        if not self.device_name:
            raise RuntimeError("No output_device configured. Run `python -m audio.devices` first.")

        self.device_index = resolve_device_index(self.device_name, "output")
        device_info = sd.query_devices(self.device_index)
        self.samplerate = samplerate or int(device_info["default_samplerate"])
        self.channels = 1

        chunk_ms = cfg.get("chunk_ms", 20)
        self.chunk_samples = chunk_samples or int(self.samplerate * chunk_ms / 1000)

        self.ring_buffer = ring_buffer
        self._stream = None
        self._underflows = 0

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print(f"[Playback] status: {status}")
        chunk = self.ring_buffer.pop()
        if chunk is None:
            # Nothing ready yet -- output silence rather than glitch noise.
            outdata[:, 0] = 0.0
            self._underflows += 1
            return
        if len(chunk) < frames:
            # Shouldn't normally happen since chunk sizes match, but pad
            # defensively rather than crash the audio callback.
            padded = np.zeros(frames, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded
        outdata[:, 0] = chunk[:frames]

    def start(self) -> None:
        try:
            self._stream = sd.OutputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.chunk_samples,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            raise RuntimeError(
                f"Failed to open '{self.device_name}' at {self.samplerate} Hz ({e}). "
                f"Some WASAPI devices refuse non-default rates even in shared mode -- "
                f"if this happens, open Windows Sound settings > this device's "
                f"Properties > Advanced tab, and set the default format's sample "
                f"rate to match (e.g. 48000 Hz), then retry."
            ) from e
        print(f"[Playback] started: '{self.device_name}' @ {self.samplerate} Hz")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def underflows(self) -> int:
        return self._underflows
