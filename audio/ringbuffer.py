"""
Phase 3: Ring buffer for chunked real-time audio.

Works in whole chunks (not individual samples) since the rest of the
pipeline (capture, model inference, playback) all operate in fixed-size
20ms chunks anyway. Thread-safe: the sounddevice capture callback writes
from an audio thread, and playback/inference reads from a different
thread.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class RingBuffer:
    def __init__(self, capacity_chunks: int):
        if capacity_chunks < 1:
            raise ValueError("capacity_chunks must be >= 1")
        self._buffer: deque = deque(maxlen=capacity_chunks)
        self._lock = threading.Lock()
        self._dropped_chunks = 0

    def push(self, chunk: np.ndarray) -> None:
        """Add a chunk. If the buffer is full, the oldest chunk is dropped
        automatically (deque maxlen behavior) -- for real-time audio,
        keeping the newest data is more useful than keeping old data."""
        with self._lock:
            if len(self._buffer) == self._buffer.maxlen:
                self._dropped_chunks += 1
            self._buffer.append(chunk)

    def pop(self):
        """Remove and return the oldest chunk, or None if empty."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def dropped_chunks(self) -> int:
        return self._dropped_chunks

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
