"""
Phase 5: Voice enrollment.

Records a reference sample of your voice, computes a speaker embedding
from it, and saves both to a profile folder under profiles/<name>/ so it
can be reloaded instantly later (Phase 6+ won't need to re-record or
re-run enrollment on every startup).

Recording control: press Enter to start, press Enter again to stop,
auto-stops at MAX_SECONDS regardless. A recording under MIN_SECONDS
still works but produces a noisier embedding, so we warn (not block).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio.capture import MicCapture
from audio.resampler import resample
from ai.embedding import compute_embedding

PROFILES_DIR = Path("profiles")
MIN_SECONDS = 8.0
MAX_SECONDS = 20.0
MODEL_SR = 16000


def _wait_for_enter(stop_event: threading.Event) -> None:
    input()
    stop_event.set()


def record_enrollment(mic: MicCapture) -> np.ndarray:
    """Press-Enter-to-start/stop recording, capped at MAX_SECONDS."""
    input("Press Enter to START recording (then talk naturally -- read "
          "something out loud, or just talk about your day)...")

    mic.ring_buffer.clear()
    stop_event = threading.Event()
    stop_thread = threading.Thread(target=_wait_for_enter, args=(stop_event,), daemon=True)

    print(f"Recording... press Enter to STOP (auto-stops at {MAX_SECONDS:.0f}s)")
    start_time = time.time()
    stop_thread.start()

    chunks = []
    while True:
        elapsed = time.time() - start_time
        if stop_event.is_set() or elapsed >= MAX_SECONDS:
            break
        chunk = mic.ring_buffer.pop()
        if chunk is not None:
            chunks.append(chunk)
        else:
            time.sleep(0.005)
        print(f"\r  {elapsed:4.1f}s recorded", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\r  stopped at {elapsed:.1f}s recorded" + " " * 10)

    if elapsed < MIN_SECONDS:
        print(f"  NOTE: that's under {MIN_SECONDS:.0f}s -- the embedding may be "
              f"noisier than ideal. You can re-run this if you want a longer sample.")

    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def save_profile(name: str, audio: np.ndarray, samplerate: int) -> Path:
    """Saves the raw enrollment audio + computed embedding to
    profiles/<name>/. Returns the profile directory."""
    profile_dir = PROFILES_DIR / name
    profile_dir.mkdir(parents=True, exist_ok=True)

    wav_path = profile_dir / "enrollment.wav"
    sf.write(wav_path, audio, samplerate)

    audio_16k = resample(audio, samplerate, MODEL_SR)
    embedding = compute_embedding(audio_16k)

    embedding_path = profile_dir / "speaker.pt"
    torch.save(torch.from_numpy(embedding), embedding_path)

    print(f"\nSaved profile '{name}':")
    print(f"  audio:     {wav_path}")
    print(f"  embedding: {embedding_path}")
    return profile_dir


def load_profile_embedding(name: str = "default") -> np.ndarray:
    """Loads a previously saved embedding without touching the mic."""
    embedding_path = PROFILES_DIR / name / "speaker.pt"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"No profile named '{name}' found at {embedding_path}. "
            f"Run enroll.py first."
        )
    return torch.load(embedding_path).numpy()


def profile_exists(name: str = "default") -> bool:
    return (PROFILES_DIR / name / "speaker.pt").exists()
