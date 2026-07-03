"""
Phase 4 goal check: mixed audio + reference embedding -> clean user speech.

This test:
  1. Records ~15s of just you talking, to build a reference speaker embedding
     (a rough stand-in for Phase 5's real enrollment flow).
  2. Plays an mp3 of someone else talking out loud through your speakers
     while simultaneously recording your mic -- so your mic naturally
     picks up an acoustic mix of "you talking" + "the mp3 playing",
     without needing a second person in the room.
  3. Runs that live-recorded mixture through separation + speaker
     verification (ai/extract.py) and saves the extracted result.

You'll want to actually talk during step 2, over the mp3, to get a real
overlapping-speech test case -- silence during playback just tests
separation on a single voice, which isn't the interesting case.

Requires: a test mp3 of someone speaking, placed anywhere on disk.
Run:
    python phase4_test.py path/to/other_speaker.mp3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audio.capture import MicCapture
from audio.devices import pick_device_now
from audio.resampler import resample
from ai.embedding import compute_embedding
from ai.extract import extract_target_speaker

OUTPUT_DIR = Path("test_output")
PROFILE_DIR = Path("profiles/default")
ENROLL_SECONDS = 15
MODEL_SR = 16000


def record_seconds(mic: MicCapture, seconds: float, label: str) -> np.ndarray:
    """Drains the mic's ring buffer for `seconds`, printing a countdown."""
    chunks = []
    print(f"\n{label} -- recording for {seconds:.0f}s...")
    start = time.time()
    while time.time() - start < seconds:
        chunk = mic.ring_buffer.pop()
        if chunk is not None:
            chunks.append(chunk)
        else:
            time.sleep(0.005)
        remaining = seconds - (time.time() - start)
        print(f"\r  {remaining:4.1f}s remaining", end="", flush=True)
    print("\r  done.                    ")
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def do_enrollment(mic: MicCapture) -> np.ndarray:
    enroll_path = PROFILE_DIR / "enrollment.wav"
    if enroll_path.exists():
        choice = input(f"Found existing enrollment at {enroll_path}. "
                        f"Press 1 to reuse it, or 2 to re-record: ").strip()
        if choice != "2":
            audio, sr = sf.read(enroll_path, dtype="float32")
            audio_16k = resample(audio, sr, MODEL_SR)
            return compute_embedding(audio_16k)

    input(f"\nAbout to record {ENROLL_SECONDS}s of just YOUR voice for enrollment. "
          f"Press Enter, then talk continuously until it says done.")
    audio = record_seconds(mic, ENROLL_SECONDS, "Enrollment")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(enroll_path, audio, mic.samplerate)
    print(f"Saved enrollment audio to {enroll_path}")

    audio_16k = resample(audio, mic.samplerate, MODEL_SR)
    return compute_embedding(audio_16k)


def do_mixture_capture(mic: MicCapture, mp3_path: Path) -> np.ndarray:
    import sounddevice as sd

    mp3_audio, mp3_sr = sf.read(mp3_path, dtype="float32")
    if mp3_audio.ndim > 1:
        mp3_audio = mp3_audio.mean(axis=1)  # downmix to mono for playback
    duration = len(mp3_audio) / mp3_sr

    print(f"\nPick the OUTPUT device to play the mp3 through -- use your "
          f"speakers/headphones (NOT CABLE Input), so your mic can pick it "
          f"up acoustically.")
    play_device = pick_device_now("output", "playback")

    print(f"\nAbout to play the mp3 ({duration:.0f}s) through '{play_device.name}' "
          f"while recording your mic. TALK over it once it starts, so we get "
          f"real overlapping speech.")
    input("Press Enter to start...")

    mic.ring_buffer.clear()
    sd.play(mp3_audio, samplerate=mp3_sr, device=play_device.index)

    # Record for the mp3's duration plus a little padding on each end.
    mixture = record_seconds(mic, duration + 1.0, "Mixture capture")
    sd.wait()
    return mixture


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase4_test.py path/to/other_speaker.mp3")
        sys.exit(1)
    mp3_path = Path(sys.argv[1])
    if not mp3_path.exists():
        print(f"File not found: {mp3_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase 4: model integration goal check")
    print("=" * 60)

    mic = MicCapture()
    mic.start()

    try:
        reference_embedding = do_enrollment(mic)
        mixture = do_mixture_capture(mic, mp3_path)
    finally:
        mic.stop()

    sf.write(OUTPUT_DIR / "mixture.wav", mixture, mic.samplerate)
    print(f"\nSaved raw mixture recording to {OUTPUT_DIR / 'mixture.wav'}")

    print("\nRunning separation + speaker verification "
          "(first run downloads model weights, may take a while)...")
    mixture_16k = resample(mixture, mic.samplerate, MODEL_SR)
    result = extract_target_speaker(mixture_16k, reference_embedding)

    out_path = OUTPUT_DIR / "extracted.wav"
    sf.write(out_path, result.audio, MODEL_SR)

    print(f"\nStream similarities to your enrolled voice: "
          f"{[f'{s:.3f}' for s in result.all_similarities]}")
    print(f"Picked stream with confidence {result.confidence:.3f}")
    print(f"Saved extracted result to {out_path}")
    print(f"\nListen to {OUTPUT_DIR / 'mixture.wav'} vs {out_path} and compare.")
    print("Phase 4 goal met if the extracted file is mostly you, "
          "with the mp3 voice suppressed.")


if __name__ == "__main__":
    main()
