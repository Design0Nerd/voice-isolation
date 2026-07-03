"""
Phase 5 goal check: one-click voice enrollment.

Run:
    python enroll.py

Records your voice (Enter to start, Enter to stop, auto-stop at 20s),
computes a speaker embedding, and saves it as your 'default' profile
under profiles/default/. Later phases (6+) load this instantly instead
of re-recording every run.
"""

from ai.enrollment import record_enrollment, save_profile, profile_exists
from audio.capture import MicCapture

PROFILE_NAME = "default"


def main():
    print("=" * 60)
    print("Phase 5: Voice Enrollment")
    print("=" * 60)

    if profile_exists(PROFILE_NAME):
        choice = input(f"Profile '{PROFILE_NAME}' already exists. "
                        f"Press 1 to re-enroll (overwrite), or anything else to cancel: ").strip()
        if choice != "1":
            print("Cancelled.")
            return

    mic = MicCapture()
    mic.start()
    try:
        audio = record_enrollment(mic)
    finally:
        mic.stop()

    if len(audio) == 0:
        print("No audio recorded, nothing saved.")
        return

    save_profile(PROFILE_NAME, audio, mic.samplerate)
    print("\nPhase 5 goal met. This profile will be used by later phases automatically.")


if __name__ == "__main__":
    main()
