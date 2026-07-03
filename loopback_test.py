"""
Phase 3 goal check: mic -> ring buffer -> speakers loopback.

Proves the whole chunked capture/buffer/playback path works end-to-end
before Phase 4 inserts the AI model in between. Talk into your mic --
you should hear yourself out of your speakers with a small delay.

NOTE: this plays back on your normal speakers (not CABLE Input -- that
wiring happens in Phase 7), so use headphones or keep volume moderate
to avoid feedback if your speakers are near your mic.

Run:
    python loopback_test.py
Stop with Ctrl+C.
"""

import time

from audio.capture import MicCapture
from audio.playback import Playback


def main():
    mic = MicCapture()
    buffer = mic.ring_buffer

    # Force the output stream to open at the mic's samplerate rather than
    # the output device's own default. WASAPI shared-mode streams (which
    # is what audio.devices filters for) can run at an arbitrary rate --
    # Windows' audio engine resamples internally for shared-mode streams,
    # so this avoids the chunk-timing mismatch that caused the underflow
    # storm (device defaulted to 192000 Hz while chunks were sized for
    # 48000 Hz, so playback callbacks fired 4x too often).
    playback = Playback(
        ring_buffer=buffer,
        samplerate=mic.samplerate,
        chunk_samples=mic.chunk_samples,
    )

    mic.start()
    playback.start()

    print("\nLoopback running. Talk into your mic -- you should hear "
          "yourself with a small delay. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            print(f"\rbuffer depth: {len(buffer):3d} chunks | "
                  f"dropped: {buffer.dropped_chunks:3d} | "
                  f"underflows: {playback.underflows:3d}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping...")
        mic.stop()
        playback.stop()


if __name__ == "__main__":
    main()
