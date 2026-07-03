"""
Phase 4: sample rate conversion.

The mic captures at 48kHz (Phase 3 decision), but the separation and
speaker-embedding models both expect 16kHz mono. Kept as a small
standalone module since Phase 6 (streaming) will need this on every
chunk, not just for offline tests.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from math import gcd


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1D float32 array from orig_sr to target_sr.

    Uses polyphase resampling (resample_poly) rather than naive FFT
    resampling -- it's both faster and better behaved for the small
    chunk sizes we'll eventually feed it in Phase 6.
    """
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)
