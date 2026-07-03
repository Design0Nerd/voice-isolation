"""
Phase 4: 2-speaker blind source separation, via ClearerVoice-Studio's
MossFormer2_SS_16K model. This is the "split the mix apart" half of our
hybrid pipeline -- it doesn't know or care who's who, it just splits a
mono mixture into 2 separate-speaker streams. embedding.py then figures
out which stream is you.

Weights auto-fetch from HuggingFace on first run and are cached locally
under models/ after that -- no runtime cloud dependency once cached.

Uses ClearVoice's file-based interface (rather than the newer numpy
interface) since it's the most stable, documented path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from clearvoice import ClearVoice

_MODEL = None
MODEL_SR = 16000


def _get_model() -> ClearVoice:
    global _MODEL
    if _MODEL is None:
        _MODEL = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    return _MODEL


def separate(mixture_16k_mono: np.ndarray) -> list[np.ndarray]:
    """mixture_16k_mono: float32 array, 16kHz, mono, containing 2 (or more,
    degraded) overlapping speakers. Returns a list of 2 float32 arrays,
    one per separated stream, in arbitrary order -- embedding.py picks
    the right one afterward."""
    model = _get_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "mixture.wav"
        sf.write(tmp_in, mixture_16k_mono, MODEL_SR)

        # online_write=False (the default) makes ClearVoice return the
        # separated streams directly instead of writing files to disk --
        # confirmed by reading clearvoice's networks.py: process() only
        # touches output_path when online_write=True. The return is a
        # list of arrays (one per speaker), each shaped (1, num_samples).
        result = model(input_path=str(tmp_in))

    if not isinstance(result, list):
        raise RuntimeError(
            f"Expected a list of separated streams from ClearVoice, got {type(result)}. "
            f"The library's return format may have changed."
        )

    streams = [np.asarray(stream).squeeze().astype(np.float32) for stream in result]
    return streams