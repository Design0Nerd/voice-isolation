"""
Phase 4: speaker embedding / verification, via SpeechBrain's pretrained
ECAPA-TDNN model (speechbrain/spkrec-ecapa-voxceleb).

This is the "who is talking" half of our hybrid pipeline: given a wav of
someone talking, produce a fixed-length embedding vector. Comparing two
embeddings with cosine similarity tells us how likely they're the same
speaker. Weights are fetched from HuggingFace on first use and cached
under ~/.cache (or wherever HF_HOME points), then reused fully offline
after that -- no runtime network calls once cached.
"""

from __future__ import annotations

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

_MODEL = None


def _get_model() -> EncoderClassifier:
    global _MODEL
    if _MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
    return _MODEL


def compute_embedding(audio_16k_mono: np.ndarray) -> np.ndarray:
    """audio_16k_mono: float32 array, 16kHz, mono. Returns a 1D embedding
    vector (192-dim for this ECAPA-TDNN checkpoint)."""
    model = _get_model()
    wav_tensor = torch.from_numpy(audio_16k_mono).float().unsqueeze(0)  # (1, T)
    with torch.no_grad():
        embedding = model.encode_batch(wav_tensor)  # (1, 1, D)
    return embedding.squeeze().cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))
