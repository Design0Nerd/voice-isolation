"""
Phase 4: the actual "extract my voice" logic -- glues separation.py and
embedding.py together.

Approach (Path B, the hybrid approach): split the mixture into 2 streams
with blind separation, embed each stream, compare each embedding against
your enrolled reference embedding, keep whichever stream matches best.

Left room here for the cascading idea you raised: if 3 people are
talking at once, a single 2-way split leaves one stream still containing
2 overlapping voices. `max_passes` lets us re-run separation on the
"losing" stream to try to peel apart a further-mixed pair -- disabled by
default (max_passes=1) since it doubles compute cost per pass and we
haven't tuned when it's actually worth triggering yet. Revisit once the
2-speaker case is proven solid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai.separation import separate
from ai.embedding import compute_embedding, cosine_similarity


@dataclass
class ExtractionResult:
    audio: np.ndarray
    confidence: float          # cosine similarity of the winning stream
    all_similarities: list[float]  # similarity of every candidate stream considered


def extract_target_speaker(
    mixture_16k_mono: np.ndarray,
    reference_embedding: np.ndarray,
    max_passes: int = 1,
) -> ExtractionResult:
    """Returns the stream from `mixture_16k_mono` that best matches
    `reference_embedding` (your enrolled voice)."""

    candidates: list[np.ndarray] = separate(mixture_16k_mono)
    similarities = [
        cosine_similarity(compute_embedding(stream), reference_embedding)
        for stream in candidates
    ]

    best_idx = int(np.argmax(similarities))
    best_audio = candidates[best_idx]
    best_score = similarities[best_idx]

    if max_passes > 1:
        # Re-split the losing stream, in case it's actually 2 overlapping
        # voices rather than 1. Only the winning sub-stream from that
        # second pass is considered -- if it beats our current best, use it.
        losing_idx = 1 - best_idx if len(candidates) == 2 else None
        if losing_idx is not None:
            second_pass = extract_target_speaker(
                candidates[losing_idx], reference_embedding, max_passes=max_passes - 1
            )
            similarities.extend(second_pass.all_similarities)
            if second_pass.confidence > best_score:
                best_audio = second_pass.audio
                best_score = second_pass.confidence

    return ExtractionResult(audio=best_audio, confidence=best_score, all_similarities=similarities)
