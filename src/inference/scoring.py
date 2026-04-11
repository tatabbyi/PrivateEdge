"""Fuse vision and audio model outputs into one score vector."""

from __future__ import annotations

from policy.types import ModelScores


def merge_vision_audio_scores(v: ModelScores, a: ModelScores) -> ModelScores:
    return ModelScores(
        p_doc=v.p_doc,
        p_face_other=v.p_face_other,
        p_nsfw=v.p_nsfw,
        p_pii_audio=a.p_pii_audio,
        p_toxicity=a.p_toxicity,
        anger=max(v.anger, a.anger),
        stress=max(v.stress, a.stress),
    )
