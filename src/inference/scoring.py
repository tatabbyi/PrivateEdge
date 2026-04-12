"""Fuse vision and audio model outputs into one score vector."""

from __future__ import annotations

from policy.types import ModelScores


def merge_vision_streams(a: ModelScores, b: ModelScores) -> ModelScores:
    """Combine two vision score vectors (e.g. webcam + screen) by taking maxima."""
    return ModelScores(
        p_doc=max(a.p_doc, b.p_doc),
        p_face_other=max(a.p_face_other, b.p_face_other),
        p_nsfw=max(a.p_nsfw, b.p_nsfw),
        p_obscene_gesture=max(a.p_obscene_gesture, b.p_obscene_gesture),
        p_pii_audio=0.0,
        p_toxicity=max(a.p_toxicity, b.p_toxicity),
        anger=max(a.anger, b.anger),
        stress=max(a.stress, b.stress),
    )


def merge_vision_audio_scores(v: ModelScores, a: ModelScores) -> ModelScores:
    return ModelScores(
        p_doc=v.p_doc,
        p_face_other=v.p_face_other,
        p_nsfw=v.p_nsfw,
        p_obscene_gesture=v.p_obscene_gesture,
        p_pii_audio=a.p_pii_audio,
        p_toxicity=a.p_toxicity,
        anger=max(v.anger, a.anger),
        stress=max(v.stress, a.stress),
    )
