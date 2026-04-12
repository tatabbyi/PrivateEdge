from __future__ import annotations

from inference.scoring import merge_vision_audio_scores, merge_vision_streams
from policy.types import ModelScores


def test_merge_vision_streams_takes_maxima() -> None:
    a = ModelScores(p_doc=0.2, p_face_other=0.8, p_nsfw=0.1, anger=0.2, stress=0.4)
    b = ModelScores(p_doc=0.7, p_face_other=0.1, p_nsfw=0.9, anger=0.6, stress=0.1)
    out = merge_vision_streams(a, b)
    assert out.p_doc == 0.7
    assert out.p_face_other == 0.8
    assert out.p_nsfw == 0.9
    assert out.anger == 0.6
    assert out.stress == 0.4


def test_merge_vision_audio_scores_uses_audio_for_audio_channels() -> None:
    v = ModelScores(p_doc=0.8, p_face_other=0.5, p_nsfw=0.2, anger=0.1, stress=0.3)
    a = ModelScores(p_pii_audio=0.9, p_toxicity=0.7, anger=0.6, stress=0.5)
    out = merge_vision_audio_scores(v, a)
    assert out.p_doc == 0.8
    assert out.p_pii_audio == 0.9
    assert out.p_toxicity == 0.7
    assert out.anger == 0.6
    assert out.stress == 0.5
