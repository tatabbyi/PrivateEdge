from __future__ import annotations

from policy.engine import PolicyEngine
from policy.types import ModelScores, PolicyContext, ProtectionMode


def _engine(mode: ProtectionMode = ProtectionMode.NORMAL) -> PolicyEngine:
    return PolicyEngine(
        PolicyContext(
            mode=mode,
            modules={
                "blur_background_faces": True,
                "blur_documents": True,
                "blur_nsfw": True,
                "mute_pii_audio": True,
                "mute_profanity": True,
                "emotion_adaptation": True,
            },
        )
    )


def test_blur_triggered_by_nsfw_score() -> None:
    decision = _engine().decide(ModelScores(p_nsfw=0.95))
    assert decision.blur_full_frame is True
    assert decision.mute_audio is False


def test_pii_audio_takes_priority_over_toxicity_reason() -> None:
    decision = _engine().decide(ModelScores(p_pii_audio=0.9, p_toxicity=0.99))
    assert decision.mute_audio is True
    assert decision.mute_reason == "pii"


def test_silent_mode_triggers_in_emotion_adaptive_mode() -> None:
    decision = _engine(ProtectionMode.EMOTION_ADAPTIVE).decide(ModelScores(anger=0.95))
    assert decision.silent_mode is True


def test_minimal_mode_does_not_force_silent_mode() -> None:
    decision = _engine(ProtectionMode.MINIMAL).decide(ModelScores(anger=0.95))
    assert decision.silent_mode is False
