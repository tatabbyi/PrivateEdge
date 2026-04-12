from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ProtectionMode(str, Enum):
    STRICT = "strict"
    NORMAL = "normal"
    MINIMAL = "minimal"
    EMOTION_ADAPTIVE = "emotion_adaptive"
    SILENT_PROTECTION = "silent_protection"


@dataclass
class ModelScores:
    p_doc: float = 0.0
    p_face_other: float = 0.0
    p_nsfw: float = 0.0
    p_obscene_gesture: float = 0.0
    p_pii_audio: float = 0.0
    p_toxicity: float = 0.0
    anger: float = 0.0
    stress: float = 0.0


@dataclass
class MaskingDecision:
    blur_full_frame: bool = False
    mute_audio: bool = False
    mute_reason: str = ""
    silent_mode: bool = False


@dataclass
class PolicyContext:
    mode: ProtectionMode = ProtectionMode.NORMAL
    tau_doc: float = 0.72
    tau_face: float = 0.65
    tau_nsfw: float = 0.55
    tau_gesture: float = 0.60
    tau_pii: float = 0.60
    tau_toxicity: float = 0.55
    tau_anger: float = 0.75
    modules: dict[str, bool] = field(default_factory=dict)
    detection_sensitivity: float = 0.55
    blur_strength: float = 0.62
    mute_sensitivity: float = 0.5

    def effective_tau(self, base: float) -> float:
        """Higher sensitivity → lower threshold (trigger sooner)."""
        s = max(0.05, min(0.95, self.detection_sensitivity))
        return float(base * (1.1 - s))
