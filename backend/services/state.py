"""In-process shared state for pipeline, policy, and API views."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from policy.engine import PolicyEngine
from policy.types import ModelScores, PolicyContext, ProtectionMode


@dataclass
class RuntimeConfig:
    """Mirrors `configs/default.yaml` module toggles and UI sliders."""

    face_masking: bool = True
    text_document_blocking: bool = True
    nsfw_detection: bool = True
    audio_pii_filtering: bool = True
    mode: str = "emotion_adaptive"
    detection_sensitivity: float = 0.55
    detection_sensitivity_secondary: float = 0.45
    blur_strength: float = 0.62
    blur_strength_secondary: float = 0.35
    mute_sensitivity: float = 0.5
    protection_enabled: bool = True


@dataclass
class Telemetry:
    fps: float = 0.0
    latency_ms: float = 0.0
    npu_percent: float = 0.0
    battery_percent: float | None = None


class AppState:
    """Thread-safe singleton for pipeline ↔ REST ↔ WebSocket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.config = RuntimeConfig()
        self.telemetry = Telemetry()
        ctx = PolicyContext(
            mode=ProtectionMode.EMOTION_ADAPTIVE,
            tau_doc=0.72,
            tau_face=0.65,
            tau_nsfw=0.55,
            tau_pii=0.60,
            tau_toxicity=0.55,
            tau_anger=0.75,
            modules={
                "blur_background_faces": True,
                "blur_documents": True,
                "blur_nsfw": True,
                "mute_pii_audio": True,
                "mute_profanity": True,
                "blur_brands": True,
                "emotion_adaptation": True,
            },
        )
        self.engine = PolicyEngine(ctx)
        self.event_log: deque[dict[str, Any]] = deque(maxlen=50)
        self.audio_status: list[dict[str, str]] = [
            {"id": "ok", "label": "Audio OK", "tone": "ok"},
        ]

    def sync_policy_from_config(self) -> None:
        with self._lock:
            c = self.config
            mode_map = {
                "strict": ProtectionMode.STRICT,
                "normal": ProtectionMode.NORMAL,
                "minimal": ProtectionMode.MINIMAL,
                "emotion_adaptive": ProtectionMode.EMOTION_ADAPTIVE,
                "silent_protection": ProtectionMode.SILENT_PROTECTION,
            }
            self.engine.ctx.mode = mode_map.get(
                c.mode, ProtectionMode.EMOTION_ADAPTIVE
            )
            self.engine.ctx.modules = {
                "blur_background_faces": c.face_masking,
                "blur_documents": c.text_document_blocking,
                "blur_nsfw": c.nsfw_detection,
                "mute_pii_audio": c.audio_pii_filtering,
                "mute_profanity": True,
                "blur_brands": True,
                "emotion_adaptation": c.mode == "emotion_adaptive",
            }
            self.engine.ctx.detection_sensitivity = c.detection_sensitivity
            self.engine.ctx.blur_strength = c.blur_strength
            self.engine.ctx.mute_sensitivity = c.mute_sensitivity

    def log_event(self, message: str, kind: str = "info") -> None:
        with self._lock:
            self.event_log.appendleft(
                {
                    "t": time.time(),
                    "message": message,
                    "kind": kind,
                }
            )

    def set_audio_statuses(self, items: list[dict[str, str]]) -> None:
        with self._lock:
            self.audio_status = items

    def to_public_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config": asdict(self.config),
                "telemetry": asdict(self.telemetry),
            }

    def update_telemetry(self, fps: float, latency_ms: float, npu_percent: float) -> None:
        with self._lock:
            self.telemetry.fps = fps
            self.telemetry.latency_ms = latency_ms
            self.telemetry.npu_percent = npu_percent


STATE = AppState()
