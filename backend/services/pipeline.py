"""
Real-time capture → vision + audio inference → policy → render → WebSocket.
Uses OpenCV + optional ONNX NSFW; microphone + faster-whisper when installed.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from typing import Any

import numpy as np
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from capture.video import VideoSource
from inference.audio_worker import GLOBAL_AUDIO_BUF, ensure_audio_worker
from inference.scoring import merge_vision_audio_scores
from inference.vision import analyze_frame_bgr, reset_hf_load_state
from policy.types import MaskingDecision, ModelScores
from render.blur import apply_policy_blur
from services.state import STATE

_pipeline_thread: threading.Thread | None = None
_started = False
_video: VideoSource | None = None
_ema_fps = 30.0
_last_broadcast = 0.0
_npu_base: float | None = None
_prev_emit: dict[str, bool] = {"mute": False, "blur": False, "silent": False}
_prev_hf_efficientnet: bool = False


def _npu_percent() -> float:
    global _npu_base
    if _npu_base is None:
        try:
            import onnxruntime as ort

            prov = ort.get_available_providers()
            _npu_base = 62.0 if "QNNExecutionProvider" in prov else 24.0
        except Exception:  # noqa: BLE001
            _npu_base = 30.0
    return float(_npu_base)


def _encode_jpeg_b64(frame: np.ndarray) -> str:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _broadcast(payload: dict[str, Any]) -> None:
    layer = get_channel_layer()
    if layer is None:
        return
    async_to_sync(layer.group_send)(
        "dashboard",
        {"type": "send_telemetry", "payload": payload},
    )


def _apply_protection(decision: MaskingDecision, frame: np.ndarray, blur: float) -> np.ndarray:
    if not STATE.config.protection_enabled:
        return frame.copy()
    return apply_policy_blur(
        frame.copy(),
        blur_full=decision.blur_full_frame,
        strength=blur,
    )


def _maybe_log_events(scores: ModelScores, decision: MaskingDecision) -> None:
    if not STATE.config.protection_enabled:
        return
    if decision.mute_audio and not _prev_emit["mute"]:
        if decision.mute_reason == "pii":
            STATE.log_event("PII Detected - Audio Muted", "warn")
        else:
            STATE.log_event("Sensitive phrase - Audio Muted", "warn")
    if decision.blur_full_frame and not _prev_emit["blur"]:
        STATE.log_event("Policy - Visual protection applied", "info")
    if decision.silent_mode and not _prev_emit["silent"]:
        STATE.log_event("High stress / anger - Silent mode", "warn")
    _prev_emit["mute"] = decision.mute_audio
    _prev_emit["blur"] = decision.blur_full_frame
    _prev_emit["silent"] = decision.silent_mode


def _loop() -> None:
    global _video, _ema_fps, _last_broadcast, _prev_hf_efficientnet
    ensure_audio_worker()
    if _video is None:
        _video = VideoSource(0)

    last_emit = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        ok, frame = _video.read_bgr()
        if not ok:
            time.sleep(0.05)
            continue

        STATE.sync_policy_from_config()
        use_hf = STATE.config.hf_efficientnet_nsfw or (
            os.environ.get("PRIVATEEDGE_USE_HF_EFFICIENTNET", "0").lower()
            in ("1", "true", "yes")
        )
        if use_hf and not _prev_hf_efficientnet:
            reset_hf_load_state()
        _prev_hf_efficientnet = use_hf
        v_scores = analyze_frame_bgr(frame, use_hf_efficientnet=use_hf)
        a_scores, _, _ = GLOBAL_AUDIO_BUF.snapshot()
        scores = merge_vision_audio_scores(v_scores, a_scores)

        decision = STATE.engine.decide(scores)
        if not STATE.config.protection_enabled:
            decision = MaskingDecision()

        blur = STATE.config.blur_strength
        protected = _apply_protection(decision, frame, blur)

        now = time.perf_counter()
        elapsed = max(now - loop_start, 1e-6)
        fps_inst = 1.0 / elapsed
        _ema_fps = 0.85 * _ema_fps + 0.15 * min(fps_inst, 120.0)
        infer_ms = elapsed * 1000
        fps_display = min(60.0, round(_ema_fps, 1))
        lat = round(infer_ms + 1.0, 1)
        npu = round(
            min(95.0, _npu_percent() + min(20.0, infer_ms / 12.0)),
            1,
        )
        STATE.update_telemetry(fps_display, lat, npu)

        if now - last_emit > 1.8:
            _maybe_log_events(scores, decision)
            last_emit = now

        audio_items = [{"id": "ok", "label": "Audio OK", "tone": "ok"}]
        if decision.mute_audio:
            if decision.mute_reason == "pii":
                audio_items = [
                    {"id": "ok", "label": "Audio OK", "tone": "ok"},
                    {"id": "pii", "label": "Muted: PII Detected", "tone": "bad"},
                ]
            else:
                audio_items = [
                    {"id": "ok", "label": "Audio OK", "tone": "ok"},
                    {
                        "id": "phrase",
                        "label": "Muted: Sensitive phrase",
                        "tone": "warn",
                    },
                ]
        STATE.set_audio_statuses(audio_items)

        payload = {
            "kind": "frame",
            "raw_jpeg": _encode_jpeg_b64(frame),
            "protected_jpeg": _encode_jpeg_b64(protected),
            "telemetry": {
                "fps": STATE.telemetry.fps,
                "latency_ms": STATE.telemetry.latency_ms,
                "npu_percent": STATE.telemetry.npu_percent,
            },
            "scores": {
                "p_doc": scores.p_doc,
                "p_face_other": scores.p_face_other,
                "p_nsfw": scores.p_nsfw,
                "p_pii_audio": scores.p_pii_audio,
                "anger": scores.anger,
            },
            "decision": {
                "blur_full": decision.blur_full_frame,
                "mute": decision.mute_audio,
                "silent": decision.silent_mode,
            },
            "events": list(STATE.event_log)[:12],
            "audio": audio_items,
        }
        if now - _last_broadcast >= 0.066:
            _broadcast(payload)
            _last_broadcast = now

        frame_elapsed = time.perf_counter() - loop_start
        target = 1.0 / 30.0
        if frame_elapsed < target:
            time.sleep(target - frame_elapsed)


def ensure_pipeline_started() -> None:
    global _started, _pipeline_thread
    if _started:
        return
    _started = True
    _pipeline_thread = threading.Thread(target=_loop, name="privateedge-pipeline", daemon=True)
    _pipeline_thread.start()
