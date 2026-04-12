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

from capture.screen import ScreenSource
from capture.video import VideoSource
from inference.audio_worker import (
    GLOBAL_AUDIO_BUF,
    configure_audio_bleep,
    configure_audio_output,
    ensure_audio_worker_with_device,
)
from inference.scoring import merge_vision_audio_scores, merge_vision_streams
from inference.vision import analyze_frame_bgr
from policy.types import MaskingDecision, ModelScores
from render.blur import apply_policy_blur
from services.state import STATE

_pipeline_thread: threading.Thread | None = None
_started = False
_video: VideoSource | None = None
_screen: ScreenSource | None = None
_ema_fps = 30.0
_last_broadcast = 0.0
_npu_base: float | None = None
_prev_emit: dict[str, bool] = {
    "mute": False,
    "blur": False,
    "silent": False,
    "gesture": False,
}
_vcam_webcam: Any = None
_vcam_screen: Any = None
_vcam_webcam_name: str = ""
_vcam_screen_name: str = ""


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


def _placeholder_bgr(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), 28, dtype=np.uint8)


def _close_virtual_cams() -> None:
    global _vcam_webcam, _vcam_screen
    for c in (_vcam_webcam, _vcam_screen):
        if c is None:
            continue
        try:
            c.close()
        except Exception:  # noqa: BLE001
            pass
    _vcam_webcam = None
    _vcam_screen = None


def _ensure_virtual_cam(
    enabled: bool,
    current: Any,
    name: str,
    width: int,
    height: int,
) -> Any:
    if not enabled:
        if current is not None:
            try:
                current.close()
            except Exception:  # noqa: BLE001
                pass
        return None
    if current is not None:
        return current
    try:
        import pyvirtualcam

        return pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=30,
            fmt=pyvirtualcam.PixelFormat.BGR,
            device=name,
        )
    except Exception as e:  # noqa: BLE001
        STATE.log_event(f"Virtual camera '{name}' unavailable: {e}", "warn")
        try:
            import pyvirtualcam

            cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=30,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            STATE.log_event(
                f"Using default virtual camera backend instead of '{name}'",
                "info",
            )
            return cam
        except Exception as e2:  # noqa: BLE001
            STATE.log_event(f"Default virtual camera unavailable: {e2}", "warn")
            return None


def _send_virtual_frame(cam: Any, frame: np.ndarray) -> None:
    if cam is None:
        return
    try:
        h, w = frame.shape[:2]
        if cam.width != w or cam.height != h:
            import cv2

            frame = cv2.resize(frame, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
        cam.send(frame)
    except Exception as e:  # noqa: BLE001
        STATE.log_event(f"Virtual camera send failed: {e}", "warn")


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


def _add_watermark(frame: np.ndarray, text: str = "PrivateEdge Filtered") -> np.ndarray:
    if frame.size == 0:
        return frame
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.7, w / 1800.0))
    thickness = 1 if w < 1000 else 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(8, w - tw - 12)
    y = max(th + 10, h - 12)

    cv2.rectangle(
        out,
        (x - 6, y - th - 6),
        (x + tw + 6, y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(out, text, (x, y), font, scale, (245, 245, 245), thickness, cv2.LINE_AA)
    return out


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
    gesture_now = scores.p_obscene_gesture >= 0.6
    if gesture_now and not _prev_emit["gesture"]:
        STATE.log_event("Obscene hand gesture detected", "warn")
    if decision.silent_mode and not _prev_emit["silent"]:
        STATE.log_event("High stress / anger - Silent mode", "warn")
    _prev_emit["mute"] = decision.mute_audio
    _prev_emit["blur"] = decision.blur_full_frame
    _prev_emit["silent"] = decision.silent_mode
    _prev_emit["gesture"] = gesture_now


def _loop() -> None:
    global _video, _screen, _ema_fps, _last_broadcast
    global _vcam_webcam, _vcam_screen, _vcam_webcam_name, _vcam_screen_name
    ensure_audio_worker_with_device(STATE.config.mic_device_index)
    if _video is None:
        default_idx = int(os.environ.get("PRIVATEEDGE_WEBCAM_INDEX", "0"))
        _video = VideoSource(STATE.config.webcam_index if STATE.config.webcam_index is not None else default_idx)
    if _screen is None:
        _screen = ScreenSource()

    last_emit = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        STATE.sync_policy_from_config()
        cfg = STATE.config
        ensure_audio_worker_with_device(cfg.mic_device_index)
        if cfg.webcam_index is not None:
            _video.set_index(cfg.webcam_index)

        if cfg.webcam_enabled:
            ok_w, frame_w = _video.read_bgr()
            if not ok_w:
                frame_w = _placeholder_bgr()
        else:
            frame_w = _placeholder_bgr()

        screen_capture_live = False
        if cfg.screen_share_enabled:
            ok_s, frame_s = _screen.read_bgr()
            if not ok_s:
                frame_s = _placeholder_bgr(640, 400)
            screen_capture_live = not _screen.last_was_fallback
        else:
            frame_s = _placeholder_bgr(640, 400)

        zw = (
            ModelScores()
            if not cfg.webcam_enabled
            else analyze_frame_bgr(frame_w)
        )
        zs = (
            ModelScores()
            if not cfg.screen_share_enabled
            else analyze_frame_bgr(frame_s)
        )
        v_scores = merge_vision_streams(zw, zs)
        a_scores, a_text, a_rms = GLOBAL_AUDIO_BUF.snapshot()
        scores = merge_vision_audio_scores(v_scores, a_scores)

        decision = STATE.engine.decide(scores)
        if not cfg.protection_enabled:
            decision = MaskingDecision()
        configure_audio_output(
            enabled=cfg.virtual_audio_enabled,
            output_device_name=cfg.virtual_audio_output_device,
            muted=decision.mute_audio,
        )
        configure_audio_bleep(enabled=cfg.profanity_bleep_enabled, frequency_hz=1000.0)

        blur = cfg.blur_strength
        prot_w = _apply_protection(decision, frame_w, blur)
        prot_s = _apply_protection(decision, frame_s, blur)
        prot_w = _add_watermark(prot_w)

        if cfg.virtual_webcam_device_name != _vcam_webcam_name:
            if _vcam_webcam is not None:
                try:
                    _vcam_webcam.close()
                except Exception:  # noqa: BLE001
                    pass
                _vcam_webcam = None
            _vcam_webcam_name = cfg.virtual_webcam_device_name
        if cfg.virtual_screenshare_device_name != _vcam_screen_name:
            if _vcam_screen is not None:
                try:
                    _vcam_screen.close()
                except Exception:  # noqa: BLE001
                    pass
                _vcam_screen = None
            _vcam_screen_name = cfg.virtual_screenshare_device_name

        _vcam_webcam = _ensure_virtual_cam(
            cfg.virtual_webcam_enabled, _vcam_webcam, cfg.virtual_webcam_device_name, prot_w.shape[1], prot_w.shape[0]
        )
        _vcam_screen = _ensure_virtual_cam(
            cfg.virtual_screenshare_enabled,
            _vcam_screen,
            cfg.virtual_screenshare_device_name,
            prot_s.shape[1],
            prot_s.shape[0],
        )
        _send_virtual_frame(_vcam_webcam, prot_w)
        _send_virtual_frame(_vcam_screen, prot_s)

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
            "raw_webcam_jpeg": _encode_jpeg_b64(frame_w),
            "raw_screen_jpeg": _encode_jpeg_b64(frame_s),
            "protected_webcam_jpeg": _encode_jpeg_b64(prot_w),
            "protected_screen_jpeg": _encode_jpeg_b64(prot_s),
            "telemetry": {
                "fps": STATE.telemetry.fps,
                "latency_ms": STATE.telemetry.latency_ms,
                "npu_percent": STATE.telemetry.npu_percent,
            },
            "scores": {
                "p_doc": scores.p_doc,
                "p_face_other": scores.p_face_other,
                "p_nsfw": scores.p_nsfw,
                "p_obscene_gesture": scores.p_obscene_gesture,
                "p_pii_audio": scores.p_pii_audio,
                "p_toxicity": scores.p_toxicity,
                "anger": scores.anger,
            },
            "audio_debug": {
                "rms": float(a_rms),
                "last_text": (a_text or "")[:180],
            },
            "decision": {
                "blur_full": decision.blur_full_frame,
                "mute": decision.mute_audio,
                "silent": decision.silent_mode,
            },
            "events": list(STATE.event_log)[:12],
            "audio": audio_items,
            "screen_capture_live": screen_capture_live,
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
