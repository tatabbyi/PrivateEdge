"""REST control plane: config, status, event logs."""

from __future__ import annotations

import platform
import sys
from typing import Any

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework.views import APIView

from capture.video import list_video_devices
from inference.audio_worker import list_input_devices, list_output_devices
from inference.vision import nsfw_runtime_info
from services.state import STATE


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("0", "false", "no", "off", ""):
            return False
        return s in ("1", "true", "yes", "on")
    return bool(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "none", "auto", "default"):
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@method_decorator(csrf_exempt, name="dispatch")
class ConfigView(APIView):
    """GET/PATCH runtime configuration (toggles, sliders, mode)."""

    def get(self, request: Any) -> Response:
        STATE.sync_policy_from_config()
        return Response({"config": STATE.to_public_dict()["config"]})

    def patch(self, request: Any) -> Response:
        data = request.data
        cfg = STATE.config
        bool_keys = frozenset(
            {
                "face_masking",
                "text_document_blocking",
                "nsfw_detection",
                "audio_pii_filtering",
                "protection_enabled",
                "webcam_enabled",
                "screen_share_enabled",
                "virtual_webcam_enabled",
                "virtual_screenshare_enabled",
                "virtual_audio_enabled",
                "hf_efficientnet_nsfw",
            }
        )
        for key in (
            "face_masking",
            "text_document_blocking",
            "nsfw_detection",
            "audio_pii_filtering",
            "mode",
            "detection_sensitivity",
            "detection_sensitivity_secondary",
            "blur_strength",
            "blur_strength_secondary",
            "mute_sensitivity",
            "protection_enabled",
            "webcam_enabled",
            "screen_share_enabled",
            "webcam_index",
            "mic_device_index",
            "virtual_webcam_enabled",
            "virtual_screenshare_enabled",
            "virtual_audio_enabled",
            "virtual_webcam_device_name",
            "virtual_screenshare_device_name",
            "virtual_audio_output_device",
            "hf_efficientnet_nsfw",
        ):
            if key not in data:
                continue
            if key in bool_keys:
                setattr(cfg, key, _as_bool(data[key]))
            elif key in ("webcam_index", "mic_device_index"):
                setattr(cfg, key, _as_optional_int(data[key]))
            else:
                setattr(cfg, key, data[key])
        STATE.sync_policy_from_config()
        return Response({"ok": True, "config": STATE.to_public_dict()["config"]})


class StatusView(APIView):
    """Latest telemetry (FPS, latency, NPU)."""

    def get(self, request: Any) -> Response:
        return Response(STATE.to_public_dict()["telemetry"])


class LogsView(APIView):
    """Recent non-raw event log for dashboard."""

    def get(self, request: Any) -> Response:
        return Response({"events": list(STATE.event_log)})


class HealthView(APIView):
    def get(self, request: Any) -> Response:
        return Response({"status": "ok", "service": "privateedge-api"})


class DevicesView(APIView):
    """Available capture input devices for runtime selection."""

    def get(self, request: Any) -> Response:
        return Response(
            {
                "video_inputs": list_video_devices(),
                "audio_inputs": list_input_devices(),
                "audio_outputs": list_output_devices(),
            }
        )


class RuntimeView(APIView):
    """Runtime diagnostics for x86/dev and Snapdragon validation."""

    def get(self, request: Any) -> Response:
        return Response(
            {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "nsfw_pipeline": nsfw_runtime_info(),
            }
        )
