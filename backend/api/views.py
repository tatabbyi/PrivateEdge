"""REST control plane: config, status, event logs."""

from __future__ import annotations

from typing import Any

from rest_framework.response import Response
from rest_framework.views import APIView

from services.state import STATE


class ConfigView(APIView):
    """GET/PATCH runtime configuration (toggles, sliders, mode)."""

    def get(self, request: Any) -> Response:
        STATE.sync_policy_from_config()
        return Response({"config": STATE.to_public_dict()["config"]})

    def patch(self, request: Any) -> Response:
        data = request.data
        cfg = STATE.config
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
        ):
            if key in data:
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
