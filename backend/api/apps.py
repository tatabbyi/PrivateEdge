import logging

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"
    verbose_name = "PrivateEdge API"

    def ready(self) -> None:
        # Quieter console: faster-whisper logs every VAD chunk at INFO by default.
        for name in ("faster_whisper", "httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)

        from services.pipeline import ensure_pipeline_started

        ensure_pipeline_started()
