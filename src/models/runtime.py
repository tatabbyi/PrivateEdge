from __future__ import annotations

from pathlib import Path
from typing import Any


def preferred_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001
        return []
    return [
        p
        for p in ("QNNExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        if p in ort.get_available_providers()
    ]


def create_inference_session(model_path: Path) -> Any:
    import onnxruntime as ort

    path = Path(model_path)
    providers: list[str | tuple[str, dict[Any, Any]]] = preferred_providers()
    if not providers:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001
        return []
    return list(ort.get_available_providers())
