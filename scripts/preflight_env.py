from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def _check_onnx(model_path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "onnxruntime_installed": False,
        "providers": [],
        "qnn_available": False,
        "model_exists": model_path.is_file(),
        "session_ok": False,
        "session_provider": None,
        "error": None,
    }
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        result["onnxruntime_installed"] = True
        result["providers"] = providers
        result["qnn_available"] = "QNNExecutionProvider" in providers

        if model_path.is_file():
            preferred = [
                p
                for p in (
                    "QNNExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                )
                if p in providers
            ]
            sess = ort.InferenceSession(str(model_path), providers=preferred or providers)
            active = sess.get_providers()
            result["session_ok"] = True
            result["session_provider"] = active[0] if active else None
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="PrivateEdge environment preflight")
    parser.add_argument(
        "--model",
        default="models/nsfw.onnx",
        help="Path to ONNX model used for readiness check",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = (root / args.model).resolve()

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model_path": str(model_path),
        "onnx": _check_onnx(model_path),
    }

    print(json.dumps(report, indent=2))

    # Non-zero exit only for clear setup blockers.
    if not report["onnx"]["onnxruntime_installed"]:
        return 2
    if not report["onnx"]["model_exists"]:
        return 3
    if not report["onnx"]["session_ok"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
