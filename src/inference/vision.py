"""OpenCV vision signals; optional ONNX NSFW or Hugging Face EfficientNet."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from policy.types import ModelScores

logger = logging.getLogger(__name__)

_face_cascade: Any = None
_nsfw_sess: Any = None
_nsfw_tried: bool = False

_hf_processor: Any = None
_hf_model: Any = None
_hf_load_failed: bool = False


def reset_hf_load_state() -> None:
    """Clear a failed HF load so the next frame can retry (e.g. after UI toggles on)."""
    global _hf_load_failed
    _hf_load_failed = False


def _get_face_cascade() -> Any:
    global _face_cascade
    if _face_cascade is None:
        import cv2

        p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(p)
        if _face_cascade.empty():
            logger.error("Failed to load Haar cascade from %s", p)
    return _face_cascade


def _load_nsfw_session() -> Any:
    global _nsfw_sess, _nsfw_tried
    if _nsfw_tried:
        return _nsfw_sess
    _nsfw_tried = True
    path = os.environ.get(
        "PRIVATEEDGE_NSFW_ONNX",
        str(Path(__file__).resolve().parents[2] / "models" / "nsfw.onnx"),
    )
    p = Path(path)
    if not p.is_file():
        _nsfw_sess = None
        return None
    try:
        from models.runtime import create_inference_session

        _nsfw_sess = create_inference_session(p)
    except Exception as e:  # noqa: BLE001
        logger.warning("NSFW ONNX not loaded: %s", e)
        _nsfw_sess = None
    return _nsfw_sess


def _document_likelihood(frame: np.ndarray) -> float:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = frame.shape[:2]
    area_f = float(h * w)
    best = 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a < area_f * 0.02 or a > area_f * 0.85:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1e-6:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) < 4:
            continue
        rw, rh = cv2.boundingRect(approx)[2:]
        ar = rw / max(1, rh)
        if 0.4 < ar < 2.5:
            rect_score = min(1.0, a / area_f) * 0.8
            best = max(best, rect_score)
    return float(min(1.0, best * 1.4))


def _face_other_score(frame: np.ndarray) -> float:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fc = _get_face_cascade()
    if fc is None or fc.empty():
        return 0.0
    faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return 0.0
    h, w = frame.shape[:2]
    areas = sorted((fw * fh for (x, y, fw, fh) in faces), reverse=True)
    if len(areas) == 1:
        return min(1.0, areas[0] / (h * w) * 0.35)
    primary = areas[0]
    others = sum(areas[1:])
    return float(min(1.0, (others / (h * w)) * 2.5))


def _skin_proxy_nsfw(frame: np.ndarray) -> float:
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 180, 255]))
    ratio = float(np.count_nonzero(mask)) / mask.size
    return float(min(1.0, max(0.0, (ratio - 0.15) * 2.0)))


def _load_hf_efficientnet() -> tuple[Any, Any]:
    """Lazy-load HF EfficientNet + image processor (once per process if successful)."""
    global _hf_processor, _hf_model, _hf_load_failed
    if _hf_processor is not None and _hf_model is not None:
        return _hf_processor, _hf_model
    if _hf_load_failed:
        return None, None
    model_id = os.environ.get(
        "PRIVATEEDGE_HF_MODEL_ID",
        "Nafi007/EfficientNetB0",
    )
    try:
        import torch
        from transformers import AutoImageProcessor, EfficientNetForImageClassification
    except ImportError as e:
        logger.warning("HF EfficientNet skipped (install torch+transformers): %s", e)
        _hf_load_failed = True
        return None, None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _hf_processor = AutoImageProcessor.from_pretrained(model_id)
        _hf_model = EfficientNetForImageClassification.from_pretrained(model_id)
        _hf_model.to(device)
        _hf_model.eval()
        logger.info("HF EfficientNet NSFW loaded: %s on %s", model_id, device)
    except Exception as e:  # noqa: BLE001
        logger.warning("HF EfficientNet not loaded: %s", e)
        _hf_processor = None
        _hf_model = None
        _hf_load_failed = True
    return _hf_processor, _hf_model


def _probs_to_p_nsfw(probs: np.ndarray, id2label: dict[Any, Any]) -> float:
    """Map classifier probs to a single [0,1] NSFW-style score using id2label."""
    labels: dict[int, str] = {}
    for k, v in id2label.items():
        try:
            labels[int(k)] = str(v).lower()
        except (TypeError, ValueError):
            continue
    nsfw_kw = ("nsfw", "porn", "explicit", "hentai", "nude", "sexual", "unsafe", "adult")
    safe_kw = ("safe", "sfw", "neutral", "drawing", "drawings", "normal")
    nsfw_mass = 0.0
    for i, p in enumerate(probs):
        lab = labels.get(i, "")
        if any(w in lab for w in nsfw_kw):
            nsfw_mass += float(p)
        elif any(w in lab for w in safe_kw):
            continue
        elif len(probs) == 2:
            # Binary: assume index 1 is the positive / sensitive class if unlabeled
            if i == 1:
                nsfw_mass += float(p)
    if nsfw_mass > 0.0:
        return float(min(1.0, nsfw_mass))
    if len(probs) == 2:
        return float(probs[1])
    return float(min(1.0, float(np.max(probs))))


def _run_nsfw_hf(frame: np.ndarray) -> float:
    import cv2
    from PIL import Image

    proc, mdl = _load_hf_efficientnet()
    if proc is None or mdl is None:
        return _skin_proxy_nsfw(frame)
    try:
        import torch
    except ImportError:
        return _skin_proxy_nsfw(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = proc(images=image, return_tensors="pt")
    device = next(mdl.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
    id2label = getattr(mdl.config, "id2label", None) or {}
    return _probs_to_p_nsfw(probs, id2label)


def _run_nsfw_onnx(sess: Any, frame: np.ndarray) -> float:
    import cv2

    try:
        inp = sess.get_inputs()[0]
        name = inp.name
        shape = inp.shape
        ih, iw = 224, 224
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            ih, iw = int(shape[2]), int(shape[3])
        x = cv2.resize(frame, (iw, ih))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
        out = sess.run(None, {name: x})[0].flatten()
        if out.size >= 2:
            e = np.exp(out - np.max(out))
            p = e / np.sum(e)
            return float(p[1] if len(p) > 1 else p[0])
        return float(max(0.0, min(1.0, float(out[0]))))
    except Exception as e:  # noqa: BLE001
        logger.debug("NSFW ONNX inference fallback: %s", e)
        return _skin_proxy_nsfw(frame)


def analyze_frame_bgr(
    frame: np.ndarray,
    *,
    use_hf_efficientnet: bool = False,
) -> ModelScores:
    """Compute vision-only scores from a BGR frame.

    When ``use_hf_efficientnet`` is True (e.g. UI toggle), run HF EfficientNet if
    no ONNX NSFW session is available. Env ``PRIVATEEDGE_USE_HF_EFFICIENTNET``
    can also be set; the pipeline merges that with runtime config.
    """
    p_doc = _document_likelihood(frame)
    p_face_other = _face_other_score(frame)

    sess = _load_nsfw_session()
    if sess is not None:
        p_nsfw = _run_nsfw_onnx(sess, frame)
    elif use_hf_efficientnet:
        p_nsfw = _run_nsfw_hf(frame)
    else:
        p_nsfw = _skin_proxy_nsfw(frame)

    return ModelScores(
        p_doc=p_doc,
        p_face_other=p_face_other,
        p_nsfw=p_nsfw,
        p_pii_audio=0.0,
        p_toxicity=0.0,
        anger=0.0,
        stress=0.0,
    )
