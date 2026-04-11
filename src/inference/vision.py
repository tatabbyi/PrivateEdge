"""OpenCV vision signals; optional ONNX NSFW when models/nsfw.onnx exists."""

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


def analyze_frame_bgr(frame: np.ndarray) -> ModelScores:
    """Compute vision-only scores from a BGR frame."""
    p_doc = _document_likelihood(frame)
    p_face_other = _face_other_score(frame)

    sess = _load_nsfw_session()
    if sess is not None:
        p_nsfw = _run_nsfw_onnx(sess, frame)
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
