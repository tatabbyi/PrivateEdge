"""OpenCV vision signals with ONNX-only NSFW scoring."""

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
_mp_hands: Any = None
_mp_hands_tried: bool = False


def reset_hf_load_state() -> None:
    """Compatibility no-op: HF path is intentionally disabled."""
    return None


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


def _softmax(values: np.ndarray) -> np.ndarray:
    e = np.exp(values - np.max(values))
    return e / np.sum(e)


def _load_hands_detector() -> Any:
    global _mp_hands, _mp_hands_tried
    if _mp_hands_tried:
        return _mp_hands
    _mp_hands_tried = True
    try:
        import mediapipe as mp

        _mp_hands = mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2,
        )
    except Exception as e:  # noqa: BLE001
        logger.info("mediapipe hands unavailable: %s", e)
        _mp_hands = None
    return _mp_hands


def _is_middle_finger_extended(hand_landmarks: Any) -> bool:
    middle_tip_y = hand_landmarks.landmark[12].y
    middle_pip_y = hand_landmarks.landmark[10].y
    index_tip_y = hand_landmarks.landmark[8].y
    index_pip_y = hand_landmarks.landmark[6].y
    ring_tip_y = hand_landmarks.landmark[16].y
    ring_pip_y = hand_landmarks.landmark[14].y
    pinky_tip_y = hand_landmarks.landmark[20].y
    pinky_pip_y = hand_landmarks.landmark[18].y

    middle_extended = middle_tip_y < middle_pip_y
    others_folded = (
        index_tip_y > index_pip_y
        and ring_tip_y > ring_pip_y
        and pinky_tip_y > pinky_pip_y
    )
    return bool(middle_extended and others_folded)


def _obscene_gesture_score(frame: np.ndarray) -> float:
    import cv2

    hands = _load_hands_detector()
    if hands is None:
        return 0.0
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
    except Exception as e:  # noqa: BLE001
        logger.debug("Hands inference failed: %s", e)
        return 0.0

    if not results or not results.multi_hand_landmarks:
        return 0.0
    for hlm in results.multi_hand_landmarks:
        if _is_middle_finger_extended(hlm):
            return 1.0
    return 0.0


def _run_nsfw_onnx(sess: Any, frame: np.ndarray) -> float:
    import cv2

    try:
        inp = sess.get_inputs()[0]
        name = inp.name
        shape = inp.shape
        ih, iw = 224, 224
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            ih, iw = int(shape[2]), int(shape[3])
        family = os.environ.get("PRIVATEEDGE_NSFW_MODEL_FAMILY", "auto").strip().lower()

        x = cv2.resize(frame, (iw, ih)).astype(np.float32)
        if family in ("yahoo", "yahoo_open_nsfw", "open_nsfw"):
            # Yahoo OpenNSFW-style preprocessing: BGR uint8-like range with mean subtraction.
            x = x - np.array([104.0, 117.0, 123.0], dtype=np.float32)
        else:
            # Generic transformer/EVA path.
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            x = (x - mean) / std

        x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
        out = sess.run(None, {name: x})[0].flatten()
        if out.size == 1:
            v = float(out[0])
            # Handle either probability or raw logit.
            if 0.0 <= v <= 1.0:
                return v
            return float(1.0 / (1.0 + np.exp(-v)))
        if out.size == 2:
            p = _softmax(out)
            return float(p[1] if len(p) > 1 else p[0])
        if out.size == 4:
            # Freepik EVA convention: [neutral, low, medium, high].
            p = _softmax(out)
            neutral, low, medium, high = [float(v) for v in p]
            # Keep low-impact content from over-triggering while strongly reacting to medium/high.
            score = high + 0.95 * medium + 0.2 * low - 0.3 * neutral
            if neutral >= 0.55 and high < 0.15:
                score *= 0.5
            return float(max(0.0, min(1.0, score)))
        if out.size >= 5:
            # Common 5-class NSFW order: drawings, hentai, neutral, porn, sexy.
            p = _softmax(out)
            drawings, hentai, neutral, porn, sexy = [float(v) for v in p[:5]]
            raw_nsfw = porn + 0.55 * sexy + 0.35 * hentai
            safety = neutral + 0.35 * drawings
            score = raw_nsfw - 0.6 * safety
            if neutral >= 0.45 and porn < 0.30:
                score *= 0.5
            return float(max(0.0, min(1.0, score * 1.35)))
        return float(max(0.0, min(1.0, float(out[0]))))
    except Exception as e:  # noqa: BLE001
        logger.debug("NSFW ONNX inference failed: %s", e)
        return 0.0


def analyze_frame_bgr(
    frame: np.ndarray,
    *,
    use_hf_efficientnet: bool = False,
) -> ModelScores:
    """Compute vision-only scores from a BGR frame (ONNX-only NSFW path)."""
    p_doc = _document_likelihood(frame)
    p_face_other = _face_other_score(frame)
    p_obscene_gesture = _obscene_gesture_score(frame)

    sess = _load_nsfw_session()
    if sess is not None:
        p_nsfw = _run_nsfw_onnx(sess, frame)
    else:
        # Competition-safe deterministic path: NSFW comes only from local ONNX.
        p_nsfw = 0.0

    return ModelScores(
        p_doc=p_doc,
        p_face_other=p_face_other,
        p_nsfw=p_nsfw,
        p_obscene_gesture=p_obscene_gesture,
        p_pii_audio=0.0,
        p_toxicity=0.0,
        anger=0.0,
        stress=0.0,
    )


def nsfw_runtime_info() -> dict[str, Any]:
    model_path = Path(
        os.environ.get(
            "PRIVATEEDGE_NSFW_ONNX",
            str(Path(__file__).resolve().parents[2] / "models" / "nsfw.onnx"),
        )
    )
    session = _load_nsfw_session()
    providers: list[str] = []
    if session is not None:
        try:
            providers = list(session.get_providers())
        except Exception:  # noqa: BLE001
            providers = []

    from models.runtime import available_providers, preferred_providers

    return {
        "onnx_model_path": str(model_path),
        "onnx_model_exists": model_path.is_file(),
        "onnx_session_loaded": session is not None,
        "onnx_session_providers": providers,
        "ort_available_providers": available_providers(),
        "preferred_provider_order": preferred_providers() or ["CPUExecutionProvider"],
        "hf_loaded": False,
        "hf_load_failed": True,
        "fallbacks_enabled": False,
    }
