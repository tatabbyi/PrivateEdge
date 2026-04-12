from __future__ import annotations

import numpy as np


def apply_policy_blur(
    frame_bgr: np.ndarray,
    *,
    blur_full: bool,
    strength: float,
) -> np.ndarray:
    """Apply a simple Gaussian blur to the full frame when policy requests it."""
    if not blur_full or frame_bgr.size == 0:
        return frame_bgr
    import cv2

    s = max(0.0, min(1.0, float(strength)))
    if s < 0.02:
        return frame_bgr

    # Nonlinear ramp so top-end slider values produce visibly stronger censoring.
    s_nl = s**0.6
    k = int(5 + 70 * s_nl)
    if k % 2 == 0:
        k += 1
    k = max(5, min(k, 95))
    out = cv2.GaussianBlur(frame_bgr, (k, k), 0)

    if s >= 0.75:
        k2 = min(151, k + 24)
        if k2 % 2 == 0:
            k2 += 1
        out = cv2.GaussianBlur(out, (k2, k2), 0)

    # At max strength, add pixelation to make censoring unmistakable.
    if s >= 0.95:
        h, w = out.shape[:2]
        dw = max(24, w // 10)
        dh = max(24, h // 10)
        out = cv2.resize(out, (dw, dh), interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

    return out
