from __future__ import annotations

import logging

import numpy as np

_log = logging.getLogger(__name__)


class ScreenSource:
    """Primary monitor capture via mss; falls back when unavailable."""

    def __init__(self) -> None:
        self._sct = None
        self.last_was_fallback: bool = True

    def _mss(self):
        if self._sct is None:
            try:
                import mss

                self._sct = mss.mss()
            except Exception as exc:  # noqa: BLE001
                _log.debug("mss unavailable: %s", exc)
                self._sct = False
        return self._sct if self._sct else None

    def read_bgr(self) -> tuple[bool, np.ndarray]:
        sct = self._mss()
        if sct is None:
            self.last_was_fallback = True
            return False, np.zeros((400, 640, 3), dtype=np.uint8)
        try:
            import cv2

            mon = sct.monitors[1]
            raw = sct.grab(mon)
            # mss pixels are BGRA; first three channels are BGR.
            frame = np.asarray(raw, dtype=np.uint8)[:, :, :3].copy()
            h, w = frame.shape[:2]
            if h > 720 or w > 1280:
                scale = min(720 / h, 1280 / w)
                frame = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            self.last_was_fallback = False
            return True, frame
        except Exception as exc:  # noqa: BLE001
            _log.debug("Screen grab failed: %s", exc)
            self.last_was_fallback = True
            return False, np.zeros((400, 640, 3), dtype=np.uint8)
