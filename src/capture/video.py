from __future__ import annotations

import logging

import cv2
import numpy as np

_log = logging.getLogger(__name__)


class VideoSource:
    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._cap: cv2.VideoCapture | None = None

    @property
    def index(self) -> int:
        return self._index

    def set_index(self, index: int) -> None:
        if index == self._index:
            return
        self._index = index
        self._close()

    def _ensure_open(self) -> cv2.VideoCapture | None:
        if self._cap is not None and self._cap.isOpened():
            return self._cap
        self._close()
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            _log.warning("Could not open webcam index %s", self._index)
            return None
        self._cap = cap
        return cap

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_bgr(self) -> tuple[bool, np.ndarray]:
        cap = self._ensure_open()
        if cap is None:
            return False, np.zeros((480, 640, 3), dtype=np.uint8)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return False, np.zeros((480, 640, 3), dtype=np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return True, frame


def list_video_devices(max_index: int = 8) -> list[dict[str, object]]:
    devices: list[dict[str, object]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        shape = tuple(frame.shape) if ok and frame is not None else None
        devices.append(
            {
                "id": i,
                "label": f"Camera {i}" + (f" ({shape[1]}x{shape[0]})" if shape else ""),
            }
        )
        cap.release()
    return devices
