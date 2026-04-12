from __future__ import annotations

import threading

import numpy as np


class DelayRingBuffer:
    """
    Mono float32 ring keyed by monotonic global sample index.
    Writer (capture), reader (playback), and processor (bleep overlay) share one lock.
    """

    def __init__(self, capacity: int, delay_samples: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if delay_samples <= 0:
            raise ValueError("delay_samples must be positive")
        self.capacity = capacity
        self.delay_samples = delay_samples
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._lock = threading.Lock()
        self._write_global = 0
        self._read_global = 0
        self._primed = False

    def write(self, block: np.ndarray) -> None:
        """Append mono float32 samples; may overwrite oldest if capacity exceeded."""
        if block.ndim != 1:
            raise ValueError("expected 1-D mono audio")
        block = np.asarray(block, dtype=np.float32)
        with self._lock:
            n = len(block)
            wg = self._write_global
            idx = (np.arange(wg, wg + n, dtype=np.int64) % self.capacity).astype(np.intp)
            self._buf[idx] = block
            self._write_global += n
            max_lag = self._write_global - self._read_global
            if max_lag > self.capacity:
                self._read_global = self._write_global - self.capacity

    def read_for_output(self, frames: int) -> np.ndarray:
        """Return `frames` samples for playback (silence until primed)."""
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            if not self._primed and self._write_global >= self.delay_samples:
                self._primed = True

            if not self._primed:
                return out

            g = np.arange(self._read_global, self._read_global + frames, dtype=np.int64)
            mask = g < self._write_global
            if np.any(mask):
                idx = (g[mask] % self.capacity).astype(np.intp)
                out[mask] = self._buf[idx]
            self._read_global += frames
        return out

    def copy_range(self, global_start: int, length: int) -> np.ndarray | None:
        """Copy contiguous global samples if fully written; else None."""
        if length <= 0:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            if self._write_global < global_start + length:
                return None
            g = np.arange(global_start, global_start + length, dtype=np.int64)
            idx = (g % self.capacity).astype(np.intp)
            return self._buf[idx].copy()

    def replace_range(self, global_start: int, samples: np.ndarray) -> None:
        """Overwrite ring samples; only safe if playback has not passed global_start."""
        samples = np.asarray(samples, dtype=np.float32)
        with self._lock:
            if global_start < self._read_global:
                return
            n = len(samples)
            if n == 0:
                return
            end_g = min(global_start + n, self._write_global)
            m = end_g - global_start
            if m <= 0:
                return
            part = samples[:m]
            g = np.arange(global_start, global_start + m, dtype=np.int64)
            idx = (g % self.capacity).astype(np.intp)
            self._buf[idx] = part

    @property
    def write_global(self) -> int:
        with self._lock:
            return self._write_global

    @property
    def read_global(self) -> int:
        with self._lock:
            return self._read_global
