from __future__ import annotations

import numpy as np


def generate_bleep(num_samples: int, sample_rate: int, frequency_hz: float = 1000.0) -> np.ndarray:
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    tone = np.sin(2.0 * np.pi * frequency_hz * t).astype(np.float32)
    # Short fade in/out to reduce clicks
    fade = min(64, num_samples // 4 or 1)
    if fade > 0 and num_samples > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]
    # Keep peak reasonable vs speech
    tone *= 0.25
    return tone
