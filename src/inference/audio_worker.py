"""Microphone capture + faster-whisper ASR + text-based PII/toxicity scores."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from inference.text_signals import score_transcript
from policy.types import ModelScores

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK = 4800


class AudioScoreBuffer:
    """Thread-safe latest audio-derived scores."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scores = ModelScores()
        self._last_text = ""
        self._rms = 0.0

    def update(self, m: ModelScores, text: str, rms: float) -> None:
        with self._lock:
            self._scores = m
            self._last_text = text
            self._rms = rms

    def snapshot(self) -> tuple[ModelScores, str, float]:
        with self._lock:
            return self._scores, self._last_text, self._rms


class AudioWorker:
    """Background mic reader + periodic transcription."""

    def __init__(self, buf: AudioScoreBuffer) -> None:
        self._buf = buf
        self._model: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._accum: list[np.ndarray] = []
        self._try_load_whisper()

    def _try_load_whisper(self) -> None:
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel("tiny", device="cpu", compute_type="int8")
            logger.info("faster-whisper tiny loaded for on-device ASR")
        except Exception as e:  # noqa: BLE001
            self._model = None
            logger.warning(
                "faster-whisper not available (%s). Install: pip install faster-whisper",
                e,
            )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="privateedge-audio", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning("sounddevice not installed; audio path disabled")
            return

        try:
            while not self._stop.is_set():
                try:
                    block = sd.rec(CHUNK, samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
                    sd.wait()
                except Exception as e:  # noqa: BLE001
                    logger.debug("Audio capture: %s", e)
                    time.sleep(0.2)
                    continue

                self._accum.append(block.copy().flatten())
                total = sum(len(x) for x in self._accum)
                if total < SAMPLE_RATE:
                    continue
                audio = np.concatenate(self._accum, axis=0)
                self._accum.clear()
                if len(audio) > SAMPLE_RATE * 8:
                    audio = audio[-SAMPLE_RATE * 8 :]

                rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-9)
                text = ""
                if self._model is not None:
                    try:
                        segments, _ = self._model.transcribe(
                            audio, language="en", vad_filter=True
                        )
                        text = " ".join(s.text for s in segments).strip()
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Whisper transcribe: %s", e)

                ts = score_transcript(text)
                stress = min(1.0, rms * 8.0)
                caps = (
                    sum(1 for c in text if c.isupper()) / max(1, len(text))
                    if text
                    else 0.0
                )
                anger = min(1.0, ts.p_toxicity * 0.7 + caps * 0.28)
                m = ModelScores(
                    p_doc=0.0,
                    p_face_other=0.0,
                    p_nsfw=0.0,
                    p_pii_audio=ts.p_pii,
                    p_toxicity=ts.p_toxicity,
                    anger=max(anger, ts.p_toxicity * 0.5),
                    stress=stress,
                )
                self._buf.update(m, text, rms)
        finally:
            pass


GLOBAL_AUDIO_BUF = AudioScoreBuffer()
_GLOBAL_AUDIO_WORKER: AudioWorker | None = None


def ensure_audio_worker() -> None:
    global _GLOBAL_AUDIO_WORKER
    if _GLOBAL_AUDIO_WORKER is None:
        _GLOBAL_AUDIO_WORKER = AudioWorker(GLOBAL_AUDIO_BUF)
        _GLOBAL_AUDIO_WORKER.start()
