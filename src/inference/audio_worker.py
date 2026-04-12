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
        self._device_lock = threading.Lock()
        self._input_device_index: int | None = None
        self._output_enabled: bool = False
        self._output_device_name: str | None = None
        self._mute_output: bool = False
        self._output_stream: Any = None
        self._try_load_whisper()

    def set_input_device(self, device_index: int | None) -> None:
        with self._device_lock:
            self._input_device_index = device_index

    def configure_output(self, enabled: bool, output_device_name: str | None) -> None:
        with self._device_lock:
            self._output_enabled = enabled
            self._output_device_name = (output_device_name or "").strip() or None
            if not enabled and self._output_stream is not None:
                try:
                    self._output_stream.stop()
                    self._output_stream.close()
                except Exception:  # noqa: BLE001
                    pass
                self._output_stream = None

    def set_output_muted(self, muted: bool) -> None:
        with self._device_lock:
            self._mute_output = muted

    def _resolve_output_device_index(self, sd: Any, wanted_name: str | None) -> int | None:
        if not wanted_name:
            return None
        wanted = wanted_name.strip().lower()
        for idx, d in enumerate(sd.query_devices()):
            if int(d.get("max_output_channels", 0) or 0) <= 0:
                continue
            if str(d.get("name", "")).strip().lower() == wanted:
                return idx
        return None

    def _write_output(self, sd: Any, block: np.ndarray) -> None:
        with self._device_lock:
            enabled = self._output_enabled
            wanted = self._output_device_name
            muted = self._mute_output
        if not enabled:
            return

        out_block = np.zeros_like(block) if muted else block
        if self._output_stream is None:
            dev_idx = self._resolve_output_device_index(sd, wanted)
            try:
                self._output_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype=np.float32,
                    device=dev_idx,
                )
                self._output_stream.start()
                if wanted and dev_idx is None:
                    logger.warning(
                        "Audio output device '%s' not found; using default output device",
                        wanted,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("Audio output stream init failed: %s", e)
                self._output_stream = None
                return
        try:
            self._output_stream.write(out_block.reshape(-1, 1))
        except Exception as e:  # noqa: BLE001
            logger.debug("Audio output write failed: %s", e)
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._output_stream = None

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
                    with self._device_lock:
                        dev = self._input_device_index
                    block = sd.rec(
                        CHUNK,
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype=np.float32,
                        device=dev,
                    )
                    sd.wait()
                except Exception as e:  # noqa: BLE001
                    logger.debug("Audio capture: %s", e)
                    time.sleep(0.2)
                    continue

                self._write_output(sd, block)
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
            if self._output_stream is not None:
                try:
                    self._output_stream.stop()
                    self._output_stream.close()
                except Exception:  # noqa: BLE001
                    pass
                self._output_stream = None


GLOBAL_AUDIO_BUF = AudioScoreBuffer()
_GLOBAL_AUDIO_WORKER: AudioWorker | None = None


def ensure_audio_worker() -> None:
    global _GLOBAL_AUDIO_WORKER
    if _GLOBAL_AUDIO_WORKER is None:
        _GLOBAL_AUDIO_WORKER = AudioWorker(GLOBAL_AUDIO_BUF)
        _GLOBAL_AUDIO_WORKER.start()


def set_audio_input_device(device_index: int | None) -> None:
    if _GLOBAL_AUDIO_WORKER is None:
        return
    _GLOBAL_AUDIO_WORKER.set_input_device(device_index)


def ensure_audio_worker_with_device(device_index: int | None) -> None:
    ensure_audio_worker()
    set_audio_input_device(device_index)


def list_input_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except ImportError:
        return []

    devices: list[dict[str, object]] = []
    for idx, d in enumerate(sd.query_devices()):
        max_in = int(d.get("max_input_channels", 0) or 0)
        if max_in <= 0:
            continue
        name = str(d.get("name", f"Input {idx}"))
        devices.append({"id": idx, "label": f"{name} ({max_in}ch)"})
    return devices


def list_output_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except ImportError:
        return []

    devices: list[dict[str, object]] = []
    for idx, d in enumerate(sd.query_devices()):
        max_out = int(d.get("max_output_channels", 0) or 0)
        if max_out <= 0:
            continue
        name = str(d.get("name", f"Output {idx}"))
        devices.append({"id": idx, "label": f"{name} ({max_out}ch)"})
    return devices


def configure_audio_output(enabled: bool, output_device_name: str | None, muted: bool) -> None:
    if _GLOBAL_AUDIO_WORKER is None:
        return
    _GLOBAL_AUDIO_WORKER.configure_output(enabled, output_device_name)
    _GLOBAL_AUDIO_WORKER.set_output_muted(muted)
