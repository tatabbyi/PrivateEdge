from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from act.bleep import generate_bleep
from act.buffer import DelayRingBuffer
from act.config import ActConfig
from act.wordlist import load_word_set, normalize_token


def _as_mono_float32(indata: np.ndarray) -> np.ndarray:
    x = np.asarray(indata, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.shape[1] == 1:
        return x[:, 0]
    return x.mean(axis=1).astype(np.float32)


def run_censoring(cfg: ActConfig, stop_event: threading.Event | None = None) -> None:
    stop_event = stop_event or threading.Event()
    word_path = Path(cfg.word_list_path)
    if not word_path.is_file():
        raise FileNotFoundError(f"word list not found: {word_path}")
    words = load_word_set(word_path)

    capacity = cfg.ring_capacity_samples()
    ring = DelayRingBuffer(capacity, cfg.delay_samples)
    model = WhisperModel(
        cfg.model_size,
        device=cfg.device,
        compute_type=cfg.compute_type,
    )

    block = cfg.block_frames
    last_asr_end = 0
    asr_lock = threading.Lock()

    def asr_worker() -> None:
        nonlocal last_asr_end
        w = cfg.asr_window_samples
        sr = cfg.sample_rate
        while not stop_event.is_set():
            time.sleep(0.15)
            with asr_lock:
                wg = ring.write_global
                end = last_asr_end + w
                if wg < end:
                    continue
                chunk = ring.copy_range(last_asr_end, w)
                if chunk is None:
                    continue
                window_start = last_asr_end
                last_asr_end = end

            segments, _ = model.transcribe(
                chunk,
                language="en",
                word_timestamps=True,
                vad_filter=True,
            )
            for seg in segments:
                if seg.words is None:
                    continue
                for wobj in seg.words:
                    raw = (wobj.word or "").strip()
                    tok = normalize_token(raw)
                    if not tok or tok not in words:
                        continue
                    g0 = window_start + int(wobj.start * sr)
                    g1 = window_start + int(wobj.end * sr)
                    if g1 <= g0:
                        continue
                    n = g1 - g0
                    bleep = generate_bleep(n, sr)
                    ring.replace_range(g0, bleep)

    asr_thread = threading.Thread(target=asr_worker, name="act-asr", daemon=True)
    asr_thread.start()

    def input_callback(indata, frames, _time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            pass
        ring.write(_as_mono_float32(indata))

    def output_callback(outdata, frames, _time, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            pass
        out = ring.read_for_output(frames)
        if outdata.shape[1] == 1:
            outdata[:, 0] = out
        else:
            outdata[:] = out.reshape(-1, 1)

    try:
        with sd.InputStream(
            device=cfg.input_device,
            channels=1,
            samplerate=cfg.sample_rate,
            blocksize=block,
            dtype="float32",
            callback=input_callback,
        ), sd.OutputStream(
            device=cfg.output_device,
            channels=1,
            samplerate=cfg.sample_rate,
            blocksize=block,
            dtype="float32",
            callback=output_callback,
        ):
            while not stop_event.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        asr_thread.join(timeout=2.0)
