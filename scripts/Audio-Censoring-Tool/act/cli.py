from __future__ import annotations

import argparse
import threading
from dataclasses import replace
from pathlib import Path

from act.config import (
    ActConfig,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DELAY_SECONDS,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_WORDLIST_PATH,
)
from act.engine import run_censoring
from act.io import print_devices


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m act",
        description="Audio Censoring Tool — live mic, fixed delay, word-bleep via faster-whisper.",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="Print sound devices and exit.",
    )
    p.add_argument(
        "--input-device",
        type=int,
        default=None,
        metavar="N",
        help="Input device index (see --list-devices). Default: system default input.",
    )
    p.add_argument(
        "--output-device",
        type=int,
        default=None,
        metavar="N",
        help="Output device index (e.g. PipeWire null sink). Default: system default output.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Playback delay in seconds (default: {DEFAULT_DELAY_SECONDS}).",
    )
    p.add_argument(
        "--asr-window",
        type=float,
        default=3.0,
        help="ASR chunk length in seconds (default: 3).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Whisper model size (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--word-list",
        type=Path,
        default=DEFAULT_WORDLIST_PATH,
        help="Path to word list file (one entry per line).",
    )
    p.add_argument(
        "--compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        help=f"faster-whisper compute type (default: {DEFAULT_COMPUTE_TYPE}).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Inference device for faster-whisper (default: {DEFAULT_DEVICE}).",
    )
    p.add_argument(
        "--block-frames",
        type=int,
        default=1024,
        help="Host audio block size in samples (default: 1024).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        print_devices()
        return

    if args.delay <= 0:
        raise SystemExit("--delay must be positive")
    if args.asr_window <= 0:
        raise SystemExit("--asr-window must be positive")

    cfg = replace(
        ActConfig(),
        delay_seconds=args.delay,
        asr_window_seconds=args.asr_window,
        model_size=args.model,
        compute_type=args.compute_type,
        device=args.device,
        word_list_path=args.word_list.resolve(),
        input_device=args.input_device,
        output_device=args.output_device,
        block_frames=args.block_frames,
    )

    stop = threading.Event()

    try:
        run_censoring(cfg, stop_event=stop)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
