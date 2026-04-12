from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SAMPLE_RATE = 16000
DEFAULT_DELAY_SECONDS = 4.0
DEFAULT_ASR_WINDOW_SECONDS = 3.0
DEFAULT_MODEL = "tiny"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_DEVICE = "cpu"

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_WORDLIST_PATH = _PACKAGE_DIR.parent / "data" / "default_wordlist.txt"


@dataclass(frozen=True)
class ActConfig:
    sample_rate: int = SAMPLE_RATE
    delay_seconds: float = DEFAULT_DELAY_SECONDS
    asr_window_seconds: float = DEFAULT_ASR_WINDOW_SECONDS
    model_size: str = DEFAULT_MODEL
    compute_type: str = DEFAULT_COMPUTE_TYPE
    device: str = DEFAULT_DEVICE
    word_list_path: Path = DEFAULT_WORDLIST_PATH
    input_device: int | None = None
    output_device: int | None = None
    block_frames: int = 1024

    @property
    def delay_samples(self) -> int:
        return int(self.delay_seconds * self.sample_rate)

    @property
    def asr_window_samples(self) -> int:
        return int(self.asr_window_seconds * self.sample_rate)

    def ring_capacity_samples(self) -> int:
        # Room for delay line + ASR window + margin so we never overwrite unplayed audio.
        margin = self.asr_window_samples
        return self.delay_samples + 2 * self.asr_window_samples + margin
