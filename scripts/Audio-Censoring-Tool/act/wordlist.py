from __future__ import annotations

from pathlib import Path


def load_word_set(path: Path) -> set[str]:
    words: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line.lower())
    return words


def normalize_token(token: str) -> str:
    return token.lower().strip(".,!?;:\"'()[]")
