"""Regex and keyword signals on transcript text (PII, toxicity)."""

from __future__ import annotations

import re
from typing import NamedTuple

_PHONE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

_TOXIC = frozenset(
    w.lower()
    for w in (
        "damn",
        "hell",
        "shit",
        "fuck",
        "bitch",
        "asshole",
        "kill",
        "hate",
        "stupid",
        "idiot",
    )
)


class TextSignals(NamedTuple):
    p_pii: float
    p_toxicity: float


def score_transcript(text: str) -> TextSignals:
    if not text or not text.strip():
        return TextSignals(0.0, 0.0)
    t = text.lower()
    pii_hits = 0
    if _PHONE.search(text):
        pii_hits += 1
    if _EMAIL.search(text):
        pii_hits += 1
    if _SSN.search(text):
        pii_hits += 1
    if _CC.search(text):
        pii_hits += 1
    p_pii = min(1.0, pii_hits * 0.35 + (0.2 if len(text) > 200 else 0))

    words = re.findall(r"[a-zA-Z]+", t)
    if not words:
        tox = 0.0
    else:
        bad = sum(1 for w in words if w in _TOXIC)
        tox = min(1.0, bad / max(3, len(words) * 0.15))
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    anger_boost = min(0.4, caps_ratio * 2.0)
    return TextSignals(p_pii, min(1.0, tox + anger_boost * 0.3))
