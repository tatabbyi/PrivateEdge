from __future__ import annotations

from inference.text_signals import score_transcript


def test_empty_text_is_neutral() -> None:
    s = score_transcript("   ")
    assert s.p_pii == 0.0
    assert s.p_toxicity == 0.0


def test_pii_patterns_raise_pii_score() -> None:
    s = score_transcript("Call me at 555-123-4567 or email test@example.com")
    assert s.p_pii >= 0.7


def test_toxic_terms_raise_toxicity_score() -> None:
    s = score_transcript("I hate this stupid thing")
    assert s.p_toxicity > 0.0
