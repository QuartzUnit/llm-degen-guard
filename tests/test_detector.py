"""Tests for DegenGuard core detector."""

import pytest

from degen_guard import DegenGuard, DegenReport
from tests.fixtures.samples import (
    NORMAL_CODE,
    NORMAL_ENGLISH,
    NORMAL_MIXED,
    NORMAL_TABLE,
    REPEAT_CHAR,
    REPEAT_LINES,
    REPEAT_SENTENCE,
    REPEAT_WORD,
    SHORT_TEXT,
)


class TestBatchAPI:
    """Test DegenGuard.check() batch API."""

    def test_detects_word_repetition(self):
        report = DegenGuard.check(REPEAT_WORD)
        assert report.is_degenerate
        assert report.composite_score > 0.6

    def test_detects_sentence_repetition(self):
        report = DegenGuard.check(REPEAT_SENTENCE)
        assert report.is_degenerate
        assert report.composite_score > 0.5

    def test_detects_char_repetition(self):
        report = DegenGuard.check(REPEAT_CHAR)
        assert report.is_degenerate

    def test_normal_english_passes(self):
        report = DegenGuard.check(NORMAL_ENGLISH)
        assert not report.is_degenerate
        assert report.composite_score < 0.4

    def test_normal_code_passes(self):
        report = DegenGuard.check(NORMAL_CODE)
        assert not report.is_degenerate

    def test_normal_table_passes(self):
        report = DegenGuard.check(NORMAL_TABLE)
        assert not report.is_degenerate

    def test_normal_mixed_passes(self):
        report = DegenGuard.check(NORMAL_MIXED)
        assert not report.is_degenerate

    def test_short_text_passes(self):
        report = DegenGuard.check(SHORT_TEXT)
        assert not report.is_degenerate
        assert report.composite_score == 0.0

    def test_empty_text(self):
        report = DegenGuard.check("")
        assert not report.is_degenerate
        assert report.composite_score == 0.0

    def test_report_has_signals(self):
        report = DegenGuard.check(REPEAT_WORD)
        assert len(report.signals) == 4
        names = {s.name for s in report.signals}
        assert names == {"ngram_diversity", "compression_ratio", "substring_match", "line_diversity"}

    def test_report_type(self):
        report = DegenGuard.check(NORMAL_ENGLISH)
        assert isinstance(report, DegenReport)
        assert isinstance(report.text_length, int)
        assert report.text_length == len(NORMAL_ENGLISH)

    def test_structural_penalty_applied_to_code(self):
        report = DegenGuard.check(NORMAL_CODE)
        assert report.structural_penalty < 1.0

    def test_structural_penalty_not_applied_to_plain(self):
        report = DegenGuard.check(NORMAL_ENGLISH)
        assert report.structural_penalty == 1.0

    def test_custom_threshold(self):
        # Very low threshold: even mild repetition triggers
        report_low = DegenGuard.check(REPEAT_LINES, score_threshold=0.01)
        # Very high threshold: hard to trigger
        report_high = DegenGuard.check(REPEAT_LINES, score_threshold=0.99)
        assert report_low.composite_score == report_high.composite_score  # same score
        assert report_low.is_degenerate  # low threshold triggers
        assert not report_high.is_degenerate  # high threshold doesn't

    def test_custom_window_size(self):
        report = DegenGuard.check(REPEAT_WORD, window_size=64)
        assert isinstance(report, DegenReport)
        # Smaller window has lower compression efficiency but still scores high
        assert report.composite_score > 0.4


class TestStreamingAPI:
    """Test DegenGuard streaming (feed-based) API."""

    def test_streaming_detects_repetition(self):
        guard = DegenGuard()
        text = REPEAT_WORD
        chunk_size = 32
        detected = False
        for i in range(0, len(text), chunk_size):
            is_degen, score = guard.feed(text[i : i + chunk_size])
            if is_degen:
                detected = True
                break
        assert detected

    def test_streaming_normal_text_passes(self):
        guard = DegenGuard()
        text = NORMAL_ENGLISH
        chunk_size = 32
        for i in range(0, len(text), chunk_size):
            is_degen, _score = guard.feed(text[i : i + chunk_size])
            assert not is_degen

    def test_streaming_char_by_char(self):
        guard = DegenGuard()
        text = REPEAT_CHAR
        detected = False
        for ch in text:
            is_degen, _score = guard.feed(ch)
            if is_degen:
                detected = True
                break
        assert detected

    def test_streaming_returns_score(self):
        guard = DegenGuard()
        # Feed enough text to trigger a check
        _is_degen, score = guard.feed(REPEAT_WORD[:300])
        assert isinstance(score, float)

    def test_reset_clears_state(self):
        guard = DegenGuard()
        guard.feed(REPEAT_WORD[:500])
        guard.reset()
        # After reset, no detection on normal text
        is_degen, score = guard.feed(NORMAL_ENGLISH)
        assert not is_degen

    def test_score_zero_before_window(self):
        guard = DegenGuard(window_size=256)
        is_degen, score = guard.feed("short")
        assert not is_degen
        assert score == 0.0

    def test_consecutive_alert_threshold(self):
        """Detection requires consecutive alerts, not just one high score."""
        guard = DegenGuard(alert_threshold=5)
        text = REPEAT_WORD
        chunk_size = 64
        scores = []
        for i in range(0, min(len(text), 512), chunk_size):
            is_degen, score = guard.feed(text[i : i + chunk_size])
            if score > 0:
                scores.append(score)
            if is_degen:
                break
        # Need at least alert_threshold checks before detection
        assert len(scores) >= 3  # had multiple checks with high scores


class TestSignals:
    """Test individual signal computations."""

    def test_ngram_high_diversity(self):
        # Diverse text → low score (normal)
        score = DegenGuard._signal_ngram(NORMAL_ENGLISH[:256])
        assert score < 0.3

    def test_ngram_low_diversity(self):
        # Repetitive text → high score (degenerate)
        score = DegenGuard._signal_ngram(REPEAT_WORD[:256])
        assert score > 0.5

    def test_compression_normal(self):
        score = DegenGuard._signal_compression(NORMAL_ENGLISH[:256])
        assert score < 0.3

    def test_compression_repetitive(self):
        score = DegenGuard._signal_compression(REPEAT_WORD[:256])
        assert score > 0.5

    def test_substring_no_match(self):
        score = DegenGuard._signal_substring(NORMAL_ENGLISH[:256])
        assert score == 0.0

    def test_substring_match(self):
        # Build text with exact match across halves
        pattern = "X" * 128
        text = pattern + pattern
        score = DegenGuard._signal_substring(text)
        assert score == 1.0

    def test_line_diversity_normal(self):
        score = DegenGuard._signal_line_diversity(NORMAL_MIXED[:256])
        assert score < 0.5

    def test_line_diversity_repetitive(self):
        repeated_lines = "same line\n" * 50
        score = DegenGuard._signal_line_diversity(repeated_lines)
        assert score > 0.5

    def test_ngram_empty(self):
        assert DegenGuard._signal_ngram("") == 0.0

    def test_compression_empty(self):
        assert DegenGuard._signal_compression("") == 0.0

    def test_substring_short(self):
        assert DegenGuard._signal_substring("abc") == 0.0

    def test_line_diversity_few_lines(self):
        assert DegenGuard._signal_line_diversity("one\ntwo") == 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_text(self):
        text = "이것은 한국어 텍스트입니다. " * 30
        report = DegenGuard.check(text)
        assert isinstance(report, DegenReport)

    def test_mixed_cjk(self):
        text = "日本語テスト。中文测试。한국어 테스트。English test. " * 20
        report = DegenGuard.check(text)
        assert isinstance(report, DegenReport)

    def test_emoji_text(self):
        text = "Hello 👋 world 🌍 " * 30
        report = DegenGuard.check(text)
        assert isinstance(report, DegenReport)

    def test_whitespace_only(self):
        text = "   \n\n\t\t   \n" * 50
        report = DegenGuard.check(text)
        assert isinstance(report, DegenReport)

    def test_single_very_long_line(self):
        text = "x" * 10000
        report = DegenGuard.check(text)
        assert report.is_degenerate

    def test_newlines_only(self):
        text = "\n" * 500
        report = DegenGuard.check(text)
        assert isinstance(report, DegenReport)

    def test_dataclass_frozen(self):
        report = DegenGuard.check(NORMAL_ENGLISH)
        with pytest.raises(AttributeError):
            report.is_degenerate = True  # type: ignore[misc]

    def test_guard_reusable_after_reset(self):
        guard = DegenGuard()
        guard.feed(REPEAT_WORD)
        guard.reset()
        # Should work normally after reset
        is_degen, _ = guard.feed(NORMAL_ENGLISH)
        assert not is_degen
