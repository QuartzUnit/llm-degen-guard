"""Core degeneration detector — 4-signal composite scoring.

Signals:
    1. Distinct 3-gram ratio (weight 0.35) — Normal: 0.70~0.95, Degenerate: <0.30
    2. zlib compression ratio (weight 0.30) — Normal: 0.40~0.65, Degenerate: <0.20
    3. Exact substring matching (weight 0.20) — Found: 1.0, Not found: 0.0
    4. Line diversity (weight 0.15) — Normal: 0.60+, Degenerate: <0.30
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

from degen_guard.report import DegenReport, SignalDetail


@dataclass
class _SignalWeights:
    """Configurable signal weights. Must sum to 1.0."""

    ngram: float = 0.35
    compression: float = 0.30
    substring: float = 0.20
    line_diversity: float = 0.15


@dataclass
class DegenGuard:
    """Model-agnostic LLM output degeneration detector.

    Supports both streaming (feed-based) and batch (check) usage.

    Streaming:
        guard = DegenGuard()
        for chunk in llm_stream:
            is_degen, score = guard.feed(chunk)
            if is_degen:
                break

    Batch:
        report = DegenGuard.check("some LLM output text")
    """

    window_size: int = 256
    check_interval: int = 64
    alert_threshold: int = 3
    score_threshold: float = 0.50
    structural_discount: float = 0.7
    weights: _SignalWeights = field(default_factory=_SignalWeights)

    # Internal state (streaming)
    _buf: str = field(default="", init=False, repr=False)
    _chars_since_check: int = field(default=0, init=False, repr=False)
    _consecutive_alerts: int = field(default=0, init=False, repr=False)
    _total_chars: int = field(default=0, init=False, repr=False)
    _last_score: float = field(default=0.0, init=False, repr=False)

    # ---- Streaming API ----

    def feed(self, text: str) -> tuple[bool, float]:
        """Feed a text chunk, return (is_degenerate, composite_score).

        Score is 0.0 until enough text accumulates for the first check.
        """
        self._buf += text
        self._chars_since_check += len(text)
        self._total_chars += len(text)

        if len(self._buf) < self.window_size or self._chars_since_check < self.check_interval:
            return False, self._last_score

        self._chars_since_check = 0
        window = self._buf[-self.window_size :]

        score = self._compute_score(window)
        self._last_score = score

        if score >= self.score_threshold:
            self._consecutive_alerts += 1
        else:
            self._consecutive_alerts = 0

        # Cap buffer at 2x window
        if len(self._buf) > self.window_size * 2:
            self._buf = self._buf[-self.window_size * 2 :]

        return self._consecutive_alerts >= self.alert_threshold, score

    def reset(self) -> None:
        """Reset internal state for reuse."""
        self._buf = ""
        self._chars_since_check = 0
        self._consecutive_alerts = 0
        self._total_chars = 0
        self._last_score = 0.0

    # ---- Batch API ----

    @classmethod
    def check(
        cls,
        text: str,
        window_size: int = 256,
        score_threshold: float = 0.50,
        structural_discount: float = 0.7,
    ) -> DegenReport:
        """One-shot check on a complete text. Returns a detailed DegenReport."""
        instance = cls(
            window_size=window_size,
            score_threshold=score_threshold,
            structural_discount=structural_discount,
        )
        if len(text) < 10:
            return DegenReport(
                is_degenerate=False,
                composite_score=0.0,
                text_length=len(text),
            )

        window = text[-window_size:] if len(text) > window_size else text
        score, signals, penalty = instance._compute_score_detailed(window)

        return DegenReport(
            is_degenerate=score >= score_threshold,
            composite_score=score,
            signals=tuple(signals),
            text_length=len(text),
            structural_penalty=penalty,
        )

    # ---- Scoring ----

    def _compute_score(self, window: str) -> float:
        """4-signal composite score. 0.0 (normal) ~ 1.0 (degenerate)."""
        if len(window) < 10:
            return 0.0

        s1 = self._signal_ngram(window)
        s2 = self._signal_compression(window)
        s3 = self._signal_substring(window)
        s4 = self._signal_line_diversity(window)

        penalty = self._structural_penalty(window)
        w = self.weights

        return (w.ngram * s1 + w.compression * s2 + w.substring * s3 + w.line_diversity * s4) * penalty

    def _compute_score_detailed(self, window: str) -> tuple[float, list[SignalDetail], float]:
        """Compute score with per-signal details."""
        w = self.weights

        s1 = self._signal_ngram(window)
        s2 = self._signal_compression(window)
        s3 = self._signal_substring(window)
        s4 = self._signal_line_diversity(window)
        penalty = self._structural_penalty(window)

        signals = [
            SignalDetail("ngram_diversity", s1, w.ngram, w.ngram * s1),
            SignalDetail("compression_ratio", s2, w.compression, w.compression * s2),
            SignalDetail("substring_match", s3, w.substring, w.substring * s3),
            SignalDetail("line_diversity", s4, w.line_diversity, w.line_diversity * s4),
        ]
        composite = sum(s.weighted_score for s in signals) * penalty
        return composite, signals, penalty

    # ---- Individual signals ----

    @staticmethod
    def _signal_ngram(window: str) -> float:
        """Signal 1: Distinct 3-gram ratio. Low diversity → high score."""
        trigrams = [window[i : i + 3] for i in range(len(window) - 2)]
        if not trigrams:
            return 0.0
        diversity = len(set(trigrams)) / len(trigrams)
        # Normal: 0.70~0.95 → 0.0, Degenerate: <0.30 → 1.0
        return max(0.0, min(1.0, (0.70 - diversity) / 0.40))

    @staticmethod
    def _signal_compression(window: str) -> float:
        """Signal 2: zlib compression ratio. Low ratio → high score (repetitive)."""
        encoded = window.encode("utf-8")
        if not encoded:
            return 0.0
        compressed = zlib.compress(encoded, 1)
        ratio = len(compressed) / len(encoded)
        # Normal: 0.40~0.65 → 0.0, Degenerate: <0.20 → 1.0
        return max(0.0, min(1.0, (0.40 - ratio) / 0.20))

    @staticmethod
    def _signal_substring(window: str) -> float:
        """Signal 3: Exact substring match between halves."""
        half = len(window) // 2
        first_half = window[:half]
        second_half = window[half:]
        match_len = 0
        for size in (128, 64, 32):
            if size <= half and first_half[-size:] == second_half[:size]:
                match_len = size
                break
        if match_len >= 64:
            return 1.0
        if match_len >= 32:
            return match_len / 64.0
        return 0.0

    @staticmethod
    def _signal_line_diversity(window: str) -> float:
        """Signal 4: Unique line ratio. Low diversity → high score."""
        lines = [ln.strip() for ln in window.split("\n") if ln.strip()]
        if len(lines) < 3:
            return 0.0
        diversity = len(set(lines)) / len(lines)
        # Normal: 0.50+ → 0.0, Degenerate: <0.30 → 1.0
        return max(0.0, min(1.0, (0.50 - diversity) / 0.20))

    def _structural_penalty(self, window: str) -> float:
        """Reduce score for structured content (code, lists, tables)."""
        markers = ("```", "- ", "| ", "def ", "class ", "import ", "    ")
        if any(marker in window for marker in markers):
            return self.structural_discount
        return 1.0
