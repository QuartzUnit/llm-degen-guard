"""Degeneration report dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SignalDetail:
    """Individual signal result."""

    name: str
    raw_score: float  # 0.0 (normal) ~ 1.0 (degenerate)
    weight: float
    weighted_score: float


@dataclass(frozen=True, slots=True)
class DegenReport:
    """Degeneration check result for a text."""

    is_degenerate: bool
    composite_score: float  # 0.0 (normal) ~ 1.0 (degenerate)
    signals: tuple[SignalDetail, ...] = field(default_factory=tuple)
    text_length: int = 0
    structural_penalty: float = 1.0  # < 1.0 means structural content detected
