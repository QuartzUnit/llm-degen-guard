"""llm-degen-guard — Model-agnostic LLM output degeneration detector.

4-signal composite scoring in a single pass. Works with any LLM provider.
"""

from degen_guard.detector import DegenGuard
from degen_guard.report import DegenReport, SignalDetail

__all__ = ["DegenGuard", "DegenReport", "SignalDetail"]
__version__ = "0.1.0"
