# llm-degen-guard

> [한국어 문서](README.ko.md)

Model-agnostic LLM output degeneration detector — 4-signal composite scoring in a single pass.

## Install

```bash
pip install llm-degen-guard
```

## Quick Start

```python
from degen_guard import DegenGuard

# Batch check
report = DegenGuard.check("Some LLM output text...")
print(report.is_degenerate, report.composite_score)

# Streaming check
guard = DegenGuard()
for chunk in llm_stream:
    is_degen, score = guard.feed(chunk)
    if is_degen:
        print(f"Degeneration detected! score={score:.2f}")
        break
```

## How It Works

Four independent signals are combined into a single composite score (0.0 = normal, 1.0 = degenerate):

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| **N-gram diversity** | 0.35 | Distinct 3-gram ratio — low diversity = repetitive |
| **Compression ratio** | 0.30 | zlib compression ratio — highly compressible = repetitive |
| **Substring match** | 0.20 | Exact overlap between first/second half of window |
| **Line diversity** | 0.15 | Unique line ratio — duplicate lines = degenerate |

Structural content (code, lists, tables) gets a discount factor to reduce false positives.

## Streaming API

```python
guard = DegenGuard(
    window_size=256,       # characters to analyze
    check_interval=64,     # characters between checks
    alert_threshold=3,     # consecutive alerts before flagging
    score_threshold=0.50,  # minimum score to trigger alert
)

for chunk in llm_stream:
    is_degen, score = guard.feed(chunk)
    if is_degen:
        break

guard.reset()  # reuse for next stream
```

## Batch API

```python
report = DegenGuard.check(full_text)

report.is_degenerate      # bool
report.composite_score    # 0.0 ~ 1.0
report.signals            # tuple of SignalDetail
report.structural_penalty # < 1.0 if code/tables detected
report.text_length        # input length
```

## Why not...?

| Alternative | Limitation | llm-degen-guard |
|------------|-----------|-----------------|
| **Antislop Sampler** | Needs model weight access (no API support) | Works on any output text |
| **SpecRA** | Paper only, no code available | `pip install` ready |
| **UQLM** | Requires N generations (cost × N) | Single pass, real-time streaming |
| **repetition_penalty** | Penalizes prompt tokens, causes incoherence | Post-generation, no side effects |

## Used in

- [watchdeck](https://github.com/QuartzUnit/watchdeck) — Web page monitoring with visual diffs and safety guards

## License

MIT
