"""Test fixtures — degenerate and normal text samples."""

# --- Degenerate samples ---

# Pure repetition: same word over and over
REPEAT_WORD = "hello " * 200

# Phrase loop: same sentence repeated
REPEAT_SENTENCE = "The quick brown fox jumps over the lazy dog. " * 50

# Character loop: single character
REPEAT_CHAR = "a" * 1000

# Structured repetition: same line pattern
REPEAT_LINES = "- Item one\n- Item two\n- Item three\n" * 30

# Degenerate with slight variation
NEAR_REPEAT = "".join(f"Step {i % 3 + 1}: Process the data. " for i in range(100))

# --- Normal samples ---

# Coherent English paragraph
NORMAL_ENGLISH = (
    "Machine learning models have transformed how we approach natural language processing. "
    "Recent advances in transformer architectures have enabled models to generate coherent, "
    "contextually appropriate text across a wide range of tasks. However, these models can "
    "sometimes fall into degenerate patterns where they repeat the same phrases or lose "
    "coherence entirely. Detection of such degeneration is crucial for production systems "
    "that rely on LLM outputs for user-facing applications. Various approaches have been "
    "proposed, including perplexity-based methods, repetition penalty during sampling, and "
    "post-generation filtering. Each approach has trade-offs in terms of computational cost, "
    "accuracy, and the types of degeneration they can catch. A robust solution should combine "
    "multiple signals to achieve high detection rates while minimizing false positives on "
    "legitimate structured content like code, tables, and enumerated lists."
)

# Code sample (structured but not degenerate)
NORMAL_CODE = '''def fibonacci(n: int) -> list[int]:
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence

def is_prime(num: int) -> bool:
    """Check if a number is prime."""
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

class DataProcessor:
    """Process and transform data records."""

    def __init__(self, records: list[dict]):
        self.records = records
        self._cache = {}

    def filter_by(self, key: str, value: str) -> list[dict]:
        cache_key = f"{key}:{value}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = [r for r in self.records if r.get(key) == value]
        self._cache[cache_key] = result
        return result
'''

# Markdown table (structured but not degenerate)
NORMAL_TABLE = """| Language | Paradigm | Year | Creator |
|----------|----------|------|---------|
| Python | Multi-paradigm | 1991 | Guido van Rossum |
| Rust | Systems | 2010 | Graydon Hoare |
| Go | Concurrent | 2009 | Robert Griesemer |
| TypeScript | Multi-paradigm | 2012 | Anders Hejlsberg |
| Kotlin | Multi-paradigm | 2011 | JetBrains |
| Swift | Multi-paradigm | 2014 | Chris Lattner |
| Julia | Scientific | 2012 | Jeff Bezanson |
| Zig | Systems | 2016 | Andrew Kelley |
"""

# Short text (below minimum threshold)
SHORT_TEXT = "Hello world"

# Mixed content (some repetition but within normal range)
NORMAL_MIXED = (
    "The system architecture consists of three main components:\n\n"
    "1. **Frontend**: React-based UI with real-time updates via SSE.\n"
    "2. **Backend**: FastAPI server handling authentication and routing.\n"
    "3. **Agent Core**: Python orchestrator with LangGraph pipelines.\n\n"
    "The frontend communicates with the backend through REST APIs.\n"
    "The backend delegates to the agent core for AI processing.\n"
    "The agent core manages tool execution and memory retrieval.\n\n"
    "Key design decisions include:\n"
    "- Using Neo4j for knowledge graph storage instead of pure vector DB\n"
    "- Implementing 2-tier inference fallback for reliability\n"
    "- Streaming responses via Server-Sent Events for low latency\n"
    "- Separating think/response content in the streaming pipeline\n"
)
