"""End-to-end tests — realistic LLM output scenarios."""

from degen_guard import DegenGuard


class TestRealWorldDegeneration:
    """Simulate actual LLM degeneration patterns observed in production."""

    def test_vllm_token_loop(self):
        """vLLM Qwen model repeating tokens — real pattern from GPU Server."""
        guard = DegenGuard()
        # Simulates streaming: model starts normal, then degenerates
        normal_prefix = (
            "To implement a REST API in Python, you'll need to use a framework like FastAPI. "
            "First, install the package with pip install fastapi uvicorn. "
            "Then create your main application file:\n\n"
        )
        for ch in normal_prefix:
            is_degen, _ = guard.feed(ch)
            assert not is_degen, f"False positive on normal text at char: {ch!r}"

        # Model starts looping — short repetitive output (real vLLM degeneration)
        loop = "print('hello')\nprint('hello')\n" * 40
        detected = False
        for i in range(0, len(loop), 16):
            is_degen, score = guard.feed(loop[i : i + 16])
            if is_degen:
                detected = True
                break
        assert detected, "Failed to detect code block repetition loop"

    def test_llama_cpp_phrase_loop(self):
        """llama.cpp Q4 quantized model repeating phrases."""
        guard = DegenGuard()
        text = (
            "The key point here is that we need to consider the implications. "
            "The key point here is that we need to consider the implications. "
            "The key point here is that we need to consider the implications. "
            "The key point here is that we need to consider the implications. "
            "The key point here is that we need to consider the implications. "
            "The key point here is that we need to consider the implications. "
            "The key point here is that we need to consider the implications. "
            "The key point here is that we need to consider the implications. "
        )
        detected = False
        for i in range(0, len(text), 32):
            is_degen, _ = guard.feed(text[i : i + 32])
            if is_degen:
                detected = True
                break
        assert detected

    def test_thinking_block_degeneration(self):
        """Model's think block spirals into repetition (Qwen thinking mode)."""
        report = DegenGuard.check(
            "Let me think about this step by step.\n"
            + "I need to analyze the data.\n" * 30
        )
        assert report.is_degenerate

    def test_unicode_degeneration_cjk(self):
        """CJK repetition pattern (seen with Chinese/Korean models)."""
        guard = DegenGuard()
        text = "이 코드를 수정하려면 " * 50
        detected = False
        for i in range(0, len(text), 20):
            is_degen, _ = guard.feed(text[i : i + 20])
            if is_degen:
                detected = True
                break
        assert detected

    def test_batch_long_normal_code(self):
        """Long but legitimate code should not trigger."""
        code = '''import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator

@dataclass
class StreamConfig:
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95

async def stream_response(config: StreamConfig) -> AsyncGenerator[str, None]:
    """Stream LLM response tokens."""
    buffer = []
    async for token in get_tokens(config):
        buffer.append(token)
        if len(buffer) >= 5:
            yield "".join(buffer)
            buffer.clear()
    if buffer:
        yield "".join(buffer)

class ResponseValidator:
    def __init__(self, min_length: int = 10):
        self.min_length = min_length
        self._history = []

    def validate(self, text: str) -> bool:
        if len(text) < self.min_length:
            return False
        self._history.append(text)
        return True

    def get_stats(self) -> dict:
        return {
            "total": len(self._history),
            "avg_length": sum(len(t) for t in self._history) / max(1, len(self._history)),
        }
'''
        report = DegenGuard.check(code)
        assert not report.is_degenerate
        assert report.structural_penalty < 1.0  # code discount applied

    def test_batch_markdown_with_lists(self):
        """Markdown with bullet lists should not trigger."""
        md = """# Architecture Overview

The system consists of several key components:

## Components

- **Frontend**: React-based SPA with SSE streaming support
- **Backend**: FastAPI server handling authentication, routing, and session management
- **Agent Core**: Python orchestrator using LangGraph for pipeline management
- **Knowledge Graph**: Neo4j for entity relationships and long-term memory
- **Vector Store**: Qdrant for semantic search and document retrieval
- **Cache Layer**: Redis for session state and rate limiting

## Data Flow

1. User sends a message through the web interface
2. Backend authenticates the request via JWT
3. Backend opens an SSE connection and forwards to Agent Core
4. Agent runs preprocessing (routing, retrieval, context assembly)
5. Agent calls the LLM for inference
6. Response streams back through SSE to the frontend
7. Post-processing extracts entities and updates memory

## Deployment

The system runs across three machines:
- Agent host: Agent host, databases, embedding models
- GPU Server: Primary LLM inference (vLLM)
- Secondary server: Secondary LLM (fallback), image generation
"""
        report = DegenGuard.check(md)
        assert not report.is_degenerate

    def test_streaming_realistic_chunk_sizes(self):
        """Simulate realistic vLLM token streaming (1-5 tokens per chunk)."""
        guard = DegenGuard()
        # Normal response followed by degeneration
        normal_tokens = ["The ", "answer ", "to ", "your ", "question ", "is ", "as ", "follows:\n\n"]
        repeat_unit = ["First, ", "you ", "need ", "to ", "understand ", "the ", "concept ", "thoroughly. "]
        tokens = normal_tokens + repeat_unit * 15  # 15 repetitions of the same phrase
        detected = False
        for tok in tokens:
            is_degen, score = guard.feed(tok)
            if is_degen:
                detected = True
                break
        assert detected

    def test_report_signal_breakdown(self):
        """Verify signal breakdown makes sense for a known degenerate text."""
        text = "abc " * 200
        report = DegenGuard.check(text)
        assert report.is_degenerate

        # Check individual signals
        for signal in report.signals:
            assert 0.0 <= signal.raw_score <= 1.0
            assert 0.0 <= signal.weighted_score <= 1.0
            assert signal.weight > 0

        # Sum of weights should be approximately 1.0
        total_weight = sum(s.weight for s in report.signals)
        assert abs(total_weight - 1.0) < 0.01
