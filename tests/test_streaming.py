"""Tests for streaming-specific scenarios."""

from degen_guard import DegenGuard
from tests.fixtures.samples import NORMAL_ENGLISH, REPEAT_WORD


class TestStreamingScenarios:
    """Real-world streaming scenarios."""

    def test_normal_then_degenerate(self):
        """Text starts normal then degenerates mid-stream."""
        guard = DegenGuard()
        # Normal start
        for i in range(0, len(NORMAL_ENGLISH), 32):
            is_degen, _ = guard.feed(NORMAL_ENGLISH[i : i + 32])
            assert not is_degen

        # Then degenerates
        detected = False
        degen_text = "bla bla bla " * 100
        for i in range(0, len(degen_text), 32):
            is_degen, _ = guard.feed(degen_text[i : i + 32])
            if is_degen:
                detected = True
                break
        assert detected

    def test_large_chunks(self):
        """Feeding large chunks at once."""
        guard = DegenGuard()
        is_degen, score = guard.feed(REPEAT_WORD)
        # Single large feed — enough for multiple checks
        assert is_degen or score > 0.3

    def test_varying_chunk_sizes(self):
        """Chunks of different sizes."""
        guard = DegenGuard()
        text = REPEAT_WORD
        sizes = [10, 50, 3, 100, 7, 200, 15]
        pos = 0
        detected = False
        for size in sizes * 5:
            chunk = text[pos : pos + size]
            if not chunk:
                break
            is_degen, _ = guard.feed(chunk)
            if is_degen:
                detected = True
                break
            pos += size
        assert detected

    def test_multiple_streams_independent(self):
        """Two guards tracking different streams don't interfere."""
        guard1 = DegenGuard()
        guard2 = DegenGuard()

        # Feed degenerate to guard1, normal to guard2
        for i in range(0, 600, 32):
            guard1.feed(REPEAT_WORD[i : i + 32] if i < len(REPEAT_WORD) else "x" * 32)
            is_degen2, _ = guard2.feed(NORMAL_ENGLISH[i % len(NORMAL_ENGLISH) : i % len(NORMAL_ENGLISH) + 32])
            assert not is_degen2

    def test_buffer_cap(self):
        """Buffer doesn't grow unbounded."""
        guard = DegenGuard(window_size=256)
        # Feed a lot of text
        for _ in range(100):
            guard.feed("x" * 100)
        # Internal buffer should be capped at 2 * window_size
        assert len(guard._buf) <= guard.window_size * 2
