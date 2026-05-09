"""
Integration tests for LLMClient — requires real API access.

Skip conditions:
    - DEEPSEEK_API_KEY not set  → skip DeepSeek tests
    - Ollama not running         → skip Ollama tests

Run with:
    DEEPSEEK_API_KEY=sk-xxx pytest tests/test_llm_client_integration.py -v
"""

import os

import pytest

from core.llm_client import LLMClient, LLMConfig


# ── Skip conditions ────────────────────────────────────────────

def _deepseek_available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _ollama_available() -> bool:
    """Probe if Ollama is running on localhost:11434."""
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        urllib.request.urlopen(req, timeout=2)
        return True
    except Exception:
        return False


deepseek_only = pytest.mark.skipif(
    not _deepseek_available(),
    reason="DEEPSEEK_API_KEY not set",
)

ollama_only = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running on localhost:11434",
)

requires_any = pytest.mark.skipif(
    not _deepseek_available() and not _ollama_available(),
    reason="No LLM provider available (set DEEPSEEK_API_KEY or start Ollama)",
)


# ── DeepSeek Integration Tests ─────────────────────────────────


class TestDeepSeekIntegration:
    """Real API calls to DeepSeek."""

    @deepseek_only
    def test_basic_chat_completion(self):
        """DeepSeek should return a non-empty response."""
        client = LLMClient.deepseek(api_key=os.environ["DEEPSEEK_API_KEY"])
        messages = [{"role": "user", "content": "Say 'Hello, World!' and nothing else."}]
        response = client.generate(messages, temperature=0.0, max_tokens=50)
        assert len(response) > 0
        assert "Hello" in response

    @deepseek_only
    def test_streaming_accumulates_same_content(self):
        """Streaming mode should return same total content as non-streaming."""
        client = LLMClient.deepseek(api_key=os.environ["DEEPSEEK_API_KEY"])
        messages = [
            {"role": "user", "content": "Reply with exactly: 'streaming works'"}
        ]

        # Non-stream
        sync = client.generate(messages, temperature=0.0, max_tokens=30, stream=False)

        # Stream
        streamed = client.generate(messages, temperature=0.0, max_tokens=30, stream=True)

        # Both should be non-empty; streaming accumulates correctly
        assert len(sync) > 0
        assert len(streamed) > 0
        # With temp=0 they should be identical (deterministic)
        assert sync.strip() == streamed.strip(), (
            f"Mismatch:\n  sync: {sync!r}\n  stream: {streamed!r}"
        )

    @deepseek_only
    @pytest.mark.asyncio
    async def test_streaming_iterator_yields_tokens(self):
        """The generate_stream() method should yield individual tokens."""
        client = LLMClient.deepseek(api_key=os.environ["DEEPSEEK_API_KEY"])
        messages = [
            {"role": "user", "content": "Count to 5: 1 2 3 4 5"}
        ]
        tokens = list(client.generate_stream(messages, temperature=0.0, max_tokens=30))
        assert len(tokens) > 0
        full = "".join(tokens)
        assert len(full) > 0

    @deepseek_only
    def test_multi_turn_conversation(self):
        """Multi-turn messages should maintain context."""
        client = LLMClient.deepseek(api_key=os.environ["DEEPSEEK_API_KEY"])
        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name? Reply with just the name."},
        ]
        response = client.generate(messages, temperature=0.0, max_tokens=30)
        assert "Alice" in response

    @deepseek_only
    def test_code_generation_capability(self):
        """LLM should generate correct Python code."""
        client = LLMClient.deepseek(api_key=os.environ["DEEPSEEK_API_KEY"])
        messages = [
            {"role": "user", "content": "Write a Python function that adds two numbers. Output only the code, no explanation."}
        ]
        response = client.generate(messages, temperature=0.0, max_tokens=100)
        assert "def" in response
        assert "return" in response
        assert "add" in response.lower()


# ── Ollama Integration Tests ───────────────────────────────────


class TestOllamaIntegration:
    """Real API calls to local Ollama."""

    @ollama_only
    def test_basic_chat_completion(self):
        """Ollama should return a non-empty response."""
        client = LLMClient.ollama()
        messages = [{"role": "user", "content": "Say hello in one word."}]
        response = client.generate(messages, temperature=0.0, max_tokens=20)
        assert len(response) > 0

    @ollama_only
    def test_streaming(self):
        """Ollama streaming should accumulate correctly."""
        client = LLMClient.ollama()
        messages = [{"role": "user", "content": "Reply with: OK"}]
        response = client.generate(messages, temperature=0.0, max_tokens=10, stream=True)
        assert len(response) > 0

    @ollama_only
    def test_streaming_iterator(self):
        """Ollama generate_stream yields individual tokens."""
        client = LLMClient.ollama()
        messages = [{"role": "user", "content": "Say: hi"}]
        tokens = list(client.generate_stream(messages, temperature=0.0, max_tokens=10))
        assert len(tokens) > 0

    @ollama_only
    def test_code_generation(self):
        """Ollama should generate valid code."""
        client = LLMClient.ollama()
        messages = [
            {"role": "user", "content": "Write a Python function: def square(x): return x*x. Output only the code."}
        ]
        response = client.generate(messages, temperature=0.0, max_tokens=50)
        assert len(response) > 0


# ── Cross-Provider Test ────────────────────────────────────────


class TestCrossProvider:
    """Verify both providers work independently."""

    @requires_any
    def test_any_provider_returns_text(self):
        """Whichever provider is available should produce text."""
        if _deepseek_available():
            client = LLMClient.deepseek(api_key=os.environ["DEEPSEEK_API_KEY"])
        else:
            client = LLMClient.ollama()

        messages = [{"role": "user", "content": "Reply with exactly: PASS"}]
        response = client.generate(messages, temperature=0.0, max_tokens=20)
        assert len(response) > 0


# ── Error Handling Tests ───────────────────────────────────────


class TestErrorHandling:
    """Graceful handling of API errors."""

    def test_invalid_api_key_gives_clear_error(self):
        """Wrong API key should raise from the OpenAI SDK, not crash."""
        # Use a clearly invalid key for DeepSeek URL
        client = LLMClient(LLMConfig(
            base_url="https://api.deepseek.com/v1",
            api_key="invalid-deadbeef-key",
            model="deepseek-chat",
        ))
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(Exception):
            client.generate(messages, temperature=0.0, max_tokens=10)

    def test_unreachable_host_raises(self):
        """If the host is unreachable, an exception should be raised."""
        client = LLMClient(LLMConfig(
            base_url="http://127.0.0.1:19999/v1",  # nothing listening here
            api_key="x",
            model="x",
        ))
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(Exception):
            client.generate(messages, temperature=0.0, max_tokens=10, timeout=2.0)  # type: ignore[call-arg]
