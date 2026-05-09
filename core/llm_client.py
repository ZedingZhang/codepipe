"""
LLMClient — provider-agnostic unified driver.

Supports any OpenAI-compatible endpoint: DeepSeek API, Ollama, etc.
No provider URL is hardcoded. All configuration comes from constructor params,
config.yaml, or environment variables.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Provider-agnostic LLM configuration. No URLs are hardcoded here."""

    base_url: str
    api_key: str
    model: str
    provider_name: str = "custom"

    # Generation defaults
    temperature: float = 0.0
    max_tokens: int = 2048


class LLMClient:
    """
    Unified LLM driver wrapping the OpenAI SDK.

    Works with any OpenAI-compatible API:
        - DeepSeek:      base_url="https://api.deepseek.com/v1"
        - Ollama:        base_url="http://localhost:11434/v1", api_key="ollama"
        - SiliconFlow:   base_url="https://api.siliconflow.cn/v1"
        - Any custom:    base_url="https://your-provider.com/v1"

    Three instantiation paths (in priority order):
        1. Direct constructor:  LLMClient(LLMConfig(...))
        2. from_config():       reads config.yaml
        3. from_env():          reads CODEPIPE_BASE_URL / CODEPIPE_API_KEY / CODEPIPE_MODEL
    """

    # Regex to resolve ${ENV_VAR} placeholders in config values
    _ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")

    def __init__(self, config: LLMConfig):
        self.config = config
        resolved_url = self._resolve_env(config.base_url)
        resolved_key = self._resolve_env(config.api_key)
        self._client = OpenAI(base_url=resolved_url, api_key=resolved_key)
        self._model = config.model
        logger.info(
            "LLMClient initialized: provider=%s model=%s base_url=%s",
            config.provider_name, config.model, resolved_url,
        )

    # ── Public API ─────────────────────────────────────────────

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate a completion. Returns the text content.

        Args:
            messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: override config default if provided
            max_tokens: override config default if provided
            stream: if True, stream chunks and concatenate

        Returns:
            The assistant's text response.
        """
        temp = temperature if temperature is not None else self.config.temperature
        mt = max_tokens if max_tokens is not None else self.config.max_tokens

        if stream:
            return self._generate_stream(messages, temp, mt, **kwargs)
        else:
            return self._generate_sync(messages, temp, mt, **kwargs)

    def generate_stream(self, messages: list[dict[str, str]], **kwargs):
        """
        Stream tokens one at a time. Yields content delta strings.

        Usage:
            for token in client.generate_stream(messages):
                print(token, end="", flush=True)
        """
        temp = kwargs.pop("temperature", self.config.temperature)
        mt = kwargs.pop("max_tokens", self.config.max_tokens)
        stream_obj: Stream[ChatCompletionChunk] = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temp,
            max_tokens=mt,
            stream=True,
            **kwargs,
        )
        for chunk in stream_obj:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── Internal ───────────────────────────────────────────────

    def _generate_sync(self, messages, temperature, max_tokens, **kwargs) -> str:
        response: ChatCompletion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _generate_stream(self, messages, temperature, max_tokens, **kwargs) -> str:
        """Stream and accumulate into a single string."""
        parts: list[str] = []
        for token in self.generate_stream(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        ):
            parts.append(token)
        return "".join(parts)

    # ── Factory constructors ──────────────────────────────────

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "config.yaml") -> LLMClient:
        """
        Build client from a YAML config file.

        config.yaml structure:
            active: deepseek
            providers:
              deepseek:
                base_url: "https://api.deepseek.com/v1"
                api_key: "${DEEPSEEK_API_KEY}"
                model: "deepseek-chat"
              ollama:
                base_url: "http://localhost:11434/v1"
                api_key: "ollama"
                model: "qwen3:8b"
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        active_name = data.get("active", "deepseek")
        providers = data.get("providers", {})
        if active_name not in providers:
            available = list(providers.keys())
            raise ValueError(
                f"Active provider '{active_name}' not found in providers. "
                f"Available: {available}"
            )

        provider_cfg = providers[active_name]
        gen_cfg = data.get("generation", {})

        config = LLMConfig(
            base_url=provider_cfg["base_url"],
            api_key=provider_cfg["api_key"],
            model=provider_cfg["model"],
            provider_name=active_name,
            temperature=float(gen_cfg.get("temperature", 0.0)),
            max_tokens=int(gen_cfg.get("max_tokens", 2048)),
        )
        return cls(config)

    @classmethod
    def from_env(cls) -> LLMClient:
        """
        Build client from environment variables.
        Reads: CODEPIPE_BASE_URL, CODEPIPE_API_KEY, CODEPIPE_MODEL
        Falls back to Ollama local defaults if not set.
        """
        base_url = os.environ.get("CODEPIPE_BASE_URL")
        api_key = os.environ.get("CODEPIPE_API_KEY")
        model = os.environ.get("CODEPIPE_MODEL")

        if not base_url:
            # Sensible default: local Ollama
            base_url = "http://localhost:11434/v1"
            api_key = api_key or "ollama"
            model = model or "qwen3:8b"
            provider_name = "ollama"
        else:
            api_key = api_key or "not-needed"
            model = model or "default"
            provider_name = "env"

        config = LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            provider_name=provider_name,
        )
        return cls(config)

    @classmethod
    def deepseek(cls, api_key: str, model: str = "deepseek-chat") -> LLMClient:
        """Convenience: DeepSeek API client."""
        config = LLMConfig(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
            model=model,
            provider_name="deepseek",
        )
        return cls(config)

    @classmethod
    def ollama(cls, model: str = "qwen3:8b", host: str = "http://localhost:11434/v1") -> LLMClient:
        """Convenience: local Ollama client."""
        config = LLMConfig(
            base_url=host,
            api_key="ollama",
            model=model,
            provider_name="ollama",
        )
        return cls(config)

    # ── Utilities ──────────────────────────────────────────────

    @staticmethod
    def _resolve_env(value: str) -> str:
        """Replace ${VAR} placeholders with environment variable values."""
        def _replacer(m: re.Match) -> str:
            var_name = m.group(1)
            return os.environ.get(var_name, "")
        return LLMClient._ENV_VAR_PATTERN.sub(_replacer, value)

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self.config.provider_name!r}, "
            f"model={self._model!r}, "
            f"base_url={self.config.base_url!r})"
        )
