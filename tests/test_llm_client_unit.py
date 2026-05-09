"""
Unit tests for LLMClient — config parsing, construction, parameter handling.
These tests do NOT require real API keys or network access.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from core.llm_client import LLMClient, LLMConfig


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory with a valid config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        yield Path(tmpdir)
        os.chdir(original_cwd)


@pytest.fixture
def deepseek_config_file(temp_config_dir):
    """Write a valid DeepSeek config.yaml and return its path."""
    config_data = {
        "active": "deepseek",
        "providers": {
            "deepseek": {
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "${DEEPSEEK_API_KEY}",
                "model": "deepseek-chat",
            },
            "ollama": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "qwen3:8b",
            },
        },
        "generation": {"temperature": 0.0, "max_tokens": 2048, "stream": False},
    }
    config_path = temp_config_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def ollama_active_config(temp_config_dir):
    """Config with ollama as active provider."""
    config_data = {
        "active": "ollama",
        "providers": {
            "deepseek": {
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "${DEEPSEEK_API_KEY}",
                "model": "deepseek-chat",
            },
            "ollama": {
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "qwen3:8b",
            },
        },
    }
    config_path = temp_config_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


# ── LLMConfig Tests ───────────────────────────────────────────


class TestLLMConfig:
    """LLMConfig dataclass: stores provider info without any URL hardcoding."""

    def test_config_stores_all_fields(self):
        cfg = LLMConfig(
            base_url="https://custom.api.com/v1",
            api_key="sk-test-123",
            model="test-model",
            provider_name="custom",
            temperature=0.7,
            max_tokens=4096,
        )
        assert cfg.base_url == "https://custom.api.com/v1"
        assert cfg.api_key == "sk-test-123"
        assert cfg.model == "test-model"
        assert cfg.provider_name == "custom"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096

    def test_config_defaults(self):
        cfg = LLMConfig(base_url="http://x", api_key="k", model="m")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 2048
        assert cfg.provider_name == "custom"

    def test_config_no_hardcoded_openai_url(self):
        """CRITICAL: URL field must accept any value, not default to OpenAI."""
        cfg1 = LLMConfig(base_url="https://api.deepseek.com/v1", api_key="k", model="m")
        assert "openai.com" not in cfg1.base_url

        cfg2 = LLMConfig(base_url="http://localhost:11434/v1", api_key="k", model="m")
        assert "openai.com" not in cfg2.base_url

        cfg3 = LLMConfig(base_url="https://custom.internal/v1", api_key="k", model="m")
        assert "openai.com" not in cfg3.base_url


# ── LLMClient Construction Tests ──────────────────────────────


class TestLLMClientConstruction:
    """LLMClient can be built via constructor, from_config, from_env, and factory methods."""

    def test_direct_constructor(self):
        cfg = LLMConfig(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-test",
            model="deepseek-chat",
        )
        client = LLMClient(cfg)
        assert client.model == "deepseek-chat"
        assert client.config.base_url == "https://api.deepseek.com/v1"
        assert client.config.provider_name == "custom"

    def test_from_config_deepseek(self, deepseek_config_file):
        with open(deepseek_config_file) as f:
            pass  # Verify file exists
        # Set env so ${DEEPSEEK_API_KEY} resolves
        os.environ["DEEPSEEK_API_KEY"] = "sk-test-deepseek"
        try:
            client = LLMClient.from_config(str(deepseek_config_file))
            assert client.config.provider_name == "deepseek"
            assert client.model == "deepseek-chat"
            assert "api.deepseek.com" in client.config.base_url
        finally:
            del os.environ["DEEPSEEK_API_KEY"]

    def test_from_config_ollama(self, ollama_active_config):
        client = LLMClient.from_config(str(ollama_active_config))
        assert client.config.provider_name == "ollama"
        assert client.model == "qwen3:8b"
        assert "localhost:11434" in client.config.base_url

    def test_from_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            LLMClient.from_config("/nonexistent/config.yaml")

    def test_from_config_invalid_active_provider(self, temp_config_dir):
        config_data = {
            "active": "nonexistent",
            "providers": {"deepseek": {"base_url": "x", "api_key": "k", "model": "m"}},
        }
        config_path = temp_config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="Active provider"):
            LLMClient.from_config(str(config_path))

    def test_from_env_defaults_to_ollama(self):
        """Without env vars, from_env should default to local Ollama."""
        # Clean env
        for var in ("CODEPIPE_BASE_URL", "CODEPIPE_API_KEY", "CODEPIPE_MODEL"):
            os.environ.pop(var, None)

        client = LLMClient.from_env()
        assert client.config.provider_name == "ollama"
        assert "localhost:11434" in client.config.base_url
        assert client.model in ("qwen3:8b", "qwen3")

    def test_from_env_custom_vars(self):
        os.environ["CODEPIPE_BASE_URL"] = "https://api.custom.com/v1"
        os.environ["CODEPIPE_API_KEY"] = "sk-custom"
        os.environ["CODEPIPE_MODEL"] = "custom-model"
        try:
            client = LLMClient.from_env()
            assert client.config.base_url == "https://api.custom.com/v1"
            assert client.model == "custom-model"
        finally:
            for var in ("CODEPIPE_BASE_URL", "CODEPIPE_API_KEY", "CODEPIPE_MODEL"):
                os.environ.pop(var, None)

    def test_from_env_partial_overrides(self):
        """If BASE_URL is set but MODEL is not, model should be 'default'."""
        os.environ["CODEPIPE_BASE_URL"] = "https://api.siliconflow.cn/v1"
        os.environ.pop("CODEPIPE_MODEL", None)
        os.environ.pop("CODEPIPE_API_KEY", None)
        try:
            client = LLMClient.from_env()
            assert "siliconflow" in client.config.base_url
            assert client.model == "default"
            assert client.config.api_key == "not-needed"
        finally:
            os.environ.pop("CODEPIPE_BASE_URL", None)

    def test_factory_deepseek(self):
        client = LLMClient.deepseek(api_key="sk-test-ds", model="deepseek-chat")
        assert client.config.provider_name == "deepseek"
        assert "api.deepseek.com" in client.config.base_url
        assert client.model == "deepseek-chat"

    def test_factory_ollama_default_model(self):
        client = LLMClient.ollama()
        assert client.config.provider_name == "ollama"
        assert "localhost:11434" in client.config.base_url
        assert client.model == "qwen3:8b"

    def test_factory_ollama_custom_model_and_host(self):
        client = LLMClient.ollama(
            model="codellama:7b",
            host="http://192.168.1.100:11434/v1",
        )
        assert client.model == "codellama:7b"
        assert "192.168.1.100:11434" in client.config.base_url


# ── ENV Variable Resolution Tests ──────────────────────────────


class TestEnvResolution:
    """The ${VAR} placeholder system in config values."""

    def test_resolves_env_var(self):
        os.environ["MY_KEY"] = "secret-abc"
        try:
            result = LLMClient._resolve_env("the key is ${MY_KEY}")
            assert result == "the key is secret-abc"
        finally:
            del os.environ["MY_KEY"]

    def test_unset_env_var_resolves_to_empty(self):
        result = LLMClient._resolve_env("${NONEXISTENT_VAR}")
        assert result == ""

    def test_no_placeholders_passthrough(self):
        result = LLMClient._resolve_env("plain text no vars")
        assert result == "plain text no vars"

    def test_multiple_placeholders(self):
        os.environ["A"] = "1"
        os.environ["B"] = "2"
        try:
            result = LLMClient._resolve_env("${A} and ${B}")
            assert result == "1 and 2"
        finally:
            del os.environ["A"], os.environ["B"]


# ── Provider Switching Tests ──────────────────────────────────


class TestProviderSwitching:
    """Verify seamless hot-switching between providers."""

    def test_build_deepseek_then_ollama(self, temp_config_dir):
        """Both clients should coexist with different configs."""
        ds = LLMClient.deepseek(api_key="sk-x", model="deepseek-chat")
        ol = LLMClient.ollama(model="qwen3:8b")

        assert "deepseek" in ds.config.base_url
        assert "localhost" in ol.config.base_url
        assert ds.model != ol.model

    def test_config_file_switching(self, deepseek_config_file, temp_config_dir):
        """Switching active: in config should change loaded client."""
        # Load with deepseek active
        os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        try:
            ds_client = LLMClient.from_config(str(deepseek_config_file))
            assert ds_client.config.provider_name == "deepseek"

            # Update config to switch to ollama
            with open(deepseek_config_file) as f:
                data = yaml.safe_load(f)
            data["active"] = "ollama"
            with open(deepseek_config_file, "w") as f:
                yaml.dump(data, f)

            ol_client = LLMClient.from_config(str(deepseek_config_file))
            assert ol_client.config.provider_name == "ollama"
        finally:
            del os.environ["DEEPSEEK_API_KEY"]


# ── Representation ────────────────────────────────────────────


class TestRepr:
    def test_repr_does_not_leak_api_key(self):
        client = LLMClient.deepseek(api_key="sk-super-secret-key-12345", model="deepseek-chat")
        rep = repr(client)
        assert "sk-super-secret-key-12345" not in rep
        assert "deepseek" in rep
        assert "deepseek-chat" in rep

    def test_repr_shows_provider_info(self):
        client = LLMClient.ollama()
        rep = repr(client)
        assert "ollama" in rep
        assert "qwen3" in rep
