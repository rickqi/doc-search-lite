"""Configuration management for doc-search application."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Supported LLM providers and their litellm prefixes
LLM_PROVIDERS = {
    "glm": {
        "prefix": "zai/",
        "default_model": "glm-4",
        "env_key": "GLM_API_KEY",
        "env_url": "GLM_BASE_URL",
        "default_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "deepseek": {
        "prefix": "deepseek/",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "env_url": "DEEPSEEK_BASE_URL",
        "default_url": "https://api.deepseek.com",
    },
}


@dataclass
class Config:
    """Configuration dataclass for doc-search application.

    Attributes:
        glm_api_key: GLM API key for authentication (also used for Rerank)
        glm_base_url: Base URL for GLM API
        llm_provider: Active LLM provider ("glm" or "deepseek")
        llm_model: LLM model name (default depends on provider)
        llm_temperature: LLM temperature parameter (default: 0.7)
        llm_max_tokens: Maximum tokens for LLM responses (default: 4096)
        deepseek_api_key: DeepSeek API key (optional, only if provider=deepseek)
        deepseek_base_url: DeepSeek base URL (optional)
        search_default_limit: Default search result limit (default: 10)
        search_bm25_k1: BM25 k1 parameter (default: 1.5)
        search_bm25_b: BM25 b parameter (default: 0.75)
        index_path: Path to the search index
        log_level: Logging level (default: "INFO")
        max_workers: Maximum number of workers (default: 4)
    """

    glm_api_key: str
    glm_base_url: str
    llm_provider: str = "glm"
    llm_model: str = "glm-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    search_default_limit: int = 10
    search_bm25_k1: float = 1.5
    search_bm25_b: float = 0.75
    index_path: str = ""
    log_level: str = "INFO"
    max_workers: int = 4

    @property
    def active_api_key(self) -> str:
        """Get the API key for the active LLM provider."""
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key or self.glm_api_key
        return self.glm_api_key

    @property
    def active_base_url(self) -> str:
        """Get the base URL for the active LLM provider."""
        if self.llm_provider == "deepseek":
            return self.deepseek_base_url or LLM_PROVIDERS["deepseek"]["default_url"]
        return self.glm_base_url

    @property
    def litellm_model(self) -> str:
        """Get the model name in LiteLLM format (with provider prefix)."""
        provider_cfg = LLM_PROVIDERS.get(self.llm_provider, LLM_PROVIDERS["glm"])
        prefix = provider_cfg["prefix"]
        model = self.llm_model
        if not model.startswith(prefix):
            model = f"{prefix}{model}"
        return model

    @property
    def provider(self) -> str:
        """Get the active LLM provider name."""
        return self.llm_provider

    @property
    def fast_model(self) -> str:
        """Get the fast/cheap model for intermediate steps."""
        env_model = os.getenv("LLM_FAST_MODEL", "")
        if env_model:
            prefix = "deepseek/" if not env_model.startswith(("deepseek/", "zai/")) else ""
            return f"{prefix}{env_model}"
        # Default: use current provider's flash model
        if self.llm_provider == "deepseek":
            return "deepseek/deepseek-v4-flash"  # Fast tier
        return self.litellm_model  # Fallback to same model

    @property
    def power_model(self) -> str:
        """Get the power model for final answers."""
        env_model = os.getenv("LLM_POWER_MODEL", "")
        if env_model:
            prefix = "deepseek/" if not env_model.startswith(("deepseek/", "zai/")) else ""
            return f"{prefix}{env_model}"
        if self.llm_provider == "deepseek":
            return "deepseek/deepseek-v4-pro"  # Pro tier
        return self.litellm_model  # Default to same model

    @property
    def use_tiered_routing(self) -> bool:
        """Whether tiered routing is enabled."""
        return os.getenv("LLM_TIERED_ROUTING", "").lower() in ("1", "true", "yes")

    @classmethod
    def from_env(cls, dotenv_path: Optional[Path] = None) -> "Config":
        """Create Config from environment variables.

        Automatically loads .env file from project root if present.
        Explicit environment variables take precedence over .env values.

        Reads:
        - GLM_API_KEY: GLM API key for authentication (required)
        - GLM_BASE_URL: GLM Base URL for API (required)
        - LLM_PROVIDER: Provider to use: "glm" or "deepseek" (default: glm)
        - LLM_MODEL: Model name (default depends on provider)
        - LLM_TEMPERATURE: Temperature (default: 0.7)
        - LLM_MAX_TOKENS: Max tokens (default: 4096)
        - DEEPSEEK_API_KEY: DeepSeek API key (required if provider=deepseek)
        - DEEPSEEK_BASE_URL: DeepSeek base URL (optional)
        - SEARCH_DEFAULT_LIMIT: Default result limit (default: 10)
        - LOG_LEVEL: Logging level (default: INFO)
        - MAX_WORKERS: Worker count (default: 4)

        Args:
            dotenv_path: Optional explicit path to .env file.
                         Defaults to project root .env.

        Returns:
            Config instance loaded from environment variables

        Raises:
            ValueError: If required environment variables are missing
        """
        # Load .env file (project root by default)
        if dotenv_path is not None:
            load_dotenv(dotenv_path, override=False)
        else:
            # Try project root .env (where pyproject.toml lives)
            project_root = Path(__file__).resolve().parent.parent.parent
            load_dotenv(project_root / ".env", override=False)

        api_key = os.environ.get("GLM_API_KEY")
        if not api_key:
            raise ValueError(
                "GLM_API_KEY not found. "
                "Set it in .env file or as environment variable."
            )

        base_url = os.environ.get("GLM_BASE_URL")
        if not base_url:
            raise ValueError(
                "GLM_BASE_URL not found. "
                "Set it in .env file or as environment variable."
            )

        # Determine provider
        provider = os.environ.get("LLM_PROVIDER", "glm").lower()
        if provider not in LLM_PROVIDERS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. "
                f"Supported: {list(LLM_PROVIDERS.keys())}"
            )

        # Default model depends on provider
        provider_cfg = LLM_PROVIDERS[provider]
        default_model = provider_cfg["default_model"]

        # Validate DeepSeek key if that provider is selected
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        deepseek_url = os.environ.get("DEEPSEEK_BASE_URL", "")
        if provider == "deepseek" and not deepseek_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. "
                "Set it in .env file or as environment variable "
                "when LLM_PROVIDER=deepseek."
            )

        return cls(
            glm_api_key=api_key,
            glm_base_url=base_url,
            llm_provider=provider,
            llm_model=os.environ.get("LLM_MODEL", default_model),
            llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "4096")),
            deepseek_api_key=deepseek_key,
            deepseek_base_url=deepseek_url,
            search_default_limit=int(
                os.environ.get("SEARCH_DEFAULT_LIMIT", "10")
            ),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            max_workers=int(os.environ.get("MAX_WORKERS", "4")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Create Config from YAML configuration file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance loaded from YAML file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required fields are missing or invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid config file format: {path}")

        # Check required fields
        if "glm_api_key" not in data:
            raise ValueError("Missing required field: glm_api_key")
        if "glm_base_url" not in data:
            raise ValueError("Missing required field: glm_base_url")

        # Determine provider
        provider = data.get("llm_provider", "glm").lower()
        if provider not in LLM_PROVIDERS:
            raise ValueError(
                f"Unknown llm_provider '{provider}'. "
                f"Supported: {list(LLM_PROVIDERS.keys())}"
            )
        provider_cfg = LLM_PROVIDERS[provider]
        default_model = provider_cfg["default_model"]

        # Extract fields with defaults
        return cls(
            glm_api_key=data["glm_api_key"],
            glm_base_url=data["glm_base_url"],
            llm_provider=provider,
            llm_model=data.get("llm_model", default_model),
            llm_temperature=data.get("llm_temperature", 0.7),
            llm_max_tokens=data.get("llm_max_tokens", 4096),
            deepseek_api_key=data.get("deepseek_api_key", ""),
            deepseek_base_url=data.get("deepseek_base_url", ""),
            search_default_limit=data.get("search_default_limit", 10),
            search_bm25_k1=data.get("search_bm25_k1", 1.5),
            search_bm25_b=data.get("search_bm25_b", 0.75),
            index_path=data.get("index_path", ""),
            log_level=data.get("log_level", "INFO"),
            max_workers=data.get("max_workers", 4),
        )

    def merge(self, other: "Config") -> "Config":
        """Merge this config with another config.

        The other config's values take precedence over this config's values.

        Args:
            other: Another Config instance to merge with

        Returns:
            New Config instance with merged values
        """
        def _prefer(default_val, other_val, sentinel):
            """Pick other_val if it differs from sentinel, else default_val."""
            return other_val if other_val != sentinel else default_val

        return Config(
            glm_api_key=other.glm_api_key or self.glm_api_key,
            glm_base_url=other.glm_base_url or self.glm_base_url,
            llm_provider=other.llm_provider if other.llm_provider != "glm" else self.llm_provider,
            llm_model=_prefer(self.llm_model, other.llm_model, "glm-4"),
            llm_temperature=_prefer(self.llm_temperature, other.llm_temperature, 0.7),
            llm_max_tokens=_prefer(self.llm_max_tokens, other.llm_max_tokens, 4096),
            deepseek_api_key=other.deepseek_api_key or self.deepseek_api_key,
            deepseek_base_url=other.deepseek_base_url or self.deepseek_base_url,
            search_default_limit=_prefer(self.search_default_limit, other.search_default_limit, 10),
            search_bm25_k1=_prefer(self.search_bm25_k1, other.search_bm25_k1, 1.5),
            search_bm25_b=_prefer(self.search_bm25_b, other.search_bm25_b, 0.75),
            index_path=other.index_path or self.index_path,
            log_level=_prefer(self.log_level, other.log_level, "INFO"),
            max_workers=_prefer(self.max_workers, other.max_workers, 4),
        )
