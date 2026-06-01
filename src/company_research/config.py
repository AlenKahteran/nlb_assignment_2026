from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[no-redef]
        """Fallback no-op when python-dotenv is not installed."""
        return False

from .models import AppConfig, Budget, LLMConfig, QAConfig

logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    value = os.getenv(name)

    if value is None or value == "":
        logger.debug("Using default integer config %s=%s", name, default)
        return default

    try:
        parsed = int(value)
        logger.debug("Loaded integer config %s=%s", name, parsed)
        return parsed
    except ValueError as exc:
        logger.error("Invalid integer config %s=%r", name, value)
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    """Read a float environment variable, falling back to a default."""
    value = os.getenv(name)

    if value is None or value == "":
        logger.debug("Using default float config %s=%s", name, default)
        return default

    try:
        parsed = float(value)
        logger.debug("Loaded float config %s=%s", name, parsed)
        return parsed
    except ValueError as exc:
        logger.error("Invalid float config %s=%r", name, value)
        raise ValueError(f"{name} must be a number") from exc


def load_config(env_file: str | Path | None = ".env") -> AppConfig:
    """Load application configuration from environment variables and an optional .env file."""
    if env_file:
        path = Path(env_file)

        if path.exists():
            logger.info("Loading configuration from %s", path)
            load_dotenv(path, override=False)
        else:
            logger.debug("Configuration file %s not found; using environment/defaults", path)

    budget = Budget(
        max_search_results=_int_env("MAX_SEARCH_RESULTS", 100),
        max_visited_pages=_int_env("MAX_VISITED_PAGES", 50),
        max_downloads=_int_env("MAX_DOWNLOADS", 16),
        max_depth=_int_env("MAX_DEPTH", 10),
        max_runtime_seconds=_int_env("MAX_RUNTIME_SECONDS", 600),
        crawl_delay_seconds=_float_env("CRAWL_DELAY_SECONDS", 1.0),
        max_file_mb=_int_env("MAX_FILE_MB", 50),
        http_timeout_seconds=_int_env("HTTP_TIMEOUT_SECONDS", 30),
    )
    llm = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", ""),
        endpoint=os.getenv("LLM_ENDPOINT", ""),
        api_key=os.getenv("LLM_API_KEY", ""),
        model_name=os.getenv("LLM_MODEL_NAME", ""),
        timeout_seconds=_int_env("LLM_TIMEOUT_SECONDS", 60),
        max_retries=_int_env("LLM_MAX_RETRIES", 2),
        mode=os.getenv("LLM_MODE", "provider") or "provider",
    )
    qa = QAConfig(
        fact_evidence_limit=_int_env("QA_FACT_EVIDENCE_LIMIT", 25),
        chunk_evidence_limit=_int_env("QA_CHUNK_EVIDENCE_LIMIT", 25),
    )

    config = AppConfig(
        search_provider=os.getenv("SEARCH_PROVIDER", "brave") or "brave",
        brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
        search_fixture_path=os.getenv("SEARCH_FIXTURE_PATH", ""),
        budget=budget,
        llm=llm,
        qa=qa,
    )
    logger.info(
        "Loaded config search_provider=%s fixture=%s llm_mode=%s",
        config.search_provider,
        bool(config.search_fixture_path),
        config.llm.mode,
    )
    logger.debug("Redacted config: %s", config.redacted())

    return config
