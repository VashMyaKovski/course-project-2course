import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class LlamaFactExtractorSettings:
    # URL именно с /v1, потому что extractor работает через /chat/completions.
    api_base_url: str
    api_key: str
    model: str
    timeout_sec: float
    temperature: float


@dataclass(frozen=True)
class AppSettings:
    llama_fact_extractor: LlamaFactExtractorSettings


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Централизованная загрузка конфигурации из окружения.
    Приоритет стандартный: переменные процесса > .env файл.
    """
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    env_dev_path = project_root / ".env.dev"
    # override=False важен: не затираем значения, если они уже пришли из окружения.
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    elif env_dev_path.exists():
        load_dotenv(dotenv_path=env_dev_path, override=False)

    llama = LlamaFactExtractorSettings(
        api_base_url=os.getenv("LLAMA_CHAT_API_BASE", "http://127.0.0.1:11434/v1").rstrip("/"),
        api_key=os.getenv("LLAMA_CHAT_API_KEY", "ollama"),
        model=os.getenv("LLAMA_CHAT_MODEL", "llama3.1:8b"),
        timeout_sec=_to_float(os.getenv("LLAMA_CHAT_TIMEOUT_SEC"), 120.0),
        temperature=_to_float(os.getenv("LLAMA_CHAT_TEMPERATURE"), 0.2),
    )

    return AppSettings(llama_fact_extractor=llama)
