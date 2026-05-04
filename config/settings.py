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
class QdrantSettings:
    """Параметры Qdrant для атомарных утверждений (вектор + payload)."""

    url: str
    api_key: str | None
    collection_name: str
    vector_size: int
    distance: str
    timeout_sec: float
    search_top_k: int
    score_threshold: float | None
    upsert_batch_size: int
    payload_text_key: str
    payload_document_id_key: str
    payload_author_key: str


@dataclass(frozen=True)
class AppSettings:
    llama_fact_extractor: LlamaFactExtractorSettings
    qdrant: QdrantSettings


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _optional_api_key(value: str | None) -> str | None:
    if value is None or value.strip() == "":
        return None
    return value.strip()


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

    qdrant = QdrantSettings(
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/"),
        api_key=_optional_api_key(os.getenv("QDRANT_API_KEY")),
        collection_name=os.getenv("QDRANT_COLLECTION_NAME", "atomic_statements"),
        vector_size=_to_int(os.getenv("QDRANT_VECTOR_SIZE"), 384),
        distance=os.getenv("QDRANT_DISTANCE", "COSINE").strip().upper(),
        timeout_sec=_to_float(os.getenv("QDRANT_TIMEOUT_SEC"), 30.0),
        search_top_k=_to_int(os.getenv("QDRANT_SEARCH_TOP_K"), 15),
        score_threshold=_optional_float(os.getenv("QDRANT_SCORE_THRESHOLD")),
        upsert_batch_size=max(1, _to_int(os.getenv("QDRANT_UPSERT_BATCH_SIZE"), 128)),
        payload_text_key=os.getenv("QDRANT_PAYLOAD_KEY_TEXT", "text"),
        payload_document_id_key=os.getenv("QDRANT_PAYLOAD_KEY_DOCUMENT_ID", "document_id"),
        payload_author_key=os.getenv("QDRANT_PAYLOAD_KEY_AUTHOR", "author"),
    )

    return AppSettings(llama_fact_extractor=llama, qdrant=qdrant)
