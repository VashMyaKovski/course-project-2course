import json
import re
import urllib.error
import urllib.request
from typing import Any

from config import get_settings
from src.fact_extractor.base import FactExtractorBase
from src.fact_extractor.enums_llama_instruct_chat import LlamaChatAPI, LlamaChatPrompt


class Llama31InstructChatCompletionFactExtractor(FactExtractorBase):
    """
    Извлечение фактов через Chat Completions (instruction/chat) против Meta Llama 3.1 8B Instruct.

    Ожидается совместимый с OpenAI HTTP API эндпоинт (/v1/chat/completions): Ollama, vLLM,
    локальный прокси, облако с таким интерфейсом.

    Переменные окружения: LLAMA_CHAT_API_BASE (например http://127.0.0.1:11434/v1),
    LLAMA_CHAT_API_KEY, LLAMA_CHAT_MODEL. У Ollama имя модели задайте тегом, например llama3.1.
    """

    def __init__(
        self,
        *,
        api_base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings().llama_fact_extractor
        # Явные параметры в конструкторе всегда имеют приоритет над .env.
        self._api_base_url = (api_base_url or settings.api_base_url).rstrip("/")
        self._api_key = api_key if api_key is not None else settings.api_key
        self._model = model or settings.model
        self._timeout_sec = timeout_sec if timeout_sec is not None else settings.timeout_sec
        self._temperature = temperature if temperature is not None else settings.temperature

    def extract(self, chunk: str) -> list[str]:
        text = chunk.strip()
        if not text:
            return []

        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": LlamaChatPrompt.SYSTEM.value},
                {"role": "user", "content": LlamaChatPrompt.USER_TEMPLATE.value.format(chunk=text)},
            ],
        }

        raw = self._post_chat(payload)
        facts = _parse_facts_json(raw)
        return _ensure_author_subject_if_needed(text, facts)

    def _post_chat(self, payload: dict[str, Any]) -> str:
        url = f"{self._api_base_url}{LlamaChatAPI.COMPLETIONS_PATH.value}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_sec) as resp:
                decoded = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Chat completion HTTP {e.code}: {detail}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Не удалось достучаться до LLM API: {e}") from e

        data = json.loads(decoded)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Неожиданный ответ Chat Completions: {data!r}") from e


_json_array_pat = re.compile(r"\[[\s\S]*\]")


def _parse_facts_json(llm_content: str) -> list[str]:
    """Вытащить массив строк из ответа модели — с учётом редких огрех формата."""

    stripped = llm_content.strip()

    candidates: list[str] = []

    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", stripped, re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))

    bracket = _json_array_pat.search(stripped)
    if bracket:
        candidates.append(bracket.group(0))

    candidates.append(stripped)

    for blob in candidates:
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, list):
                facts = [_normalize_fact(item) for item in parsed]
                return [x for x in facts if x]
        except json.JSONDecodeError:
            continue

    # Fallback: построчно, если модель игнорирует JSON
    lines = [
        re.sub(r"^[-*•]\s*", "", ln).strip()
        for ln in stripped.splitlines()
        if ln.strip()
    ]
    return [ln for ln in lines if ln]


def _normalize_fact(item: Any) -> str:
    return str(item).strip() if item is not None else ""


def _ensure_author_subject_if_needed(chunk: str, facts: list[str]) -> list[str]:
    """
    Если исходный чанк про автора (я/мы), приводим каждый факт к явному субъекту «Автор».
    Это стабилизирует формат для последующих этапов пайплайна.
    """
    if not re.search(r"\b(я|мы)\b", chunk, flags=re.IGNORECASE):
        return facts

    normalized: list[str] = []
    for fact in facts:
        cleaned = fact.strip()
        if not cleaned:
            continue
        if cleaned.startswith("Автор"):
            normalized.append(cleaned)
            continue
        # Нормализуем регистр начала, чтобы получить читаемое "Автор ...".
        normalized.append(f"Автор {cleaned[:1].lower()}{cleaned[1:]}")
    return normalized
