import json
import unittest
import urllib.error
import urllib.request

from config import get_settings
from src.fact_extractor import Llama31InstructChatCompletionFactExtractor


class TestFactExtractorLive(unittest.TestCase):
    """Интеграционные проверки against живой Ollama/OpenAI-compatible endpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        settings = get_settings().llama_fact_extractor
        cls._base_url = settings.api_base_url
        cls._model = settings.model

        # Тест должен быть удобным для локального запуска: если сервер не поднят, просто skip.
        version_url = cls._base_url.replace("/v1", "/api/version")
        req = urllib.request.Request(version_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=2):
                pass
        except (urllib.error.URLError, urllib.error.HTTPError):
            raise unittest.SkipTest(
                f"LLM endpoint недоступен: {version_url}. Подними Ollama и повтори."
            )

    def test_extract_returns_non_empty_list(self) -> None:
        chunk = (
            "Я руковожу центром разработки клиентских и аналитических решений — "
            "50+ человек в семи кросс-функциональных командах с разными стеками. "
            "Аналитики, разработчики, QA, DevOps, сопровождение — всё в одной цепочке, "
            "от детализации требований до эксплуатации."
        )

        print(f'Chunk: \n',chunk)

        extractor = Llama31InstructChatCompletionFactExtractor()
        facts = extractor.extract(chunk)

        print(f'Facts: \n',facts)

        self.assertIsInstance(facts, list)
        self.assertGreater(len(facts), 0)
        self.assertTrue(all(isinstance(fact, str) and fact.strip() for fact in facts))

    def test_extract_facts_start_with_author(self) -> None:
        chunk = (
            "Я отвечаю за развитие платформы и управляю командой аналитики. "
            "Мы внедрили сквозной процесс поставки от требований до продакшена."
        )

        print(f'Chunk: \n', chunk)

        extractor = Llama31InstructChatCompletionFactExtractor()
        facts = extractor.extract(chunk)

        print(f'Facts: \n', facts)

        self.assertGreater(len(facts), 0)
        invalid = [fact for fact in facts if not fact.startswith("Автор")]
        self.assertEqual(
            [],
            invalid,
            msg=(
                "Ожидались факты с явным субъектом «Автор». "
                f"Модель={self._model}, факты={json.dumps(facts, ensure_ascii=False)}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
