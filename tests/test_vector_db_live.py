import os
import time
import unittest
import uuid
from dataclasses import dataclass
from functools import wraps


def timed_test(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"\n[vector-db-live] '{func.__name__}' done in {duration:.3f} sec")
        return result

    return wrapper


class TestVectorDBLive(unittest.TestCase):
    """
    Интеграционные тесты: list[str] -> EmbeddingGenerator -> Qdrant upsert/search.

    Тесты скипаются, если:
    - недоступен Qdrant на QDRANT_URL;
    - недоступны зависимости эмбеддингов или модель не загрузилась.
    """

    @dataclass(frozen=True)
    class _Record:
        point_id: str | int
        embedding: list[float]
        text: str
        document_id: str
        author: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._old_collection = os.getenv("QDRANT_COLLECTION_NAME")
        cls._old_vector_size = os.getenv("QDRANT_VECTOR_SIZE")

        try:
            from config import get_settings
            from qdrant_client.http.exceptions import ResponseHandlingException as _RespExc
            from src.vector_db.collection import ensure_atomic_statements_collection, make_qdrant_client
            from src.vector_db.store import AtomicStatementsStore
        except Exception as exc:
            raise unittest.SkipTest(
                f"Зависимости для live-теста недоступны (пакеты не установлены?): {exc}",
            ) from exc
        cls._get_settings = get_settings
        cls._resp_exc = _RespExc
        cls._ensure_collection = ensure_atomic_statements_collection
        cls._make_client = make_qdrant_client
        cls._store_cls = AtomicStatementsStore

        try:
            from src.embeddings.generator import EmbeddingGenerator
        except Exception as exc:
            raise unittest.SkipTest(f"Не удалось импортировать EmbeddingGenerator: {exc}") from exc

        try:
            cls._embedder = EmbeddingGenerator()
        except Exception as exc:
            raise unittest.SkipTest(f"Не удалось инициализировать EmbeddingGenerator: {exc}") from exc

        cls._texts = [
            "Кошки любят спать на солнце.",
            "Многие коты часто дремлют под теплыми лучами солнца.",
            "База данных Qdrant хранит векторы и payload.",
        ]
        cls._embeddings = cls._embedder.generate(sentences=cls._texts)
        cls._vector_size = int(cls._embeddings.shape[1])

        cls._collection_name = f"atomic_statements_test_{uuid.uuid4().hex[:10]}"
        os.environ["QDRANT_COLLECTION_NAME"] = cls._collection_name
        os.environ["QDRANT_VECTOR_SIZE"] = str(cls._vector_size)
        cls._get_settings.cache_clear()

        cls._settings = cls._get_settings().qdrant
        cls._client = cls._make_client(cls._settings)

        try:
            cls._client.get_collections()
        except (cls._resp_exc, Exception) as exc:
            raise unittest.SkipTest(
                f"Qdrant недоступен по адресу {cls._settings.url}: {exc}",
            ) from exc

        cls._ensure_collection(cls._client, cls._settings)
        cls._store = cls._store_cls(cls._client, cls._settings)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            if hasattr(cls, "_client") and hasattr(cls, "_collection_name"):
                print(f"[vector-db-live] drop test collection: {cls._collection_name}")
                cls._client.delete_collection(collection_name=cls._collection_name)
        finally:
            if cls._old_collection is None:
                os.environ.pop("QDRANT_COLLECTION_NAME", None)
            else:
                os.environ["QDRANT_COLLECTION_NAME"] = cls._old_collection

            if cls._old_vector_size is None:
                os.environ.pop("QDRANT_VECTOR_SIZE", None)
            else:
                os.environ["QDRANT_VECTOR_SIZE"] = cls._old_vector_size

            cls._get_settings.cache_clear()

    @timed_test
    def test_upsert_and_find_nearest(self) -> None:
        doc_id = f"doc-upsert-{uuid.uuid4().hex[:8]}"
        author = f"author-upsert-{uuid.uuid4().hex[:6]}"
        records = [
            self._Record(
                point_id=str(uuid.uuid4()),
                embedding=self._embeddings[i].tolist(),
                text=self._texts[i],
                document_id=doc_id,
                author=author,
            )
            for i in range(len(self._texts))
        ]
        self._store.upsert_atomic_statements(records)
        print(f"[vector-db-live] inserted points: {len(records)}")

        hits = self._store.search_nearby(
            self._embeddings[0].tolist(),
            limit=3,
            filter_document_id=doc_id,
            filter_author=author,
            use_config_score_threshold=False,
            score_threshold=None,
        )
        print("[vector-db-live] nearest hits:")
        for idx, hit in enumerate(hits, start=1):
            print(f"  {idx}. score={hit.score:.4f}, text={hit.text!r}, author={hit.author}")

        self.assertGreaterEqual(len(hits), 2)
        self.assertEqual(hits[0].text, self._texts[0])
        top2_texts = {hits[0].text, hits[1].text}
        self.assertIn(self._texts[1], top2_texts)

    @timed_test
    def test_filters_and_exclude(self) -> None:
        doc_main = f"doc-filter-main-{uuid.uuid4().hex[:8]}"
        doc_other = f"doc-filter-other-{uuid.uuid4().hex[:8]}"
        rec_main = self._Record(
            point_id=str(uuid.uuid4()),
            embedding=self._embeddings[0].tolist(),
            text=self._texts[0],
            document_id=doc_main,
            author="alice",
        )
        rec_other = self._Record(
            point_id=str(uuid.uuid4()),
            embedding=self._embeddings[1].tolist(),
            text=self._texts[1],
            document_id=doc_other,
            author="bob",
        )
        self._store.upsert_atomic_statements([rec_main, rec_other])
        print("[vector-db-live] inserted 2 points for filter test")

        only_alice = self._store.search_nearby(
            self._embeddings[0].tolist(),
            filter_author="alice",
            limit=10,
            use_config_score_threshold=False,
            score_threshold=None,
        )
        print(f"[vector-db-live] only_alice hits={len(only_alice)}")
        self.assertTrue(only_alice)
        self.assertTrue(all(hit.author == "alice" for hit in only_alice))

        exclude_doc = self._store.search_nearby(
            self._embeddings[0].tolist(),
            exclude_document_ids=[doc_main],
            limit=10,
            use_config_score_threshold=False,
            score_threshold=None,
        )
        print(f"[vector-db-live] exclude_doc hits={len(exclude_doc)}")
        self.assertTrue(all(hit.document_id != doc_main for hit in exclude_doc))

        exclude_point = self._store.search_nearby(
            self._embeddings[0].tolist(),
            exclude_point_ids=[rec_main.point_id],
            limit=10,
            use_config_score_threshold=False,
            score_threshold=None,
        )
        print(f"[vector-db-live] exclude_point hits={len(exclude_point)}")
        self.assertTrue(all(hit.point_id != rec_main.point_id for hit in exclude_point))


if __name__ == "__main__":
    unittest.main()

