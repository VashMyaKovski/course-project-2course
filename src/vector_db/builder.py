import uuid
from collections.abc import Sequence
from typing import Any

from loguru import logger

from config import get_settings

from src.vector_db.collection import ensure_atomic_statements_collection, make_qdrant_client
from src.vector_db.models import AtomicStatementRecord, NeighborHit
from src.vector_db.store import AtomicStatementsStore


class VectorDBBuilder:
    """
    Шаг пайплайна: записать пары (эмбеддинг, текст утверждения) в Qdrant.

    Один вызов ``build`` задаёт общий synthetic ``document_id`` для всего батча;
    автор в payload — ``__pipeline__`` (пока пайплайн не передаёт реальные метаданные документа).
    """

    def __init__(self) -> None:
        self._settings = get_settings().qdrant
        self._client = make_qdrant_client(self._settings)
        self._store = AtomicStatementsStore(self._client, self._settings)
        ensure_atomic_statements_collection(self._client, self._settings)

    def build(self, embeddings: Any, triplets: Any) -> list[AtomicStatementRecord]:
        if not triplets:
            return []
        facts = list(triplets)

        if embeddings is None:
            logger.warning("vector_db: эмбеддинги отсутствуют — индексация в Qdrant пропущена.")
            return []

        vectors = list(embeddings)
        if len(vectors) != len(facts):
            raise ValueError(
                f"vector_db: len(embeddings)={len(vectors)} != len(facts)={len(facts)}",
            )
        if not vectors:
            logger.warning("vector_db: пустой список эмбеддингов — индексация пропущена.")
            return []

        pairs = [(v, str(t).strip()) for v, t in zip(vectors, facts) if str(t).strip()]
        if not pairs:
            return []

        doc_id = str(uuid.uuid4())
        author = "__pipeline__"
        records = [
            AtomicStatementRecord(
                point_id=str(uuid.uuid4()),
                embedding=list(vec),
                text=text,
                document_id=doc_id,
                author=author,
            )
            for vec, text in pairs
        ]
        self._store.upsert_atomic_statements(records)
        return records

    def search_neighbors(
        self,
        query_embedding: Sequence[float],
        *,
        exclude_point_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[NeighborHit]:
        return self._store.search_nearby(
            query_embedding,
            exclude_point_ids=exclude_point_ids,
            limit=limit,
        )
