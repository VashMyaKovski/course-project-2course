import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchAny,
    MatchValue,
    PointStruct,
)

from config.settings import QdrantSettings

from src.vector_db.models import AtomicStatementRecord, NeighborHit


def _normalize_point_id(point_id: str | int) -> str | int:
    if isinstance(point_id, int):
        if point_id < 0:
            raise ValueError(f"point_id must be non-negative int, got {point_id}")
        return point_id
    try:
        uuid.UUID(point_id)
        return point_id
    except ValueError:
        pass
    if point_id.isdigit():
        return int(point_id)
    raise ValueError(
        f"point_id must be UUID string or non-negative int/str int, got {point_id!r}",
    )


class AtomicStatementsStore:
    """Загрузка атомарных утверждений и поиск ближайших с фильтрами по payload."""

    def __init__(self, client: QdrantClient, settings: QdrantSettings) -> None:
        self._client = client
        self._settings = settings

    def upsert_atomic_statements(self, records: Sequence[AtomicStatementRecord]) -> None:
        if not records:
            return

        size = self._settings.vector_size
        batch = self._settings.upsert_batch_size
        name = self._settings.collection_name
        tk = self._settings.payload_text_key
        dk = self._settings.payload_document_id_key
        ak = self._settings.payload_author_key

        chunk: list[PointStruct] = []
        for rec in records:
            if len(rec.embedding) != size:
                raise ValueError(
                    f"embedding dim {len(rec.embedding)} != QDRANT_VECTOR_SIZE={size} "
                    f"(point_id={rec.point_id!r})",
                )
            chunk.append(
                PointStruct(
                    id=_normalize_point_id(rec.point_id),
                    vector=rec.embedding,
                    payload={
                        tk: rec.text,
                        dk: rec.document_id,
                        ak: rec.author,
                    },
                ),
            )
            if len(chunk) >= batch:
                self._client.upsert(collection_name=name, points=chunk)
                chunk = []
        if chunk:
            self._client.upsert(collection_name=name, points=chunk)

    def search_nearby(
        self,
        query_vector: Sequence[float],
        *,
        filter_document_id: str | None = None,
        filter_author: str | None = None,
        exclude_document_ids: Sequence[str] | None = None,
        exclude_point_ids: Sequence[str] | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
        use_config_score_threshold: bool = True,
    ) -> list[NeighborHit]:
        """
        Ищет до ``limit`` ближайших точек (по умолчанию из конфига).

        Порог по score Qdrant: при ``use_config_score_threshold=True`` значение
        ``score_threshold=None`` подставляет порог из настроек (или без порога, если в .env
        пусто). При ``use_config_score_threshold=False`` в поиск уходит ровно
        ``score_threshold`` (в том числе ``None`` — без отсечения по порогу).
        """
        if len(query_vector) != self._settings.vector_size:
            raise ValueError(
                f"query_vector dim {len(query_vector)} != "
                f"QDRANT_VECTOR_SIZE={self._settings.vector_size}",
            )

        eff_limit = self._settings.search_top_k if limit is None else limit
        if use_config_score_threshold:
            eff_threshold = (
                self._settings.score_threshold if score_threshold is None else score_threshold
            )
        else:
            eff_threshold = score_threshold

        must: list[FieldCondition] = []
        if filter_document_id is not None:
            must.append(
                FieldCondition(
                    key=self._settings.payload_document_id_key,
                    match=MatchValue(value=filter_document_id),
                ),
            )
        if filter_author is not None:
            must.append(
                FieldCondition(
                    key=self._settings.payload_author_key,
                    match=MatchValue(value=filter_author),
                ),
            )

        must_not: list[FieldCondition | HasIdCondition] = []
        ex_docs = list(exclude_document_ids or [])
        if ex_docs:
            must_not.append(
                FieldCondition(
                    key=self._settings.payload_document_id_key,
                    match=MatchAny(any=ex_docs),
                ),
            )
        ex_points = list(exclude_point_ids or [])
        if ex_points:
            must_not.append(HasIdCondition(has_id=ex_points))

        query_filter = (
            Filter(must=must, must_not=must_not)
            if (must or must_not)
            else None
        )

        response = self._client.query_points(
            collection_name=self._settings.collection_name,
            query=list(query_vector),
            query_filter=query_filter,
            limit=eff_limit,
            score_threshold=eff_threshold,
            with_payload=True,
        )

        tk = self._settings.payload_text_key
        dk = self._settings.payload_document_id_key
        ak = self._settings.payload_author_key

        out: list[NeighborHit] = []
        for h in response.points:
            payload = h.payload or {}
            out.append(
                NeighborHit(
                    point_id=str(h.id),
                    score=float(h.score or 0.0),
                    text=str(payload.get(tk, "")),
                    document_id=str(payload.get(dk, "")),
                    author=str(payload.get(ak, "")),
                ),
            )
        return out
