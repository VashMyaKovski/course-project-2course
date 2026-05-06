from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from config.settings import QdrantSettings


def make_qdrant_client(settings: QdrantSettings) -> QdrantClient:
    kwargs: dict = {"url": settings.url, "timeout": settings.timeout_sec}
    if settings.api_key is not None:
        kwargs["api_key"] = settings.api_key
    return QdrantClient(**kwargs)


def parse_distance(name: str) -> Distance:
    key = name.upper().strip()
    mapping: dict[str, Distance] = {
        "COSINE": Distance.COSINE,
        "DOT": Distance.DOT,
        "EUCLID": Distance.EUCLID,
        "EUCLIDEAN": Distance.EUCLID,
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported QDRANT_DISTANCE={name!r}; use one of {sorted(mapping)}",
        )
    return mapping[key]


def _payload_index_fields(settings: QdrantSettings) -> tuple[str, str]:
    return settings.payload_document_id_key, settings.payload_author_key


def ensure_atomic_statements_collection(client: QdrantClient, settings: QdrantSettings) -> None:
    """
    Создаёт коллекцию под атомарные утверждения, если её ещё нет,
    и keyword-индексы по полям document_id и author в payload.
    """
    name = settings.collection_name
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.vector_size,
                distance=parse_distance(settings.distance),
            ),
        )

    doc_key, author_key = _payload_index_fields(settings)
    for field_name in (doc_key, author_key):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as exc:
            err = (exc.content or b"").decode(errors="replace").lower()
            if exc.status_code == 409 or "already exists" in err:
                continue
            raise
