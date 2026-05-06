from dataclasses import dataclass


@dataclass(frozen=True)
class AtomicStatementRecord:
    """Одно атомарное утверждение для upsert в Qdrant."""

    # Qdrant принимает неотрицательный int или UUID-строку; см. нормализацию в store.
    point_id: str | int
    embedding: list[float]
    text: str
    document_id: str
    author: str


@dataclass(frozen=True)
class NeighborHit:
    """Результат поиска похожих утверждений."""

    point_id: str
    score: float
    text: str
    document_id: str
    author: str
