#!/usr/bin/env python3
"""Создаёт коллекцию Qdrant для атомарных утверждений, если её ещё нет."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings

from src.vector_db.collection import ensure_atomic_statements_collection, make_qdrant_client


def main() -> None:
    settings = get_settings().qdrant
    client = make_qdrant_client(settings)
    ensure_atomic_statements_collection(client, settings)
    print(f"Коллекция {settings.collection_name!r} готова (URL {settings.url}).")


if __name__ == "__main__":
    main()
