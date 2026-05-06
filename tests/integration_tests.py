from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config.settings import EmbeddingModelConfig
from src.chunking.paragraph_chunker import ParagraphChunker
from src.embeddings.generator import EmbeddingGenerator
from src.fact_extractor import Llama31InstructChatCompletionFactExtractor
from src.ingestion.pipeline import IngestionPipeline


@pytest.mark.integration
def test_ingestion_plus_chunking(tmp_path: Path) -> None:
    file_path = tmp_path / "integration_doc.txt"
    file_path.write_text(
        "Ешьте конфету паркур2.\n\n"
        "Она очень вкусная.\n\n"
        "И имеет отношение к спорту, физической и зерновой культуре.",
        encoding="utf-8",
    )

    document = IngestionPipeline().run(str(file_path))
    text, metadata = next(iter(document.items()))

    chunked = ParagraphChunker(min_chunk_size=5).chunk(
        {"data": text, "metadata": metadata["uuid"]}
    )

    assert metadata["file_name"] == "integration_doc.txt"
    assert len(chunked["chunks"]) == 3
    assert chunked["chunks"][0] == "Ешьте конфету паркур2."
    assert chunked["chunks"][1] == "Она очень вкусная."
    assert chunked["chunks"][2] == "И имеет отношение к спорту, физической и зерновой культуре."


@pytest.mark.integration
@pytest.mark.slow
def test_fact_extraction_plus_embeddings_generation() -> None:
    extractor = Llama31InstructChatCompletionFactExtractor(timeout_sec=240.0)
    facts = extractor.extract(
        "Георгий Колхозов - нормальный парень. Он из города Красногорска, а не из города Ярославля."
    )

    generator = EmbeddingGenerator(config=EmbeddingModelConfig(device="cpu"))
    vectors = generator.generate(sentences=facts)

    assert isinstance(facts, list)
    assert len(facts) >= 2
    assert all(isinstance(fact, str) and fact.strip() for fact in facts)

    assert vectors.shape[0] == len(facts)
    assert vectors.shape[1] > 0
    assert vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


@pytest.mark.integration
@pytest.mark.slow
def test_chunking_plus_atomic_fact_extraction() -> None:
    text = (
        "У Антона в наличии есть ноутбук. Сева ест ноутбук Антона.\n\n"
        "Максим ест чебуреки из закусочной ЗакутОК. Максим отравился и теперь ему очень болит живот!"
    )

    chunks = ParagraphChunker().chunk_text(text)
    assert len(chunks) == 2

    extractor = Llama31InstructChatCompletionFactExtractor(timeout_sec=240.0)
    facts_per_chunk = [extractor.extract(chunk) for chunk in chunks]
    facts = [fact for chunk_facts in facts_per_chunk for fact in chunk_facts]

    assert all(isinstance(chunk_facts, list) for chunk_facts in facts_per_chunk)
    assert all(len(chunk_facts) > 0 for chunk_facts in facts_per_chunk)
    assert len(facts) >= len(chunks)
    assert all(isinstance(fact, str) and fact.strip() for fact in facts)
