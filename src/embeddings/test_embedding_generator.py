from __future__ import annotations

import numpy as np
import pytest

from config.settings import EmbeddingModelConfig

from src.embeddings.generator import EmbeddingGenerator, IEmbeddingGenerator

E5_BASE_DIM = 768
E5_DEFAULT_MODEL = "intfloat/multilingual-e5-base"
E5_DEFAULT_PREFIX = "passage: "


@pytest.fixture(scope="module")
def cpu_config() -> EmbeddingModelConfig:
    """Explicit CPU so runs are stable across machines with/without CUDA."""
    return EmbeddingModelConfig(model_name=E5_DEFAULT_MODEL, device="cpu")


@pytest.fixture(scope="module")
def generator(cpu_config: EmbeddingModelConfig) -> EmbeddingGenerator:
    return EmbeddingGenerator(config=cpu_config)


class TestEmbeddingModelConfig:
    def test_explicit_device_is_preserved(self) -> None:
        cfg = EmbeddingModelConfig(model_name=E5_DEFAULT_MODEL, device="cpu")
        assert cfg.device == "cpu"

    def test_default_prefix_and_model(self) -> None:
        cfg = EmbeddingModelConfig(device="cpu")
        assert cfg.model_name == E5_DEFAULT_MODEL
        assert cfg.sentence_prefix == E5_DEFAULT_PREFIX


class TestEmbeddingGeneratorInitialization:
    def test_implements_protocol(self, generator: EmbeddingGenerator) -> None:
        assert isinstance(generator, IEmbeddingGenerator)

    def test_init_default_config(self, generator: EmbeddingGenerator) -> None:
        assert generator._config.model_name == E5_DEFAULT_MODEL
        assert generator._config.sentence_prefix == E5_DEFAULT_PREFIX
        assert generator._config.device == "cpu"

    def test_init_custom_prefix(self, cpu_config: EmbeddingModelConfig) -> None:
        cfg = EmbeddingModelConfig(
            model_name=E5_DEFAULT_MODEL,
            sentence_prefix="doc: ",
            device=cpu_config.device,
        )
        gen = EmbeddingGenerator(config=cfg)
        assert gen._config.sentence_prefix == "doc: "
        out = gen.generate(sentences=["hello"])
        assert out.shape == (1, E5_BASE_DIM)


class TestEmbeddingGeneration:
    def test_generate_requires_keyword_sentences(self, generator: EmbeddingGenerator) -> None:
        with pytest.raises(TypeError):
            generator.generate(["only positional"])  # type: ignore[call-arg]

    def test_generate_single_sentence(self, generator: EmbeddingGenerator) -> None:
        sentences = ["The cat sits on the mat."]
        embeddings = generator.generate(sentences=sentences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, E5_BASE_DIM)
        assert embeddings.dtype == np.float32
        assert np.allclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-5)

    def test_generate_multiple_sentences(self, generator: EmbeddingGenerator) -> None:
        sentences = [
            "The cat sits on the mat.",
            "Dogs are loyal animals.",
            "Birds can fly in the sky.",
        ]
        embeddings = generator.generate(sentences=sentences)
        assert embeddings.shape == (len(sentences), E5_BASE_DIM)
        assert embeddings.dtype == np.float32

    def test_embedding_l2_normalization(self, generator: EmbeddingGenerator) -> None:
        sentences = [
            "This is a test sentence.",
            "Another test example.",
            "Final verification sentence.",
        ]
        embeddings = generator.generate(sentences=sentences)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_identical_inputs_equal_rows(self, generator: EmbeddingGenerator) -> None:
        text = "Repeatable embedding check."
        a = generator.generate(sentences=[text, text])
        assert np.allclose(a[0], a[1], atol=1e-6)

    def test_semantic_similarity_ordering(self, generator: EmbeddingGenerator) -> None:
        similar = [
            "The cat is sleeping on the couch.",
            "A cat is resting on the sofa.",
        ]
        different = [
            "The cat is sleeping on the couch.",
            "I like to eat pizza for lunch.",
        ]
        sim_emb = generator.generate(sentences=similar)
        diff_emb = generator.generate(sentences=different)

        sim_score = float(np.dot(sim_emb[0], sim_emb[1]))
        diff_score = float(np.dot(diff_emb[0], diff_emb[1]))
        assert sim_score > diff_score


class TestEmbeddingStatistics:
    def test_embedding_shape_and_finite(self, generator: EmbeddingGenerator) -> None:
        sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
        ]
        embeddings = generator.generate(sentences=sentences)
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == E5_BASE_DIM
        assert np.all(np.isfinite(embeddings))


class TestEmbeddingIntegration:
    def test_similarity_ranking(self, generator: EmbeddingGenerator) -> None:
        documents = [
            "Artificial intelligence is transforming technology.",
            "Machine learning enables computers to learn from data.",
            "Deep learning uses neural networks for pattern recognition.",
            "Natural language processing helps understand human language.",
            "Computer vision enables machines to see and interpret images.",
        ]
        embeddings = generator.generate(sentences=documents)
        assert embeddings.shape == (len(documents), E5_BASE_DIM)

        query = "Deep learning and neural networks"
        query_embedding = generator.generate(sentences=[query])
        assert query_embedding.shape == (1, E5_BASE_DIM)

        similarities = np.dot(embeddings, query_embedding.T).flatten()
        assert similarities[2] >= similarities[0]
        assert similarities[2] >= similarities[4]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
