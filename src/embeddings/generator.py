from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import EmbeddingModelConfig


class IEmbeddingGenerator(ABC):
    """
    Contract for embedding generator.
    """

    @abstractmethod
    def generate(self, *, sentences: list[str]) -> np.ndarray:
        """
        Convert a list of strings to a matrix of embeddings of shape (N, D).
        """
        raise NotImplementedError


class EmbeddingGenerator(IEmbeddingGenerator):
    """
    Embedding generator based on a SentenceTransformer model (default: multilingual-e5-base).

    Prepends ``sentence_prefix`` to each string (E5 expects e.g. ``passage: ``), then encodes
    with L2-normalized outputs when the underlying model supports it.
    """

    def __init__(self, config: EmbeddingModelConfig | None = None) -> None:
        self._config = config or EmbeddingModelConfig()
        self._model = SentenceTransformer(self._config.model_name, device=self._config.device)

        logger.info(
            "EmbeddingGenerator initialized: model='{}', device='{}'",
            self._config.model_name,
            self._config.device,
        )

    def _add_sentence_prefix(self, *, sentences: list[str]) -> list[str]:
        return [f"{self._config.sentence_prefix}{sentence}" for sentence in sentences]

    def generate(self, *, sentences: list[str]) -> np.ndarray:
        """
        Convert a list of strings to a matrix of embeddings of shape (N, D), dtype float32.
        """
        logger.info(f"Generating embeddings for {len(sentences)} sentences")

        vectors = self._model.encode(
            self._add_sentence_prefix(sentences=sentences),
            normalize_embeddings=True,
        )

        return np.asarray(vectors, dtype=np.float32)
