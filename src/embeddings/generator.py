from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from src.utils.device import resolve_device

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """
    Embedding generator model configuration.
    """

    model_name: str = "intfloat/multilingual-e5-base"
    expected_dimension: int = 768
    default_batch_size: int = 32
    max_length: int = 512
    passage_prefix: str = "passage: "


class IEmbeddingGenerator(ABC):
    """
    Contract for embedding generator.
    """

    @abstractmethod
    def generate(self, *, sentences: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert a list of strings to a matrix of embeddings of shape (N, D).
        """
        raise NotImplementedError


class EmbeddingGenerator(IEmbeddingGenerator):
    """
    Embedding generator based on intfloat/multilingual-e5-base.

    Implements:
    - adding a mandatory prefix "passage: " for E5;
    - batch processing;
    - calculations on GPU if available, otherwise CPU;
    - L2-normalization of output vectors.
    """

    def __init__(
        self,
        config: EmbeddingModelConfig | None = None,
        tokenizer: AutoTokenizer | None = None,
        model: AutoModel | None = None,
        device: str | None = None,
    ) -> None:
        self._config = config or EmbeddingModelConfig()
        self._tokenizer, self._model = self._initialize_components(tokenizer, model)
        self._device = resolve_device(device)
        self._model.to(self._device)
        self._model.eval()

        hidden_size = int(getattr(self._model.config, "hidden_size", 0))
        if hidden_size != self._config.expected_dimension:
            raise ValueError(
                f"Model hidden size is {hidden_size}, expected {self._config.expected_dimension}."
            )

        logger.info(
            f"EmbeddingGenerator initialized: model='{self._config.model_name}', "
            f"device='{self._device}', dim={self._config.expected_dimension}"
        )

    def generate(self, *, sentences: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates L2-normalized embeddings of shape (N, 768).

        Args:
            sentences: List of atomic statements (facts).
            batch_size: Batch size.

        Returns:
            np.ndarray: Matrix of embeddings of shape (N, 768), dtype float32.
        """
        effective_batch_size = self._validate_batch_size(batch_size=batch_size)
        preprocessed_sentences = self._validate_sentences(sentences=sentences)

        logger.info(
            f"Generating embeddings for {len(preprocessed_sentences)} sentences "
            f"with batch_size={effective_batch_size}"
        )

        try:
            batches: list[np.ndarray] = []
            for start in range(0, len(preprocessed_sentences), effective_batch_size):
                batch = preprocessed_sentences[start : start + effective_batch_size]
                batch_embeddings = self._encode_batch(batch=batch)
                batches.append(batch_embeddings)

            embeddings = np.vstack(batches).astype(np.float32, copy=False)
            if embeddings.shape[1] != self._config.expected_dimension:
                raise RuntimeError(
                    "Invalid embedding dimension: "
                    f"{embeddings.shape[1]} != {self._config.expected_dimension}"
                )
            return embeddings
        except Exception as exc:
            logger.exception("Embedding generation failed: {}", exc)
            raise RuntimeError("Failed to generate embeddings.") from exc

    def _initialize_components(
        self,
        tokenizer: AutoTokenizer | None,
        model: AutoModel | None,
    ) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        if tokenizer is not None and model is not None:
            return tokenizer, model

        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "Required dependencies are missing. "
                "Install 'transformers' and 'torch' packages."
            )

        logger.info("Loading embedding model '{}'", self._config.model_name)
        loaded_tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self._config.model_name
        )
        loaded_model = model or AutoModel.from_pretrained(self._config.model_name)
        return loaded_tokenizer, loaded_model

    def _validate_sentences(self, *, sentences: list[str]) -> list[str]:
        if not sentences:
            raise ValueError("Input list cannot be empty.")

        normalized_sentences: list[str] = []
        for idx, sentence in enumerate(sentences):
            if not isinstance(sentence, str):
                raise TypeError(f"Sentence at index {idx} is not a string.")
            cleaned = sentence.strip()
            if not cleaned:
                raise ValueError(f"Sentence at index {idx} is empty or whitespace.")
            normalized_sentences.append(f"{self._config.passage_prefix}{cleaned}")
        return normalized_sentences

    @staticmethod
    def _validate_batch_size(*, batch_size: int) -> int:
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        return batch_size

    def _encode_batch(self, *, batch: list[str]) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Required dependency 'torch' is missing. Install it before use."
            ) from exc

        encoded = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = self._model(**encoded)
            pooled_embeddings = self._mean_pooling(
                token_embeddings=outputs.last_hidden_state.detach().cpu().numpy(),
                attention_mask=encoded["attention_mask"].detach().cpu().numpy(),
            )
            return self._normalize_embeddings(pooled_embeddings)

    @staticmethod
    def _mean_pooling(
        *, token_embeddings: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        mask_expanded = attention_mask[..., None].astype(np.float32, copy=False)
        pooled = np.sum(token_embeddings * mask_expanded, axis=1)
        normalizer = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
        return pooled / normalizer

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, 1e-12, None)
        return normalized.astype(np.float32, copy=False)
