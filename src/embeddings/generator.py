from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    torch = None
    AutoModel = None
    AutoTokenizer = None


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
    def generate(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert a list of strings to a matrix of embeddings of shape (N, D).
        """
        pass


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
        config: Optional[EmbeddingModelConfig] = None,
        tokenizer: Optional["AutoTokenizer"] = None,
        model: Optional["AutoModel"] = None,
        device: Optional["torch.device"] = None,
    ) -> None:
        self._config = config or EmbeddingModelConfig()
        self._tokenizer, self._model = self._initialize_components(tokenizer, model)
        self._device = device or self._resolve_device()
        self._model.to(self._device)
        self._model.eval()

        hidden_size = int(getattr(self._model.config, "hidden_size", 0))
        if hidden_size != self._config.expected_dimension:
            raise ValueError(
                f"Model hidden size is {hidden_size}, expected {self._config.expected_dimension}."
            )

        logger.info(
            "EmbeddingGenerator initialized: model='{}', device='{}', dim={}",
            self._config.model_name,
            self._device,
            self._config.expected_dimension,
        )

    def generate(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates L2-normalized embeddings of shape (N, 768).

        Args:
            sentences: List of atomic statements (facts).
            batch_size: Batch size.

        Returns:
            np.ndarray: Matrix of embeddings of shape (N, 768), dtype float32.
        """
        self._validate_sentences(sentences)
        effective_batch_size = self._validate_batch_size(batch_size)
        preprocessed_sentences = self._preprocess_sentences(sentences)

        logger.info(
            "Generating embeddings for {} sentences with batch_size={}",
            len(preprocessed_sentences),
            effective_batch_size,
        )

        try:
            batches: List[np.ndarray] = []
            for start in range(0, len(preprocessed_sentences), effective_batch_size):
                batch = preprocessed_sentences[start : start + effective_batch_size]
                batch_embeddings = self._encode_batch(batch)
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
        tokenizer: Optional["AutoTokenizer"],
        model: Optional["AutoModel"],
    ) -> Tuple["AutoTokenizer", "AutoModel"]:
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

    def _resolve_device(self) -> "torch.device":
        if torch is None:
            raise ImportError(
                "Required dependency 'torch' is missing. Install it before use."
            )
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _validate_sentences(self, sentences: Sequence[str]) -> None:
        if not isinstance(sentences, list):
            raise TypeError("Input must be List[str].")
        if not sentences:
            raise ValueError("Input list cannot be empty.")
        for idx, sentence in enumerate(sentences):
            if not isinstance(sentence, str):
                raise TypeError(f"Sentence at index {idx} is not a string.")
            if not sentence.strip():
                raise ValueError(f"Sentence at index {idx} is empty or whitespace.")

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        return batch_size

    def _preprocess_sentences(self, sentences: Sequence[str]) -> List[str]:
        return [self._config.passage_prefix + sentence.strip() for sentence in sentences]

    def _encode_batch(self, batch: Sequence[str]) -> np.ndarray:
        if torch is None:
            raise ImportError(
                "Required dependency 'torch' is missing. Install it before use."
            )

        encoded = self._tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            token_embeddings = outputs.last_hidden_state
            sentence_embeddings = self._mean_pooling(
                token_embeddings, encoded["attention_mask"]
            )
            normalized = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return normalized.detach().cpu().numpy()

    @staticmethod
    def _mean_pooling(
        token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        pooled = torch.sum(token_embeddings * mask_expanded, dim=1)
        normalizer = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return pooled / normalizer
