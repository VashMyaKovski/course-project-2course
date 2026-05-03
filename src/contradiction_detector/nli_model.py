import os
from typing import Dict, List, Union

from sentence_transformers import CrossEncoder
import torch


class NLIModel:
    """
    NLI модель для определения отношений между предложениями.
    Использует Cross-Encoder для классификации: contradiction, entailment, neutral.
    """

    DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    def __init__(self, model_name: str = None):
        """
        Инициализация NLI модели.

        Args:
            model_name: Название модели. Если None, берется из переменной окружения NLI_MODEL_NAME.
        """
        if model_name is None:
            model_name = os.getenv("NLI_MODEL_NAME", self.DEFAULT_MODEL_NAME)

        print(f"🔄 Загрузка NLI модели: {model_name}...")

        # Автоматический выбор устройства
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(model_name, device=device)

        # Маппинг лейблов
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        print(f"✅ Модель загружена на {device}.")
        print(f"   Доступные классы: {list(self.id2label.values())}")

    def predict_relation(self, premise: str, hypothesis: str) -> str:
        """
        Определяет отношение между premise и hypothesis.

        Args:
            premise: Основное предложение.
            hypothesis: Предложение для сравнения.

        Returns:
            str: 'contradiction', 'entailment' или 'neutral'.
        """
        scores = self.model.predict(
            [(premise, hypothesis)],
            apply_softmax=True,
            convert_to_numpy=True
        )[0]

        # Находим класс с максимальной вероятностью
        predicted_class_idx = scores.argmax()
        return self.id2label[predicted_class_idx]

    def predict_relations_batch(self, premise: str, hypotheses: List[str]) -> Dict[str, str]:
        """
        Определяет отношения между premise и списком hypotheses.

        Args:
            premise: Основное предложение.
            hypotheses: Список предложений для сравнения.

        Returns:
            Dict[str, str]: Словарь hypothesis -> relation.
        """
        if not hypotheses:
            return {}

        pairs = [(premise, hyp) for hyp in hypotheses]
        scores = self.model.predict(pairs, apply_softmax=True, convert_to_numpy=True)

        results = {}
        for hyp, score in zip(hypotheses, scores):
            predicted_class_idx = score.argmax()
            results[hyp] = self.id2label[predicted_class_idx]

        return results