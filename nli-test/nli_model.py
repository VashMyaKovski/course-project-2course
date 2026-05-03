import json
import os
from typing import Dict, Union

import torch
from sentence_transformers import CrossEncoder

DEFAULT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


class ContradictionDetector:
    """
    Детектор противоречий на базе Cross-Encoder.
    Обрабатывает premise и hypothesis совместно для максимальной точности.
    """

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = os.getenv("NLI_MODEL_NAME", DEFAULT_MODEL_NAME)

        print(f"🔄 Загрузка Cross-Encoder модели: {model_name}...")

        # Автоматический выбор устройства (CUDA/CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

        # Получение маппинга лейблов из конфигурации модели
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        # Динамический поиск индекса класса 'contradiction'
        if "contradiction" in self.label2id:
            self.contradiction_idx = self.label2id["contradiction"]
        else:
            raise ValueError(
                f"Модель не содержит лейбла 'contradiction'. Доступные: {list(self.id2label.values())}"
            )

        print(f"✅ Модель загружена на {device}.")
        print(f"   Классы: {self.id2label}")
        print(f"   Индекс 'contradiction': {self.contradiction_idx}")

    def detect(
        self, premise: str, hypothesis: str
    ) -> Dict[str, Union[float, bool, str]]:
        """
        Анализ пары предложений.

        Returns:
            dict:
                - contradiction_probability: float (0..1)
                - is_contradiction: bool
                - predicted_label: str
                - all_probabilities: dict
        """
        scores = self.model.predict(
            [(premise, hypothesis)], apply_softmax=True, convert_to_numpy=True
        )[0]

        contradiction_prob = float(scores[self.contradiction_idx])

        # Определение предсказанного класса (для отладки/статистики)
        predicted_class_idx = scores.argmax()
        predicted_label = self.id2label[predicted_class_idx]

        return {
            "contradiction_probability": round(contradiction_prob, 4),
            "is_contradiction": bool(contradiction_prob >= 0.5),
            "predicted_label": predicted_label,
            "all_probabilities": {
                label: round(float(prob), 4)
                for label, prob in zip(self.id2label.values(), scores)
            },
        }

    def to_json(self, premise: str, hypothesis: str) -> str:
        """Сериализация результата в JSON."""
        return json.dumps(self.detect(premise, hypothesis), ensure_ascii=False)
