#!/usr/bin/env python3
"""Тест для ContradictionDetector с реальной моделью."""

import os

from src.contradiction_detector.detector import ContradictionDetector


def test_detector():
    """Тестируем детектор противоречий с реальной моделью."""
    print("Тестируем ContradictionDetector с реальной NLI моделью...")

    # Устанавливаем переменную окружения для модели
    os.environ["NLI_MODEL_NAME"] = (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )

    detector = ContradictionDetector()

    dict_of_facts = {
        "main_fact_what_is_going_to_be_checked": "Солнце встает на востоке.",
        "list_of_facts": [
            "Солнце встает на западе.",  # должно быть contradiction
            "Солнце светит днем.",  # должно быть entailment
            "Земля плоская.",  # должно быть neutral
        ],
    }

    result = detector.detect_all(dict_of_facts)

    print("Результат получен!")
    print(f"Base sentence: {result['base_sentence']}")
    print("NLI results:")
    for ref, rel in result["nli_results"].items():
        print(f"  '{ref}' -> {rel}")

    # Проверяем структуру
    assert "base_sentence" in result
    assert "nli_results" in result
    assert isinstance(result["nli_results"], dict)
    assert len(result["nli_results"]) == 3

    # Проверяем, что все отношения корректные
    valid_relations = {"contradiction", "entailment", "neutral"}
    for rel in result["nli_results"].values():
        assert rel in valid_relations, f"Некорректное отношение: {rel}"

    print("Все проверки прошли!")


if __name__ == "__main__":
    test_detector()
