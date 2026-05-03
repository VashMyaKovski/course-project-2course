from typing import Dict, List

from .nli_model import NLIModel


class ContradictionDetector:
    """
    Детектор противоречий на основе NLI модели.
    Проверяет основной факт против списка фактов.
    """

    def __init__(self, nli_model: NLIModel = None):
        """
        Инициализация детектора.

        Args:
            nli_model: Экземпляр NLI модели. Если None, создается новый.
        """
        self.nli_model = nli_model or NLIModel()

    def detect_all(self, dict_of_facts: Dict[str, object]) -> Dict[str, Dict[str, str]]:
        """
        Проверяет основной факт против списка фактов.

        Args:
            dict_of_facts: Словарь с ключами:
                - main_fact_what_is_going_to_be_checked: str
                - list_of_facts: List[str]

        Returns:
            Dict[str, Dict[str, str]]: Результаты в формате:
                {
                    base_sentence: str,
                    nli_results: {ref_sentence: str, rel_class: str}
                }
        """

        main_fact = dict_of_facts.get("main_fact_what_is_going_to_be_checked")
        list_of_facts = dict_of_facts.get("list_of_facts", [])

        # Получаем отношения для всех фактов
        nli_results = self.nli_model.predict_relations_batch(main_fact, list_of_facts)

        return {"base_sentence": main_fact, "nli_results": nli_results}
