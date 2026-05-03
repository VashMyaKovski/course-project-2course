from abc import ABC, abstractmethod


class FactExtractorBase(ABC):
    """
    Контракт извлекателя фактов: текстовый фрагмент → список атомарных утверждений.

    Каждый элемент списка — самодостаточное утверждение, которое понятно без окружающего контекста.
    """

    @abstractmethod
    def extract(self, chunk: str) -> list[str]:
        ...
