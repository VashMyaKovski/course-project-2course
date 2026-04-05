from abc import ABC, abstractmethod
from typing import Any, Dict

from loguru import logger


class BasePipeline(ABC):
    """Базовый класс для всех пайплайнов"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state: Dict[str, Any] = {}

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Запуск пайплайна"""
        pass

    def _log_step(self, step_name: str, status: str = "started"):
        logger.info(f"Pipeline step '{step_name}': {status}")
