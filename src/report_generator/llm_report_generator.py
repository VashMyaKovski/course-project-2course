from abc import ABC, abstractmethod
from typing import Optional

import httpx

from config import get_settings


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Follows Dependency Inversion Principle.
    """

    @abstractmethod
    def generate_report(self, prompt: str) -> str:
        """
        Generate a report using the LLM.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            str: The generated report text.
        """
        pass


class OpenRouterLLMProvider(LLMProvider):
    """
    Implementation of LLM provider using OpenRouter API.
    Handles communication with Google Gemma or other models via OpenRouter.
    """

    DEFAULT_MODEL = "openrouter/owl-alpha"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize OpenRouter LLM provider.

        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
            model: Model identifier.
            temperature: Sampling temperature (0-1).
            max_tokens: Maximum tokens in response.

        Raises:
            ValueError: If API key is not provided and not in environment.
        """
        settings = get_settings().report_generator
        self.api_key = settings.openrouter_api_key

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_report(self, prompt: str) -> str:
        """
        Generate report using OpenRouter API.

        Args:
            prompt: The prompt to send.

        Returns:
            str: Generated report from LLM.

        Raises:
            RuntimeError: If API call fails.
        """
        try:
            response = self._call_api(prompt)
            return self._extract_content(response)
        except Exception as e:
            raise RuntimeError(f"Failed to generate report from LLM: {str(e)}")

    def _call_api(self, prompt: str) -> dict:
        """Make API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Contradiction Detector",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        with httpx.Client() as client:
            api_response = client.post(
                self.OPENROUTER_API_URL,
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            api_response.raise_for_status()

        return api_response.json()

    def _extract_content(self, response: dict) -> str:
        """Extract text content from API response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected API response format: {str(e)}")


class ReportGenerator:
    """
    Generates contradiction analysis reports using LLM.
    Coordinates prompt building and LLM invocation.
    Single Responsibility: Report generation orchestration.
    """

    def __init__(self, llm_provider: LLMProvider, prompt_builder):
        """
        Initialize report generator.

        Args:
            llm_provider: LLM provider instance (dependency injection).
            prompt_builder: Prompt builder instance.
        """
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder

    def generate(self, contradictions: list) -> str:
        """
        Generate a report from contradiction analysis results.

        Args:
            contradictions: List of contradiction detection results.

        Returns:
            str: Generated report text.
        """
        prompt = self.prompt_builder.build_prompt(contradictions)
        report = self.llm_provider.generate_report(prompt)
        return report

    def generate_with_metadata(self, contradictions: list) -> dict:
        """
        Generate report with metadata about generation.

        Args:
            contradictions: List of contradiction detection results.

        Returns:
            dict: Report with metadata.
        """
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()
        report_text = self.generate(contradictions)

        return {
            "timestamp": timestamp,
            "model": getattr(self.llm_provider, "model", "unknown"),
            "num_contradictions": len(contradictions),
            "report": report_text,
        }

if __name__ == "__main__":
    report_generator = OpenRouterLLMProvider()

    response = report_generator.generate_report(
        prompt="Сгенерируй что-нибудь.",
    )

    print(response)

