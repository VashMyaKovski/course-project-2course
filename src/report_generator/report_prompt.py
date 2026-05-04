from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ReportPromptBuilder(ABC):
    """
    Abstract base class for building prompts for LLM.
    Follows Interface Segregation Principle.
    """

    @abstractmethod
    def build_prompt(self, contradictions: List[Dict[str, Any]]) -> str:
        """
        Build a prompt from contradiction detection results.

        Args:
            contradictions: List of contradiction detection results from ContradictionDetector.
                Each item has 'base_sentence' and 'nli_results' keys.

        Returns:
            str: Formatted prompt for LLM.
        """
        pass


class ContradictionAnalysisPromptBuilder(ReportPromptBuilder):
    """
    Concrete prompt builder for contradiction analysis.
    Formats contradiction detection results for LLM analysis.
    """

    def __init__(self, model_name: str = "openrouter/owl-alpha"):
        """
        Initialize the prompt builder.

        Args:
            model_name: Name of the model (for context in prompts).
        """
        self.model_name = model_name
        self.system_instruction = self._get_system_instruction()

    def build_prompt(self, contradictions: List[Dict[str, Any]]) -> str:
        """
        Build a comprehensive prompt for analyzing contradictions.

        Args:
            contradictions: List of contradiction analysis results.

        Returns:
            str: Complete prompt with instructions and data.
        """
        if not contradictions:
            return self._empty_analysis_prompt()

        formatted_data = self._format_contradictions(contradictions)
        return self._build_complete_prompt(formatted_data)

    def _get_system_instruction(self) -> str:
        """Get the system instruction for the LLM."""
        return (
            "You are an expert fact-checker and text analysis assistant. "
            "Your task is to analyze contradiction detection results and provide a comprehensive report. "
            "Focus on: "
            "1. Summary of contradictions found "
            "2. Analysis of entailment relationships "
            "3. Overall consistency assessment "
            "4. Recommendations for text revision "
            "Provide clear, structured analysis suitable for document generation."
        )

    def _format_contradictions(self, contradictions: List[Dict[str, Any]]) -> str:
        """Format contradiction data into readable text format."""
        formatted_items = []

        for idx, item in enumerate(contradictions, 1):
            base_sentence = item.get("base_sentence", "N/A")
            nli_results = item.get("nli_results", {})

            formatted_items.append(f"Analysis {idx}:")
            formatted_items.append(f"Base Sentence: {base_sentence}")
            formatted_items.append("Relationships Found:")

            for ref_sentence, relation in nli_results.items():
                relation_label = self._get_relation_label(relation)
                formatted_items.append(f"  - {ref_sentence}")
                formatted_items.append(f"    Relation: {relation_label}")

            formatted_items.append("")

        return "\n".join(formatted_items)

    def _get_relation_label(self, relation: str) -> str:
        """Get human-readable label for relation type."""
        labels = {
            "contradiction": "CONTRADICTION (conflicting information)",
            "entailment": "ENTAILMENT (logically follows)",
            "neutral": "NEUTRAL (independent facts)",
        }
        return labels.get(relation, relation)

    def _build_complete_prompt(self, formatted_data: str) -> str:
        """Build the complete prompt with system instruction and data."""
        return (
            f"{self.system_instruction}\n\n"
            f"Analyze the following contradiction detection results:\n\n"
            f"{formatted_data}\n"
            f"Based on this analysis, provide a detailed report including:\n"
            f"1. Summary of findings\n"
            f"2. Identified contradictions and their implications\n"
            f"3. Consistency score (0-100%)\n"
            f"4. Recommended corrections\n"
            f"5. Overall quality assessment"
        )

    def _empty_analysis_prompt(self) -> str:
        """Build prompt for empty contradiction list."""
        return (
            f"{self.system_instruction}\n\n"
            f"No contradictions were detected in the analyzed text. "
            f"Please provide a brief summary indicating the text appears to be internally consistent."
        )
