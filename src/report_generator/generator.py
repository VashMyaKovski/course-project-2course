import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_report_generator import OpenRouterLLMProvider
from .llm_report_generator import ReportGenerator as LLMReportGenerator
from .report_prompt import ContradictionAnalysisPromptBuilder


class ReportGenerator:
    """
    Main report generator that orchestrates the entire report generation pipeline.
    Coordinates contradiction detection results -> analysis -> file output.
    Single Responsibility: Coordinates report generation and persistence.
    """

    def __init__(
        self,
        output_dir: str = "data",
        llm_provider: Optional[OpenRouterLLMProvider] = None,
    ):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save generated reports.
            llm_provider: LLM provider instance. If None, creates default OpenRouter provider.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.llm_provider = llm_provider or OpenRouterLLMProvider()
        self.prompt_builder = ContradictionAnalysisPromptBuilder()
        self.llm_generator = LLMReportGenerator(
            self.llm_provider,
            self.prompt_builder,
        )

    def generate(
        self,
        contradictions: List[Dict[str, Any]],
        report_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a report from contradiction detection results and save to file.

        Args:
            contradictions: List of contradiction detection results.
            report_name: Optional name for the report file (without extension).

        Returns:
            dict: Result metadata including file path and report info.
        """
        if not contradictions:
            raise ValueError("contradictions list cannot be empty")

        # Generate report text
        report_data = self.llm_generator.generate_with_metadata(contradictions)

        # Save to file
        file_path = self._save_report(report_data, report_name)

        return {
            "status": "success",
            "file_path": str(file_path),
            "timestamp": report_data["timestamp"],
            "model": report_data["model"],
            "num_contradictions_analyzed": report_data["num_contradictions"],
        }

    def generate_batch(
        self,
        contradictions_list: List[List[Dict[str, Any]]],
        batch_name: str = "batch",
    ) -> List[Dict[str, Any]]:
        """
        Generate reports for multiple contradiction sets.

        Args:
            contradictions_list: List of contradiction lists.
            batch_name: Base name for batch reports.

        Returns:
            list: Results for each generated report.
        """
        results = []
        for idx, contradictions in enumerate(contradictions_list, 1):
            report_name = f"{batch_name}_{idx:03d}"
            result = self.generate(contradictions, report_name)
            results.append(result)

        return results

    def _save_report(
        self,
        report_data: Dict[str, Any],
        report_name: Optional[str] = None,
    ) -> Path:
        """
        Save report to file in JSON format.

        Args:
            report_data: Report data with metadata.
            report_name: Optional custom report name.

        Returns:
            Path: Path to saved report file.
        """
        if report_name is None:
            report_name = self._generate_default_name()

        file_path = self.output_dir / f"{report_name}.json"

        # Ensure unique file name
        counter = 1
        original_path = file_path
        while file_path.exists():
            stem = original_path.stem
            file_path = self.output_dir / f"{stem}_{counter:03d}.json"
            counter += 1

        # Write report to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        return file_path

    def _generate_default_name(self) -> str:
        """Generate default report name with timestamp."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"report_{timestamp}"

if __name__ == "__main__":
    report_generator = ReportGenerator()

    response = report_generator.generate(
        contradictions=[]
    )

    print(response)
