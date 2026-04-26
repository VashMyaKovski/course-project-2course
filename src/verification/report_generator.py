import json
import logging
from typing import Dict, Any, List
from .result_aggregator import VerificationResult
from .vector_store import VectorRecord

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate JSON reports from verification results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_report(self, result: VerificationResult) -> Dict[str, Any]:
        """
        Generate a JSON-serializable report from a VerificationResult.
        
        Args:
            result: VerificationResult object
            
        Returns:
            Dictionary representing the report
        """
        report = {
            "claim": result.claim,
            "verification": {
                "overall_label": result.overall_label,
                "confidence_score": result.confidence_score
            },
            "evidence": {
                "supporting": self._serialize_evidence(result.supporting_evidence),
                "contradicting": self._serialize_evidence(result.contradicting_evidence),
                "neutral": self._serialize_evidence(result.neutral_evidence)
            },
            "metadata": result.metadata
        }
        
        logger.info(f"Generated report for claim: {result.claim[:50]}...")
        return report

    def _serialize_evidence(self, evidence: List[VectorRecord]) -> List[Dict[str, Any]]:
        """
        Serialize a list of VectorRecord objects to a list of dictionaries.
        
        Args:
            evidence: List of VectorRecord objects
            
        Returns:
            List of dictionaries representing the evidence
        """
        serialized = []
        for record in evidence:
            serialized.append({
                "id": record.id,
                "text": record.text,
                "metadata": record.metadata
                # Note: embedding is not included in the report to keep it lightweight
            })
        return serialized

    def save_report(self, report: Dict[str, Any], filepath: str) -> None:
        """
        Save the report to a JSON file.
        
        Args:
            report: Report dictionary
            filepath: Path to save the JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {filepath}")