from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from .vector_store import VectorRecord
from .nli_classifier import NLIClassifier

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verification for a single claim."""
    claim: str
    supporting_evidence: List[VectorRecord]
    contradicting_evidence: List[VectorRecord]
    neutral_evidence: List[VectorRecord]
    overall_label: str
    confidence_score: float
    metadata: Dict[str, Any]


class ResultAggregator:
    """Aggregates verification results from NLI classification."""

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the result aggregator.
        
        Args:
            confidence_threshold: Minimum confidence score for considering a prediction
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

    def aggregate_results(
        self,
        claim: str,
        evidence_records: List[VectorRecord],
        nli_predictions: List[Tuple[str, float]]
    ) -> VerificationResult:
        """
        Aggregate NLI predictions into a verification result.
        
        Args:
            claim: The claim being verified
            evidence_records: List of evidence VectorRecord objects
            nli_predictions: List of (label, confidence) tuples from NLI classifier
            
        Returns:
            VerificationResult object
        """
        if len(evidence_records) != len(nli_predictions):
            raise ValueError("Number of evidence records must match number of predictions")

        supporting_evidence = []
        contradicting_evidence = []
        neutral_evidence = []
        
        # Weighted voting for overall label
        label_weights = {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTION": 0.0}
        total_weight = 0.0
        
        for record, (label, confidence) in zip(evidence_records, nli_predictions):
            # Only consider predictions above confidence threshold
            if confidence < self.confidence_threshold:
                continue
                
            weight = confidence
            label_weights[label] += weight
            total_weight += weight
            
            # Categorize evidence
            if label == "ENTAILMENT":
                supporting_evidence.append(record)
            elif label == "CONTRADICTION":
                contradicting_evidence.append(record)
            else:  # NEUTRAL
                neutral_evidence.append(record)
        
        # Determine overall label based on weighted voting
        if total_weight > 0:
            overall_label = max(label_weights, key=label_weights.get)
            # Normalize weights to get confidence score
            confidence_score = label_weights[overall_label] / total_weight if total_weight > 0 else 0.0
        else:
            overall_label = "UNCERTAIN"
            confidence_score = 0.0
        
        result = VerificationResult(
            claim=claim,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            neutral_evidence=neutral_evidence,
            overall_label=overall_label,
            confidence_score=confidence_score,
            metadata={
                "total_evidence": len(evidence_records),
                "filtered_evidence": len([r for r, (_, c) in zip(evidence_records, nli_predictions) 
                                        if c >= self.confidence_threshold]),
                "confidence_threshold": self.confidence_threshold
            }
        )
        
        logger.info(f"Aggregated verification result: {overall_label} (confidence: {confidence_score:.4f})")
        return result

    def update_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold.
        
        Args:
            threshold: New confidence threshold value (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to {threshold}")