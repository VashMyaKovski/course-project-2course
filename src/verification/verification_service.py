import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from .config import config
from .vector_store import VectorStoreInterface, VectorRecord
from .metadata_filter import MetadataFilter
from .nli_classifier import NLIClassifier
from .result_aggregator import ResultAggregator, VerificationResult
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class VerificationService:
    """Main verification service that orchestrates the verification process."""

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        nli_model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ):
        """
        Initialize the verification service.
        
        Args:
            vector_store: Vector store implementation for searching evidence
            nli_model_name: Name or path of NLI model (uses config if None)
            confidence_threshold: Confidence threshold for aggregation (uses config if None)
        """
        self.vector_store = vector_store
        self.metadata_filter = MetadataFilter()
        
        # Load configuration
        nli_model = nli_model_name or config.get('nli.model_name', 'microsoft/deberta-v3-large')
        threshold = confidence_threshold or config.get('verification.confidence_threshold', 0.7)
        
        self.nli_classifier = NLIClassifier(model_name_or_path=nli_model)
        self.result_aggregator = ResultAggregator(confidence_threshold=threshold)
        self.report_generator = ReportGenerator()
        
        logger.info("VerificationService initialized")

    async def verify_claim(
        self,
        claim: str,
        search_limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify a claim by searching for evidence and running NLI classification.
        
        Args:
            claim: The claim to verify
            search_limit: Maximum number of evidence records to retrieve
            metadata_filters: Optional filters to apply to search results
            
        Returns:
            VerificationResult object
        """
        logger.info(f"Verifying claim: {claim[:50]}...")
        
        # Step 1: Generate embedding for the claim (placeholder - in practice use real embedding model)
        embedding_dim = config.get('vector_store.embedding_dimension', 768)
        claim_embedding = self._embed_claim(claim, embedding_dim)
        
        # Step 2: Search for similar vectors in the vector store with metadata filters
        search_results = await self.vector_store.search(
            query_embedding=claim_embedding,
            limit=search_limit,
            filter_conditions=metadata_filters
        )
        
        if not search_results:
            logger.warning("No evidence found for claim")
            return VerificationResult(
                claim=claim,
                supporting_evidence=[],
                contradicting_evidence=[],
                neutral_evidence=[],
                overall_label="UNCERTAIN",
                confidence_score=0.0,
                metadata={"total_evidence": 0}
            )
        
        # Step 3: Prepare premise-hypothesis pairs for NLI
        # Premise: evidence text, Hypothesis: claim
        premises = [record.text for record in search_results]
        hypotheses = [claim] * len(search_results)
        
        # Step 4: Run NLI classification
        nli_predictions = self.nli_classifier.batch_predict(premises, hypotheses)
        
        # Step 5: Aggregate results
        verification_result = self.result_aggregator.aggregate_results(
            claim=claim,
            evidence_records=search_results,
            nli_predictions=nli_predictions
        )
        
        return verification_result

    async def verify_claim_and_generate_report(
        self,
        claim: str,
        search_limit: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None,
        report_path: Optional[str] = None
    ) -> Tuple[VerificationResult, Dict[str, Any]]:
        """
        Verify a claim and generate a report.
        
        Args:
            claim: The claim to verify
            search_limit: Maximum number of evidence records to retrieve
            metadata_filters: Optional filters to apply to search results
            report_path: Optional path to save the JSON report
            
        Returns:
            Tuple of (VerificationResult, report_dict)
        """
        result = await self.verify_claim(claim, search_limit, metadata_filters)
        report = self.report_generator.generate_report(result)
        
        if report_path:
            self.report_generator.save_report(report, report_path)
            
        return result, report

    def _embed_claim(self, claim: str, dimension: int) -> List[float]:
        """
        Placeholder for claim embedding. In a real system, this would use an embedding model.
        
        Args:
            claim: The claim text
            dimension: Dimension of the embedding vector
            
        Returns:
            Dummy embedding vector of specified dimension
        """
        logger.warning("Using dummy embedding for claim. Replace with real embedding model.")
        # Return a dummy vector (e.g., all zeros) for demonstration
        return [0.0] * dimension

    def update_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold for the result aggregator.
        
        Args:
            threshold: New confidence threshold value (0.0-1.0)
        """
        self.result_aggregator.update_confidence_threshold(threshold)