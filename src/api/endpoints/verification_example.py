"""
Example API endpoint demonstrating usage of the verification module.
This shows how to integrate the verification service with a FastAPI application.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from src.verification.verification_service import VerificationService
from src.verification.vector_store import VectorStoreInterface
from src.api.schemas.verification import VerificationRequest, VerificationResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# Example dependency for getting a vector store instance
# In a real application, this would be properly initialized (e.g., with FAISS, Qdrant, etc.)
def get_example_vector_store() -> VectorStoreInterface:
    """
    Example vector store dependency.
    This is a placeholder that would be replaced with a real implementation.
    """
    # This is just for demonstration - in reality you'd initialize a real vector store
    # For example:
    # from src.verification.faiss_store import FAISSVectorStore
    # return FAISSVectorStore(index_path="./data/vector_index")
    raise NotImplementedError("This is an example - replace with real vector store implementation")


def get_verification_service(
    vector_store: VectorStoreInterface = Depends(get_example_vector_store)
) -> VerificationService:
    """
    Dependency to get the verification service instance.
    """
    return VerificationService(vector_store=vector_store)


@router.post("/verify-example", response_model=Dict[str, Any])
async def verify_claim_example(
    request: VerificationRequest,
    verification_service: VerificationService = Depends(get_verification_service)
) -> Dict[str, Any]:
    """
    Example endpoint for verifying a claim.
    Demonstrates how to use the verification service in an API endpoint.
    """
    try:
        # Convert Pydantic model to dict for metadata filters
        metadata_filters = None
        if request.metadata_filters:
            metadata_filters = request.metadata_filters.dict(exclude_unset=True)

        # Run verification
        result, report = await verification_service.verify_claim_and_generate_report(
            claim=request.claim,
            search_limit=request.search_limit,
            metadata_filters=metadata_filters
        )

        # Return the report (already in the correct format for the response)
        return report

    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during verification")


# Alternative simpler endpoint that just returns the result
@router.post("/verify-simple", response_model=Dict[str, Any])
async def verify_claim_simple(
    request: VerificationRequest,
    verification_service: VerificationService = Depends(get_verification_service)
) -> Dict[str, Any]:
    """
    Simple verification endpoint that returns just the verification result.
    """
    try:
        # Convert Pydantic model to dict for metadata filters
        metadata_filters = None
        if request.metadata_filters:
            metadata_filters = request.metadata_filters.dict(exclude_unset=True)

        # Run verification (without generating report)
        result = await verification_service.verify_claim(
            claim=request.claim,
            search_limit=request.search_limit,
            metadata_filters=metadata_filters
        )

        # Convert result to dictionary format
        return {
            "claim": result.claim,
            "verification": {
                "overall_label": result.overall_label,
                "confidence_score": result.confidence_score
            },
            "evidence_counts": {
                "supporting": len(result.supporting_evidence),
                "contradicting": len(result.contradicting_evidence),
                "neutral": len(result.neutral_evidence)
            },
            "metadata": result.metadata
        }

    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during verification")