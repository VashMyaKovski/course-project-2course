from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from src.verification.verification_service import VerificationService
from src.verification.vector_store import VectorStoreInterface
from src.api.schemas.verification import VerificationRequest, VerificationResponse
from src.config import config  # Assuming there is a global config object

logger = logging.getLogger(__name__)

router = APIRouter()


def get_vector_store() -> VectorStoreInterface:
    """
    Dependency to get the vector store instance.
    In a real application, this would be initialized and managed properly (e.g., using a dependency injection container).
    For now, we'll return a placeholder. The actual implementation should be provided by the application.
    """
    # This is a placeholder. In practice, you would initialize and return a real vector store (e.g., FAISS, Qdrant, etc.)
    # For example: return FAISSVectorStore(...)
    raise NotImplementedError("Vector store dependency not implemented")


def get_verification_service(
    vector_store: VectorStoreInterface = Depends(get_vector_store)
) -> VerificationService:
    """
    Dependency to get the verification service instance.
    """
    return VerificationService(vector_store=vector_store)


@router.post("/verify", response_model=VerificationResponse)
async def verify_claim(
    request: VerificationRequest,
    verification_service: VerificationService = Depends(get_verification_service)
) -> Dict[str, Any]:
    """
    Verify a claim and return the verification result.
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

        # The report is already in the format we want for the response
        return report

    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during verification")