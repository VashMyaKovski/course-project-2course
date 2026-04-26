from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class MetadataFilterBase(BaseModel):
    start_date: Optional[datetime] = Field(None, description="Start date in ISO 8601 format (inclusive)")
    end_date: Optional[datetime] = Field(None, description="End date in ISO 8601 format (inclusive)")
    entities: Optional[List[str]] = Field(None, description="List of entities to filter by")
    sources: Optional[List[str]] = Field(None, description="List of sources to filter by")


class VerificationRequest(BaseModel):
    claim: str = Field(..., description="The claim to verify")
    search_limit: int = Field(10, ge=1, le=100, description="Maximum number of evidence records to retrieve")
    metadata_filters: Optional[MetadataFilterBase] = Field(None, description="Optional metadata filters")


class EvidenceRecord(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]


class VerificationResponse(BaseModel):
    claim: str
    verification: Dict[str, Any]
    evidence: Dict[str, List[EvidenceRecord]]
    metadata: Dict[str, Any]


class VerificationResultDetail(BaseModel):
    overall_label: str
    confidence_score: float