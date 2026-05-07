from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    report_name: Optional[str] = Field(
        default=None,
        description="Optional report name without extension.",
    )


class DetectionResponse(BaseModel):
    status: str
    report: Dict[str, Any]
    contradictions_count: int
    facts_count: int
    chunks_count: int
    metadata: Dict[str, Any]
    contradictions: List[Dict[str, Any]]
