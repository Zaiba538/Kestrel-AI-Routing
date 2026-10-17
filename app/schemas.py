
"""Request and response schemas for the Kestrel routing API."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RoutingRequest(BaseModel):
    """A single service request submitted for routing."""

    model_config = ConfigDict(extra="forbid")

    request_id: Optional[str] = None
    channel: str = Field(min_length=1, max_length=50)
    product_family: str = Field(min_length=1, max_length=100)
    warranty_status: str = Field(min_length=1, max_length=50)
    request_text: str = Field(min_length=1, max_length=10000)
    source: str = Field(default="crm", min_length=1, max_length=50)


class RoutingResponse(BaseModel):
    """Structured prediction returned to the employee."""

    request_id: Optional[str] = None
    predicted_team: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: List[str]
    review_required: bool
    review_reasons: List[str]
    model_name: str
    model_validation_accuracy: float
    disclaimer: str
