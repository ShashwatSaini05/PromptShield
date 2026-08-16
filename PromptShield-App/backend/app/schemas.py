"""
schemas.py
----------
Pydantic models for request/response validation on every API endpoint.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.config import settings


# -- Auth --


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# -- Prediction --


class PredictRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=settings.MAX_PROMPT_LENGTH)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt must not be blank or whitespace-only.")
        return v


class PredictResponse(BaseModel):
    label: str
    confidence: float


class BatchPredictRequest(BaseModel):
    prompts: List[str] = Field(..., min_length=1, max_length=50)

    @field_validator("prompts")
    @classmethod
    def validate_each_prompt(cls, v: List[str]) -> List[str]:
        max_len = settings.MAX_PROMPT_LENGTH
        for i, p in enumerate(v):
            if not p or not p.strip():
                raise ValueError(f"Prompt at index {i} must not be empty.")
            if len(p) > max_len:
                raise ValueError(
                    f"Prompt at index {i} exceeds max length of {max_len}."
                )
        return v


class BatchPredictItem(BaseModel):
    prompt: str
    label: str
    confidence: float


class BatchPredictResponse(BaseModel):
    results: List[BatchPredictItem]


# -- History --


class HistoryItem(BaseModel):
    id: int
    prompt_text: str
    predicted_label: str
    confidence: float
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedHistory(BaseModel):
    items: List[HistoryItem]
    total: int
    page: int
    page_size: int
    pages: int


# -- Health --


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[dict] = None
