"""
predict_routes.py
-----------------
POST /predict        - classify a single prompt
POST /predict/batch  - classify multiple prompts in one call

Both endpoints log every prediction to the database (with user_id=NULL
for anonymous requests). Auth requirement is configurable via
REQUIRE_AUTH_FOR_PREDICT.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth import get_current_user, get_optional_user
from app.config import settings
from app.database import get_db
from app.ml_service import predict_prompt
from app.models import PredictionLog, User
from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictItem,
    BatchPredictResponse,
)

router = APIRouter(prefix="/predict", tags=["predict"])


def _get_user_dependency():
    """Return the correct auth dependency based on config."""
    if settings.REQUIRE_AUTH_FOR_PREDICT:
        return get_current_user
    return get_optional_user


def _log_prediction(
    db: Session,
    user: Optional[User],
    prompt_text: str,
    label: str,
    confidence: float,
):
    log = PredictionLog(
        user_id=user.id if user else None,
        prompt_text=prompt_text,
        predicted_label=label,
        confidence=confidence,
    )
    db.add(log)
    db.commit()


@router.post("", response_model=PredictResponse)
def predict_single(
    body: PredictRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(_get_user_dependency()),
):
    label, confidence = predict_prompt(body.prompt)
    _log_prediction(db, user, body.prompt, label, confidence)
    return PredictResponse(label=label, confidence=confidence)


@router.post("/batch", response_model=BatchPredictResponse)
def predict_batch(
    body: BatchPredictRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(_get_user_dependency()),
):
    results = []
    for prompt_text in body.prompts:
        label, confidence = predict_prompt(prompt_text)
        _log_prediction(db, user, prompt_text, label, confidence)
        results.append(
            BatchPredictItem(prompt=prompt_text, label=label, confidence=confidence)
        )
    return BatchPredictResponse(results=results)
