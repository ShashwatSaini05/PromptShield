"""
history_routes.py
-----------------
GET /history  - paginated list of the current user's own past predictions.

Server-side isolation: the query always filters by user_id = current_user.id,
so users can never see each other's history regardless of frontend behaviour.
"""

import math

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import PredictionLog, User
from app.schemas import PaginatedHistory, HistoryItem

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=PaginatedHistory)
def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Enforce user isolation at the query level
    base_query = db.query(PredictionLog).filter(
        PredictionLog.user_id == current_user.id
    )

    total = base_query.count()
    pages = math.ceil(total / page_size) if total > 0 else 1

    items = (
        base_query.order_by(PredictionLog.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return PaginatedHistory(
        items=[HistoryItem.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )
