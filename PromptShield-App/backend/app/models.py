"""
models.py
---------
SQLAlchemy ORM models.

Tables
------
- users            - registered accounts (email + bcrypt hash, no plaintext).
- prediction_logs  - every prediction request, auth'd or anonymous.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(320), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # One-to-many: a user can have many prediction logs
    predictions = relationship("PredictionLog", back_populates="user")

    def __repr__(self):
        return f"<User id={self.id} email={self.email!r}>"


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    prompt_text = Column(Text, nullable=False)
    predicted_label = Column(String(20), nullable=False)  # SAFE | PROMPT_INJECTION
    confidence = Column(Float, nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    user = relationship("User", back_populates="predictions")

    def __repr__(self):
        return (
            f"<PredictionLog id={self.id} label={self.predicted_label} "
            f"conf={self.confidence:.2f}>"
        )
