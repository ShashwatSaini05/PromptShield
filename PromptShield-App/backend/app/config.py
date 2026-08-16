"""
config.py
---------
Application settings loaded from environment variables (or .env file).
Uses Pydantic Settings for validation and type coercion.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Resolve paths relative to this file so they work regardless of cwd
_THIS_DIR = Path(__file__).resolve().parent          # backend/app/
_BACKEND_DIR = _THIS_DIR.parent                      # backend/
_PROJECT_DIR = _BACKEND_DIR.parent                   # PromptShield-App/
_ML_MODEL_DIR = _PROJECT_DIR.parent / "PromptShield" / "model"  # ../PromptShield/model


class Settings(BaseSettings):
    """Central configuration - every value can be overridden via env vars."""

    # -- Database --
    DATABASE_URL: str = f"sqlite:///{_BACKEND_DIR / 'promptshield.db'}"

    # -- JWT / Auth --
    SECRET_KEY: str = "change-me-in-production-use-a-random-256-bit-hex"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # -- ML Model --
    MODEL_DIR: str = str(_ML_MODEL_DIR)

    # -- Prediction behaviour --
    REQUIRE_AUTH_FOR_PREDICT: bool = False
    MAX_PROMPT_LENGTH: int = 4000

    # -- Rate limiting --
    RATE_LIMIT_PER_MINUTE: int = 60

    # -- CORS --
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = {
        "env_file": str(_BACKEND_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
