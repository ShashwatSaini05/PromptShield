"""
ml_service.py
-------------
Singleton loader for the PromptShield-ML classifier.

Loads model, vectorizer, and model_info.json ONCE at import time so every
request reuses the same objects (no disk I/O per request).
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Optional

import joblib

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_vectorizer = None
_model_info: Optional[dict] = None
_loaded = False


def _load():
    """Load ML artifacts from disk. Called once at module init."""
    global _model, _vectorizer, _model_info, _loaded

    model_dir = Path(settings.MODEL_DIR)
    model_path = model_dir / "prompt_injection_model.pkl"
    vec_path = model_dir / "tfidf_vectorizer.pkl"
    info_path = model_dir / "model_info.json"

    try:
        _model = joblib.load(str(model_path))
        _vectorizer = joblib.load(str(vec_path))
        logger.info("ML model loaded from %s", model_path)
    except FileNotFoundError as exc:
        logger.error("Could not load ML artifacts: %s", exc)
        return

    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            _model_info = json.load(f)

    _loaded = True


# Load on import (singleton pattern)
_load()


def is_model_loaded() -> bool:
    return _loaded


def get_model_info() -> Optional[dict]:
    return _model_info


def predict_prompt(text: str) -> Tuple[str, float]:
    """
    Classify a single prompt.

    Returns
    -------
    (label, confidence) where label is 'SAFE' or 'PROMPT_INJECTION'
    and confidence is a float in [0, 1].
    """
    if not _loaded:
        raise RuntimeError("ML model is not loaded.")

    x = _vectorizer.transform([text])
    pred = _model.predict(x)[0]
    label = "PROMPT_INJECTION" if pred == 1 else "SAFE"

    if hasattr(_model, "predict_proba"):
        confidence = float(_model.predict_proba(x)[0][pred])
    else:
        score = _model.decision_function(x)[0]
        confidence = float(1 / (1 + abs(score)))

    return label, confidence
