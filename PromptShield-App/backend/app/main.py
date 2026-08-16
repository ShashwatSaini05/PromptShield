"""
main.py
-------
FastAPI application entry-point.

- Registers all route modules
- Sets up CORS, rate limiting, and request logging
- Creates database tables on startup via lifespan
- Exposes /health endpoint
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import Base, engine
from app.middleware import limiter, request_logging_middleware
from app.routes import auth_routes, predict_routes, history_routes
from app.ml_service import is_model_loaded, get_model_info
from app.schemas import HealthResponse

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
)


# -- Lifespan (replaces deprecated on_event) --
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create all tables on startup.

    We use create_all() rather than Alembic for this stage because the schema
    is small (2 tables) and stable. Alembic would be added when schema
    evolution becomes frequent.
    """
    Base.metadata.create_all(bind=engine)
    logging.getLogger(__name__).info(
        "Database tables created / verified. Model loaded: %s", is_model_loaded()
    )
    yield


# -- App --
app = FastAPI(
    title="PromptShield API",
    description="Prompt-injection detection service wrapping the PromptShield-ML classifier.",
    version="1.0.0",
    lifespan=lifespan,
)

# -- CORS --
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Rate limiting --
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# -- Request logging --
app.middleware("http")(request_logging_middleware)

# -- Routes --
app.include_router(auth_routes.router)
app.include_router(predict_routes.router)
app.include_router(history_routes.router)


# -- Health --
@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        model_version=get_model_info(),
    )
