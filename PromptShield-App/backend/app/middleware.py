"""
middleware.py
-------------
Rate limiting (slowapi) and request-logging middleware.

Rate limits are applied per-IP on prediction endpoints. Authenticated
users are additionally keyed by user ID, so rate limits are per-identity.
"""

import logging
import time

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

logger = logging.getLogger("promptshield.requests")


# -- Rate limiter (slowapi) --

def _rate_key(request: Request) -> str:
    """Key function: use IP address, append user id if authenticated."""
    ip = get_remote_address(request)
    # If auth header is present and was decoded by the route dependency,
    # the user object may be attached to request.state by the route itself.
    user_id = getattr(getattr(request, "state", None), "user_id", None)
    if user_id:
        return f"{ip}:{user_id}"
    return ip


limiter = Limiter(key_func=_rate_key)

# Build the rate-limit string once
PREDICT_RATE_LIMIT = f"{settings.RATE_LIMIT_PER_MINUTE}/minute"


# -- Request logging middleware --

async def request_logging_middleware(request: Request, call_next):
    """Log method, path, status, and duration. Never log passwords or tokens."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response
