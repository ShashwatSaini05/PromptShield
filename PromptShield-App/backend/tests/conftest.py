"""
conftest.py
-----------
Shared pytest fixtures: in-memory SQLite database, FastAPI TestClient,
and helper functions for creating test users.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ---- bootstrap: override settings BEFORE any app module is imported ----
os.environ["DATABASE_URL"] = "sqlite://"  # in-memory
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["RATE_LIMIT_PER_MINUTE"] = "60"
os.environ["REQUIRE_AUTH_FOR_PREDICT"] = "false"

from app.database import Base, get_db
from app.main import app

# In-memory test database
_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def reset_tables():
    """Drop and recreate all tables between tests for isolation."""
    Base.metadata.drop_all(bind=_engine)
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture()
def client():
    """FastAPI TestClient."""
    # Reset rate limiter state between tests
    from app.middleware import limiter
    limiter.reset()
    return TestClient(app)


@pytest.fixture()
def db_session():
    """Raw DB session for direct queries in tests."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_user(client_: TestClient, email: str = "test@example.com", password: str = "password123"):
    """Helper: sign up a user and return (user_data, token)."""
    resp = client_.post("/auth/signup", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    user_data = resp.json()

    resp = client_.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]

    return user_data, token


def auth_header(token: str) -> dict:
    """Build an Authorization header dict."""
    return {"Authorization": f"Bearer {token}"}
