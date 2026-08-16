"""
test_auth.py
------------
Tests for /auth/signup, /auth/login, and /auth/me endpoints.

Covers:
- Signup + login round-trip
- Wrong password returns generic error
- Duplicate email signup returns 409
- /me requires valid token
"""

from tests.conftest import create_user, auth_header


class TestSignup:
    def test_signup_success(self, client):
        resp = client.post(
            "/auth/signup",
            json={"email": "new@example.com", "password": "securepass123"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "new@example.com"
        assert "password" not in data
        assert "password_hash" not in data
        assert "id" in data

    def test_duplicate_email_rejected(self, client):
        client.post(
            "/auth/signup",
            json={"email": "dup@example.com", "password": "password123"},
        )
        resp = client.post(
            "/auth/signup",
            json={"email": "dup@example.com", "password": "otherpass456"},
        )
        assert resp.status_code == 409

    def test_short_password_rejected(self, client):
        resp = client.post(
            "/auth/signup",
            json={"email": "test@example.com", "password": "short"},
        )
        assert resp.status_code == 422

    def test_invalid_email_rejected(self, client):
        resp = client.post(
            "/auth/signup",
            json={"email": "not-an-email", "password": "password123"},
        )
        assert resp.status_code == 422


class TestLogin:
    def test_login_success(self, client):
        create_user(client, "login@example.com", "password123")
        resp = client.post(
            "/auth/login",
            json={"email": "login@example.com", "password": "password123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_wrong_password(self, client):
        create_user(client, "user@example.com", "correctpassword")
        resp = client.post(
            "/auth/login",
            json={"email": "user@example.com", "password": "wrongpassword"},
        )
        assert resp.status_code == 401
        # Generic message - must NOT reveal whether email exists
        assert "invalid email or password" in resp.json()["detail"].lower()

    def test_nonexistent_email(self, client):
        resp = client.post(
            "/auth/login",
            json={"email": "noone@example.com", "password": "anything123"},
        )
        assert resp.status_code == 401
        assert "invalid email or password" in resp.json()["detail"].lower()


class TestMe:
    def test_me_authenticated(self, client):
        user_data, token = create_user(client, "me@example.com", "password123")
        resp = client.get("/auth/me", headers=auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["email"] == "me@example.com"

    def test_me_no_token(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code in (401, 403)

    def test_me_invalid_token(self, client):
        resp = client.get("/auth/me", headers=auth_header("invalid.token.here"))
        assert resp.status_code == 401
