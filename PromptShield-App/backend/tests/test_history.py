"""
test_history.py
---------------
Tests for /history endpoint.

Covers:
- Authenticated user sees only their own predictions
- Cross-user isolation (User A cannot see User B's history)
- Unauthenticated request returns 401
- Pagination works correctly
"""

from tests.conftest import create_user, auth_header


class TestHistoryIsolation:
    def test_user_sees_own_history(self, client):
        _, token = create_user(client, "alice@example.com", "password123")

        # Make some predictions as Alice
        client.post(
            "/predict",
            json={"prompt": "Hello, how are you?"},
            headers=auth_header(token),
        )
        client.post(
            "/predict",
            json={"prompt": "What is 2 + 2?"},
            headers=auth_header(token),
        )

        resp = client.get("/history", headers=auth_header(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    def test_cross_user_isolation(self, client):
        """User A must NOT see User B's predictions."""
        _, token_a = create_user(client, "alice@test.com", "password123")
        _, token_b = create_user(client, "bob@test.com", "password456")

        # Alice makes 3 predictions
        for prompt in ["Alice prompt 1", "Alice prompt 2", "Alice prompt 3"]:
            client.post(
                "/predict",
                json={"prompt": prompt},
                headers=auth_header(token_a),
            )

        # Bob makes 1 prediction
        client.post(
            "/predict",
            json={"prompt": "Bob prompt 1"},
            headers=auth_header(token_b),
        )

        # Alice's history should contain exactly 3 items
        resp_a = client.get("/history", headers=auth_header(token_a))
        assert resp_a.status_code == 200
        assert resp_a.json()["total"] == 3

        # Bob's history should contain exactly 1 item
        resp_b = client.get("/history", headers=auth_header(token_b))
        assert resp_b.status_code == 200
        assert resp_b.json()["total"] == 1

        # Verify Bob cannot see Alice's prompts
        bob_prompts = [item["prompt_text"] for item in resp_b.json()["items"]]
        assert all("Alice" not in p for p in bob_prompts)

    def test_unauthenticated_history_rejected(self, client):
        resp = client.get("/history")
        assert resp.status_code in (401, 403)


class TestHistoryPagination:
    def test_pagination(self, client):
        _, token = create_user(client, "pager@example.com", "password123")

        # Create 5 predictions
        for i in range(5):
            client.post(
                "/predict",
                json={"prompt": f"Pagination test prompt {i}"},
                headers=auth_header(token),
            )

        # Request page 1 with page_size=2
        resp = client.get(
            "/history?page=1&page_size=2", headers=auth_header(token)
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert data["pages"] == 3
        assert len(data["items"]) == 2
