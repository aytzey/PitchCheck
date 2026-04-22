import os
os.environ["TRIBE_ALLOW_MOCK"] = "1"
os.environ.pop("OPENROUTER_API_KEY", None)

import pytest
from fastapi.testclient import TestClient
from tribe_service.app import app

client = TestClient(app)


class TestHealth:
    def test_returns_200(self):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert data["service"] == "pitchscore-tribe"

    def test_has_model_info(self):
        res = client.get("/health")
        data = res.json()
        assert "model_id" in data
        assert "device" in data
        assert "runtime" in data
        assert "pipeline" in data
        assert data["pipeline"]["idle_unload_seconds"] >= 0
        assert data["pipeline"]["active_scores"] >= 0
        assert "configured_oom_fallback_text_devices" in data["runtime"]
        assert "last_score" in data["runtime"]
        assert "openrouter_enabled" in data


class TestScore:
    def test_valid_request(self):
        res = client.post("/score", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "CTO at a mid-stage startup, technical background",
        })
        assert res.status_code == 200
        data = res.json()
        report = data["report"]
        assert 0 <= report["persuasion_score"] <= 100
        assert isinstance(report["verdict"], str)
        assert isinstance(report["narrative"], str)
        assert len(report["breakdown"]) == 5
        assert len(report["neural_signals"]) == 6
        assert isinstance(report["strengths"], list)
        assert isinstance(report["risks"], list)
        assert isinstance(report["rewrite_suggestions"], list)
        assert isinstance(report["persuasion_evidence"], dict)
        assert isinstance(report["robustness"], dict)
        assert 0 <= report["robustness"]["confidence"] <= 1
        assert isinstance(report["robustness"]["neuro_axes"], dict)
        assert "self_value" in report["robustness"]["neuro_axes"]
        assert isinstance(report["robustness"]["scientific_caveats"], list)
        assert report["platform"] == "general"

    def test_with_platform(self):
        res = client.post("/score", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "VP of Engineering, enterprise",
            "platform": "email",
        })
        assert res.status_code == 200
        assert res.json()["report"]["platform"] == "email"

    def test_short_message_returns_422(self):
        res = client.post("/score", json={
            "message": "Hi",
            "persona": "CTO at startup",
        })
        assert res.status_code == 422

    def test_missing_persona_returns_422(self):
        res = client.post("/score", json={
            "message": "Our platform reduces deployment time by 80%",
        })
        assert res.status_code == 422

    def test_missing_message_returns_422(self):
        res = client.post("/score", json={
            "persona": "CTO at startup",
        })
        assert res.status_code == 422

    def test_cors_headers(self):
        res = client.options("/score", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        })
        assert "access-control-allow-origin" in res.headers

    def test_breakdown_sections_have_correct_keys(self):
        res = client.post("/score", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "CTO at startup, technical",
        })
        report = res.json()["report"]
        breakdown_keys = {b["key"] for b in report["breakdown"]}
        expected = {"emotional_resonance", "clarity", "urgency", "credibility", "personalization_fit"}
        assert breakdown_keys == expected

    def test_neural_signals_have_correct_keys(self):
        res = client.post("/score", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "CTO at startup, technical",
        })
        report = res.json()["report"]
        signal_keys = {s["key"] for s in report["neural_signals"]}
        expected = {"emotional_engagement", "personal_relevance", "social_proof_potential", "memorability", "attention_capture", "cognitive_friction"}
        assert signal_keys == expected

    def test_runtime_unload_releases_loaded_pipeline(self):
        res = client.post("/score", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "CTO at startup, technical",
        })
        assert res.status_code == 200
        assert client.get("/health").json()["pipeline"]["model_loaded"] is True

        unload = client.post("/runtime/unload")
        assert unload.status_code == 200
        assert unload.json()["ok"] is True
        assert client.get("/health").json()["pipeline"]["model_loaded"] is False


class TestPitchServerAuth:
    def _enable_auth(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv("PITCHSERVER_AUTH_REQUIRED", "1")
        monkeypatch.setenv("PITCHSERVER_AUTH_FILE", str(tmp_path / "auth.json"))
        monkeypatch.setenv("PITCHSERVER_AUTH_SEED_USERNAME", "pitchserver")
        monkeypatch.setenv("PITCHSERVER_AUTH_SEED_PASSWORD", "initial-pass-123")

    def test_score_requires_login_when_auth_enabled(self, monkeypatch, tmp_path):
        self._enable_auth(monkeypatch, tmp_path)
        res = client.post("/score", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "CTO at a mid-stage startup, technical background",
        })
        assert res.status_code == 401

    def test_login_allows_scoring_when_auth_enabled(self, monkeypatch, tmp_path):
        self._enable_auth(monkeypatch, tmp_path)
        login = client.post("/auth/login", json={
            "username": "pitchserver",
            "password": "initial-pass-123",
        })
        assert login.status_code == 200
        token = login.json()["token"]

        res = client.post(
            "/score",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "message": "Our platform reduces deployment time by 80% for enterprise teams",
                "persona": "CTO at a mid-stage startup, technical background",
            },
        )
        assert res.status_code == 200

    def test_logged_in_user_can_change_credentials(self, monkeypatch, tmp_path):
        self._enable_auth(monkeypatch, tmp_path)
        login = client.post("/auth/login", json={
            "username": "pitchserver",
            "password": "initial-pass-123",
        })
        token = login.json()["token"]

        changed = client.post(
            "/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "initial-pass-123",
                "new_username": "newpitch",
                "new_password": "new-pass-456",
            },
        )
        assert changed.status_code == 200
        assert changed.json()["username"] == "newpitch"

        old_login = client.post("/auth/login", json={
            "username": "pitchserver",
            "password": "initial-pass-123",
        })
        assert old_login.status_code == 401

        new_login = client.post("/auth/login", json={
            "username": "newpitch",
            "password": "new-pass-456",
        })
        assert new_login.status_code == 200

    def test_change_credentials_accepts_desktop_camel_case_payload(self, monkeypatch, tmp_path):
        self._enable_auth(monkeypatch, tmp_path)
        login = client.post("/auth/login", json={
            "username": "pitchserver",
            "password": "initial-pass-123",
        })
        token = login.json()["token"]

        changed = client.post(
            "/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "currentPassword": "initial-pass-123",
                "newUsername": "desktopuser",
                "newPassword": "desktop-pass-456",
            },
        )

        assert changed.status_code == 200
        assert changed.json()["username"] == "desktopuser"
