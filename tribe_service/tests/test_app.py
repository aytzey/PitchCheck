import os
import asyncio
import threading
import time
os.environ["TRIBE_ALLOW_MOCK"] = "1"
os.environ.pop("OPENROUTER_API_KEY", None)

import pytest
from fastapi.testclient import TestClient
import tribe_service.app as service_app
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
        assert 0 < report["robustness"]["prediction_quality_weight"] <= 1
        assert isinstance(report["robustness"]["neuro_axes"], dict)
        assert "self_value" in report["robustness"]["neuro_axes"]
        assert isinstance(report["robustness"]["scientific_caveats"], list)
        assert report["fmri_output"]["prediction_subject_basis"] == "average_subject"
        assert report["fmri_output"]["cortical_mesh"] == "fsaverage5"
        assert "calibration_quality" in report["persuasion_evidence"]
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

    def test_queue_timeout_is_reported_separately(self, monkeypatch):
        async def run_case():
            score_lock = asyncio.Semaphore(1)
            await score_lock.acquire()
            monkeypatch.setattr(service_app, "_score_lock", score_lock)
            monkeypatch.setattr(service_app, "TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS", 0.01)

            with pytest.raises(service_app.ScoreQueueTimeoutError):
                await service_app._score_text_with_backpressure("valid pitch message")

            score_lock.release()

        asyncio.run(run_case())

    def test_run_timeout_keeps_pipeline_active_until_worker_finishes(self, monkeypatch):
        finished = threading.Event()

        def slow_score_text(_: str):
            time.sleep(0.05)
            finished.set()
            return [[1.0]]

        monkeypatch.setattr(service_app, "_active_scores", 0)
        monkeypatch.setattr(service_app, "score_text", slow_score_text)
        monkeypatch.setattr(service_app, "TRIBE_SCORE_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(service_app, "TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS", 0.05)

        async def run_case():
            monkeypatch.setattr(service_app, "_score_lock", asyncio.Semaphore(1))

            with pytest.raises(service_app.ScoreRunTimeoutError):
                await service_app._score_text_with_backpressure("valid pitch message")

            assert (await service_app._pipeline_status())["active_scores"] == 1
            assert finished.wait(1.0)
            for _ in range(20):
                if (await service_app._pipeline_status())["active_scores"] == 0:
                    break
                await asyncio.sleep(0.01)
            assert (await service_app._pipeline_status())["active_scores"] == 0

        asyncio.run(run_case())

    def test_refine_uses_llm_without_tribe_rescore(self, monkeypatch):
        def fail_score_text(_: str):
            raise AssertionError("/refine should not call TRIBE scoring")

        def fake_refine_pitch_message(**kwargs):
            assert kwargs["suggestions"] == ["Make the CTA easier"]
            return {
                "refined_message": "Improved message with a lower-friction CTA.",
                "model": "test-refiner",
                "methodology": "llm_semantic_refine_no_tribe_rescore",
            }

        monkeypatch.setattr(service_app, "score_text", fail_score_text)
        monkeypatch.setattr(service_app, "refine_pitch_message", fake_refine_pitch_message)

        res = client.post("/refine", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "CTO at a mid-stage startup, technical background",
            "platform": "email",
            "suggestions": ["Make the CTA easier"],
        })

        assert res.status_code == 200
        data = res.json()
        assert data["refined_message"] == "Improved message with a lower-friction CTA."
        assert data["model"] == "test-refiner"
        assert data["methodology"] == "llm_semantic_refine_no_tribe_rescore"

    def test_refine_can_return_clarifying_questions(self, monkeypatch):
        def fake_refine_pitch_message(**kwargs):
            return {
                "refined_message": None,
                "model": "test-refiner",
                "needs_clarification": True,
                "questions": [{
                    "id": "proof",
                    "label": "Proof",
                    "question": "Which verified proof can we mention?",
                    "why": "Avoids invented claims.",
                }],
                "safety_notes": ["No unverified claims added"],
                "persuasion_profile": {"proof_threshold": "high"},
                "methodology": "llm_semantic_refine_with_optional_clarifying_questions",
            }

        monkeypatch.setattr(service_app, "refine_pitch_message", fake_refine_pitch_message)

        res = client.post("/refine", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "CTO at a mid-stage startup, technical background",
            "suggestions": ["Add proof"],
        })

        assert res.status_code == 200
        data = res.json()
        assert data["refined_message"] is None
        assert data["needs_clarification"] is True
        assert data["questions"][0]["id"] == "proof"
        assert data["safety_notes"] == ["No unverified claims added"]

    def test_refine_reports_missing_openrouter_key(self, monkeypatch):
        def fake_refine_pitch_message(**kwargs):
            raise RuntimeError("OpenRouter API key is missing; LLM refine is unavailable.")

        monkeypatch.setattr(service_app, "refine_pitch_message", fake_refine_pitch_message)

        res = client.post("/refine", json={
            "message": "Our platform reduces deployment time by 80% for enterprise teams",
            "persona": "CTO at a mid-stage startup, technical background",
        })

        assert res.status_code == 503
        assert "OpenRouter API key is missing" in res.json()["detail"]


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
