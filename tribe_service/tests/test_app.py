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
        assert isinstance(report["top_moves"], list)
        assert len(report["top_moves"]) >= 1
        assert report["top_moves"][0]["title"]
        assert report["top_moves"][0]["do"]

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

    def test_runtime_load_is_idempotent_and_can_be_unloaded(self):
        client.post("/runtime/unload")
        for _ in range(2):
            loaded = client.post("/runtime/load")
            assert loaded.status_code == 200
            assert loaded.json()["ok"] is True
            assert loaded.json()["pipeline"]["model_loaded"] is True
            assert loaded.json()["pipeline"]["text_model_loaded"] is True
            assert loaded.json()["pipeline"]["active_scores"] == 0
        assert client.post("/runtime/unload").json()["ok"] is True
        assert client.get("/health").json()["pipeline"]["text_model_loaded"] is False

    def test_cancelled_unload_keeps_pipeline_locked_until_worker_finishes(self, monkeypatch):
        started = threading.Event()
        allow_finish = threading.Event()

        def slow_unload():
            started.set()
            allow_finish.wait(2)

        async def run_case():
            monkeypatch.setattr(service_app, "_pipeline_lock", asyncio.Lock())
            monkeypatch.setattr(service_app, "_active_scores", 0)
            monkeypatch.setattr(service_app, "is_model_loaded", lambda: True)
            monkeypatch.setattr(service_app, "unload_model", slow_unload)
            task = asyncio.create_task(service_app._unload_pipeline("test"))
            try:
                assert await asyncio.to_thread(started.wait, 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert service_app._pipeline_lock.locked()
            finally:
                allow_finish.set()
                if service_app._worker_tasks:
                    await asyncio.gather(*tuple(service_app._worker_tasks), return_exceptions=True)
            assert not service_app._pipeline_lock.locked()

        asyncio.run(run_case())

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

    def test_cancelled_request_keeps_gpu_busy_until_worker_finishes(self, monkeypatch):
        started, finish = threading.Event(), threading.Event()

        def blocked_score(_):
            started.set()
            assert finish.wait(2)
            return [[1.0]]

        async def run_case():
            monkeypatch.setattr(service_app, "_score_lock", asyncio.Semaphore(1))
            monkeypatch.setattr(service_app, "_pipeline_lock", asyncio.Lock())
            monkeypatch.setattr(service_app, "_active_scores", 0)
            monkeypatch.setattr(service_app, "score_text", blocked_score)
            task = asyncio.create_task(service_app._score_text_with_backpressure("valid pitch"))
            try:
                assert await asyncio.to_thread(started.wait, 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert (await service_app._pipeline_status())["active_scores"] == 1
                assert service_app._score_lock.locked()
                assert (await service_app._unload_pipeline("test"))["reason"] == "score_in_progress"
            finally:
                finish.set()
                for _ in range(100):
                    if not service_app._score_lock.locked():
                        break
                    await asyncio.sleep(0.01)
            assert not service_app._score_lock.locked()
            assert service_app._active_scores == 0

        asyncio.run(run_case())

    def test_cancelled_refine_holds_gpu_slot_for_the_entire_candidate_batch(self, monkeypatch):
        started, finish = threading.Event(), threading.Event()
        selected = []

        def blocked_batch(*args):
            started.set()
            assert finish.wait(2)
            return []

        async def run_case():
            monkeypatch.setattr(service_app, "_score_lock", asyncio.Semaphore(1))
            monkeypatch.setattr(service_app, "_llm_lock", asyncio.Semaphore(2))
            monkeypatch.setattr(service_app, "_pipeline_lock", asyncio.Lock())
            monkeypatch.setattr(service_app, "_active_scores", 0)
            monkeypatch.setattr(service_app, "refine_pitch_message", lambda **kwargs: {"candidates": ["First draft", "Second draft", "Third draft"]})
            monkeypatch.setattr(service_app, "_measure_refine_candidates", blocked_batch)
            monkeypatch.setattr(service_app, "select_tribe_refinement", lambda *args: selected.append(True))
            request = service_app.PitchRefineRequest(message="A valid invitation", persona="A friend", platform="general")
            task = asyncio.create_task(service_app.refine_pitch(request, _="test"))
            try:
                assert await asyncio.to_thread(started.wait, 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert service_app._score_lock.locked()
                assert (await service_app._pipeline_status())["active_scores"] == 1
                assert (await service_app._unload_pipeline("test"))["reason"] == "score_in_progress"
                assert not selected
            finally:
                finish.set()
                for _ in range(100):
                    if not service_app._score_lock.locked():
                        break
                    await asyncio.sleep(0.01)
            assert not service_app._score_lock.locked()
            assert service_app._active_scores == 0

        asyncio.run(run_case())

    def test_streamed_oversized_body_rejected_before_validation(self):
        res = client.post("/score", content=iter([b" " * 70_000, b" " * 70_000]),
                          headers={"Content-Type": "application/json"})
        assert res.status_code == 413

    def test_llm_worker_retains_its_slot_after_timeout_and_recovers_after_failure(self, monkeypatch):
        started, finish = threading.Event(), threading.Event()

        def blocked_llm():
            started.set()
            assert finish.wait(2)
            raise RuntimeError("provider failed")

        async def run_case():
            lock = asyncio.Semaphore(1)
            monkeypatch.setattr(service_app, "TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS", 0.01)
            try:
                with pytest.raises(service_app.ScoreRunTimeoutError):
                    await service_app._run_with_backpressure(blocked_llm, lock=lock, timeout=0.01)
                assert started.is_set() and lock.locked()
                with pytest.raises(service_app.ScoreQueueTimeoutError):
                    await service_app._run_with_backpressure(lambda: None, lock=lock, timeout=1)
            finally:
                finish.set()
                await asyncio.gather(*service_app._worker_tasks, return_exceptions=True)
            assert not lock.locked()
            assert await service_app._run_with_backpressure(lambda: "recovered", lock=lock, timeout=1) == "recovered"

        asyncio.run(run_case())

    def test_refine_suggestion_length_is_bounded(self):
        res = client.post("/refine", json={
            "message": "A valid pitch message for engineering teams",
            "persona": "Engineering manager", "suggestions": ["x" * 2001],
        })
        assert res.status_code == 422

    def test_scoring_failure_does_not_log_customer_copy(self, monkeypatch, caplog):
        def fail_score(_):
            raise ValueError("confidential customer copy")
        monkeypatch.setattr(service_app, "score_text", fail_score)
        response = client.post("/score", json={
            "message": "A valid pitch for the engineering team", "persona": "Engineering manager",
        })
        assert response.status_code == 500
        assert "Scoring failed (ValueError)" in caplog.text
        assert "confidential customer copy" not in caplog.text

    def test_refine_selects_a_measured_candidate_and_rejects_invented_proof(self, monkeypatch):
        from tribe_service import llm_layer
        measured = []
        original = "Benimle Çilekeş konserine gelmelisin, çok eğleneceğiz."
        candidates = [
            "Çilekeş sana göre değil biliyorum; benimle bir akşam geçirmek ister misin?",
            "Çilekeş favorin değil ama seninle konsere gitmeyi isterim. Bana eşlik eder misin?",
            "Sana iki ücretsiz bilet aldım, 15 saniyelik klibi izle ve beraber gidelim.",
        ]
        scores = {original: 35, candidates[0]: 45, candidates[1]: 80, candidates[2]: 99}
        def fake_score(text):
            measured.append(text)
            return scores[text]

        def fake_analysis(value, **kwargs):
            signals = {key: value for key in service_app.PERSUASION_SIGNAL_LABELS}
            signals["cognitive_friction"] = 100 - value
            return ({"mean_abs": .25, "peak_abs": .6, "temporal_std": .08,
                     "spatial_spread": .07, "focus_ratio": .35, "sustain_ratio": .6},
                    {"segments": 4, "voxel_count": 20484, "temporal_trace": [.12, .31, .28, .18]}, signals)

        def fake_refine_pitch_message(**kwargs):
            assert kwargs["suggestions"] == ["Make the CTA easier"]
            assert kwargs["clarification_round"] == 1
            assert kwargs["force_rewrite"] is True
            return {
                "candidates": candidates, "model": "test-refiner",
                "refined_message": None, "safety_notes": [], "questions": [],
            }

        def fake_chat(system, prompt, model, temperature):
            import json
            assert "Actual TRIBE measurements" in prompt
            assert "Çilekeş" in prompt
            return json.dumps({"evaluations": [
                {"id": key, "supported": key != "c3", "intent_preserved": True,
                 "recipient_respected": True, "voice_preserved": True,
                 "context_fit": {facet: 40 if key == "original" else 80
                                 for facet in llm_layer.CONTEXT_FIT_KEYS},
                 "issues": ["Invented tickets and clip"] if key == "c3" else []}
                for key in ["original", "c1", "c2", "c3"]
            ]})

        monkeypatch.setattr(service_app, "score_text", fake_score)
        monkeypatch.setattr(service_app, "analyze_predictions", fake_analysis)
        monkeypatch.setattr(service_app, "refine_pitch_message", fake_refine_pitch_message)
        monkeypatch.setattr(llm_layer, "_post_refine_chat", fake_chat)
        payload = {
            "message": original, "persona": "Çilekeş'i sevmeyen flörtüm", "platform": "general",
            "suggestions": ["Make the CTA easier"],
            "clarificationRound": 1,
            "forceRewrite": True,
        }
        res = client.post("/refine", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["refined_message"] == candidates[1]
        assert data["model"] == "test-refiner"
        assert measured == [original, *candidates]
        assert data["tribe_guidance"]["candidate_count"] == 3
        assert data["tribe_guidance"]["selected_id"] == "c2"
        assert data["tribe_guidance"]["evaluations"][3]["eligible"] is False
        # Same semantic judgment, different real-model evidence: selection must change.
        scores[candidates[0]], scores[candidates[1]] = 90, 20
        res = client.post("/refine", json=payload)
        assert res.json()["refined_message"] == candidates[0]

        # Even a mistaken critic cannot authorize invented calendar/resource facts.
        fabricated = "Konser haftaya cuma, iki bilet aldım; benimle gelir misin?"
        candidates[0] = fabricated
        scores[fabricated] = 99
        res = client.post("/refine", json=payload)
        assert res.json()["tribe_guidance"]["evaluations"][1]["eligible"] is False
        assert res.json()["refined_message"] != fabricated
        payload["clarificationAnswers"] = [{"id": "date", "question": "Konser haftaya cuma mı, bilet var mı?", "answer": ""}]
        res = client.post("/refine", json=payload)
        assert res.json()["tribe_guidance"]["evaluations"][1]["eligible"] is False
        payload["clarificationAnswers"][0]["answer"] = "Konser haftaya cuma; iki bilet aldım."
        res = client.post("/refine", json=payload)
        assert res.json()["tribe_guidance"]["evaluations"][1]["eligible"] is True
        assert res.json()["refined_message"] == fabricated

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

    def test_refine_model_failure_does_not_expose_private_worker_details(self, monkeypatch, caplog):
        monkeypatch.setattr(service_app, "refine_pitch_message", lambda **kwargs: {"candidates": ["First draft", "Second draft", "Third draft"]})
        def fail_batch(*args):
            raise RuntimeError("confidential customer copy and private model path")
        monkeypatch.setattr(service_app, "_measure_refine_candidates", fail_batch)
        response = client.post("/refine", json={"message": "A valid invitation", "persona": "A friend", "platform": "general"})
        assert response.status_code == 502
        assert "confidential" not in response.text
        assert "confidential" not in caplog.text

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
    def test_session_store_prunes_expired_logins_and_stays_bounded(self, monkeypatch):
        from tribe_service.auth import AuthStore
        store = AuthStore()
        monkeypatch.setenv("PITCHSERVER_AUTH_REQUIRED", "0")
        monkeypatch.setattr("tribe_service.auth._now", lambda: 0)
        first = store.login("unused", "unused")["token"]
        monkeypatch.setattr("tribe_service.auth._now", lambda: 100_000)
        store.login("unused", "unused")
        assert first not in store._sessions
        for _ in range(130):
            store.login("unused", "unused")
        assert len(store._sessions) <= 128

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

    def test_runtime_load_requires_login_when_auth_enabled(self, monkeypatch, tmp_path):
        self._enable_auth(monkeypatch, tmp_path)
        assert client.post("/runtime/load").status_code == 401

    def test_logout_revokes_only_the_callers_session(self, monkeypatch, tmp_path):
        self._enable_auth(monkeypatch, tmp_path)
        headers = []
        for _ in range(2):
            login = client.post('/auth/login', json={'username': 'pitchserver', 'password': 'initial-pass-123'})
            assert login.status_code == 200
            headers.append({'Authorization': 'Bearer ' + login.json()['token']})
        assert client.post('/auth/logout', headers=headers[0]).status_code == 200
        assert client.post('/runtime/load', headers=headers[0]).status_code == 401
        assert client.post('/runtime/load', headers=headers[1]).status_code == 200

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
