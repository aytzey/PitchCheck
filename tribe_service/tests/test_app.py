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
        assert report["robustness"]["prompt_doctrine_version"] == "doctrine_v2"
        assert report["robustness"]["context_fit_rubric_version"] == "v2"
        assert report["robustness"]["persona_read_prior"] is None
        assert 0 <= report["robustness"]["confidence"] <= 1
        assert 0 < report["robustness"]["prediction_quality_weight"] <= 1
        assert isinstance(report["robustness"]["neuro_axes"], dict)
        assert "self_value" in report["robustness"]["neuro_axes"]
        assert isinstance(report["robustness"]["scientific_caveats"], list)
        assert report["fmri_output"]["prediction_subject_basis"] == "average_subject"
        assert report["fmri_output"]["cortical_mesh"] == "fsaverage5"
        assert report["fmri_output"]["roi_summary"] is None
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

    def test_refine_selects_a_measured_candidate_and_rejects_invented_proof(self, monkeypatch, tmp_path):
        from tribe_service import llm_layer
        measured = []
        original = "Benimle Çilekeş konserine gelmelisin, çok eğleneceğiz."
        candidates = [
            "Çilekeş sana göre değil biliyorum; benimle bir akşam geçirmek ister misin?",
            "Çilekeş favorin değil ama seninle konsere gitmeyi isterim. Eşlik eder misin?",
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
                    {"segments": 4, "voxel_count": 20484, "temporal_trace": [.12, .31, .28, .18],
                     "roi_summary": {"locations": {"mpfc_falk2010": {"minus_cortex": -value / 100}}}}, signals)

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
            assert 'roi_' not in prompt
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
        log = tmp_path / "scores.jsonl"
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_PATH", str(log))
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_KEY", "k" * 32)
        res = client.post("/refine", json=payload)
        assert res.status_code == 200
        data = res.json()
        rows = log.read_text(encoding="utf-8").splitlines()
        assert len(rows) == 4 and "Çilekeş" not in log.read_text(encoding="utf-8")
        assert sum('"selected":true' in row for row in rows) == 1
        monkeypatch.delenv("PITCHCHECK_SCORE_LOG_PATH")
        assert data["refined_message"] == candidates[1]
        assert data["model"] == "test-refiner"
        assert measured == [original, *candidates]
        assert data["tribe_guidance"]["candidate_count"] == 3
        assert data["tribe_guidance"]["prompt_doctrine_version"] == "doctrine_v2"
        assert data["tribe_guidance"]["context_fit_rubric_version"] == "v2"
        assert data["tribe_guidance"]["selected_id"] == "c2"
        assert data["tribe_guidance"]["roi_shadow_selected_id"] == "c2"
        assert data["tribe_guidance"]["evaluations"][1]["roi_deltas"]["mpfc_falk2010"] == -0.1
        assert data["tribe_guidance"]["evaluations"][3]["eligible"] is False
        # Context fit alone ties c1/c2; the measured TRIBE prior is what picked c2.
        assert data["tribe_guidance"]["selected_id_semantic_only"] == "c1"
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

        # A model-approved, high-scoring placeholder is still not ready to send.
        placeholder = "Çilekeş konserine beraber gidelim mi? [onun sevdiği bir mekan]"
        candidates[0] = placeholder
        scores[placeholder] = 99
        res = client.post("/refine", json=payload)
        assert res.json()["tribe_guidance"]["evaluations"][1]["eligible"] is False
        assert res.json()["refined_message"] != placeholder

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


@pytest.mark.parametrize("jev", [False, True])
@pytest.mark.parametrize("all_wrong_length", [False, True])
def test_refine_rejects_length_mismatch_after_measuring_all_candidates(monkeypatch, jev, all_wrong_length):
    import json
    import httpx
    from tribe_service import llm_layer
    paragraph = "Our proposal covers the complete product range while preserving your stated production constraints."
    original = "\n\n".join([" ".join([paragraph] * 10)] * 3)
    drafts = ["Can we review the proposed coverage?",
              "Would you review the complete coverage?" if all_wrong_length else original.replace("Our", "The", 1),
              original + "\n\n" + original]
    measured, writer_payloads, critic_payloads = [], [], []
    monkeypatch.setattr(llm_layer, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(llm_layer, "OPENROUTER_ENABLED", True)
    def measure(text):
        measured.append(text)
        return [[.2]]
    def analysis(*args, **kwargs):
        return ({}, {"segments": 4, "voxel_count": 20484, "temporal_trace": [.2, .5, .1, .3],
                     "temporal_trace_basis": "synthetic_word_order"},
                {key: 50.0 for key in service_app.PERSUASION_SIGNAL_LABELS})
    def review_score(name):
        return 2 if name.startswith("original") else 3 if name.startswith("c2") else 4
    def post(url, **kwargs):
        payload = kwargs["json"]
        if url.endswith("/decisions"):
            questions = payload["questions"]
            chosen = {"relationship": "business", "objection": "relevance", "angle": "evidence",
                      "tone": "professional", "repair": "proof", "move": "concrete_value", "repair_target": "whole",
                      "winner": "c1"}
            if "winner" in questions:
                critic_payloads.append(payload)
            answers = {}
            for name, question in questions.items():
                kind = question["type"]
                if kind == "choice":
                    answers[name] = {"type": kind, "choice": chosen[name], "confidence": .9,
                                     "probabilities": {key: float(key == chosen[name]) for key in question["criteria"]}}
                elif kind == "noul":
                    answers[name] = {"type": kind, "noul": 1.0}
                else:
                    score = review_score(name)
                    answers[name] = {"type": kind, "score": score, "confidence": .9,
                                     "probabilities": {str(i): float(i == score) for i in range(5)}}
            body = {"model": "typesafe/jev-test", "answers": answers}
        elif "Actual TRIBE measurements" in payload["messages"][1]["content"]:
            critic_payloads.append(payload)
            body = {"choices": [{"message": {"content": json.dumps({"evaluations": [
                {"id": key, "supported": True, "intent_preserved": True, "recipient_respected": True,
                 "voice_preserved": True, "context_fit": {facet: 25 * review_score(key)
                     for facet in llm_layer.CONTEXT_FIT_KEYS}, "issues": []}
                for key in ("original", "c1", "c2", "c3")]})}}]}
        else:
            writer_payloads.append(payload)
            content = {"drafts": [{"anchor": "proposal", "idea": idea, "message": draft}
                for idea, draft in zip(("Coverage", "Evidence", "Request"), drafts)]} if jev else {"candidates": drafts}
            body = {"choices": [{"message": {"content": json.dumps(content)}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))
    monkeypatch.setattr(service_app, "score_text", measure)
    monkeypatch.setattr(service_app, "analyze_predictions", analysis)
    monkeypatch.setattr(llm_layer.httpx, "post", post)

    response = client.post("/refine", json={"message": original, "persona": "A manufacturer reviewing product coverage",
        "platform": "email", "jevStrategy": jev, "forceRewrite": True})

    assert response.status_code == 200, response.text
    result = response.json()
    assert measured == [original, *drafts]
    assert len(writer_payloads) == len(critic_payloads) == 1
    assert result["tribe_guidance"]["candidate_count"] == 3
    assert result["tribe_guidance"]["prompt_doctrine_version"] == "doctrine_v2"
    assert result["tribe_guidance"]["context_fit_rubric_version"] == "v2"
    assert result["refined_message"] == (original if all_wrong_length else drafts[1])
    evaluations = result["tribe_guidance"]["evaluations"]
    assert evaluations[1]["eligible"] is evaluations[3]["eligible"] is False
    assert evaluations[2]["eligible"] is (not all_wrong_length)
    assert "length" in str(result["critic_notes"]).lower()
    critic_text = json.dumps(critic_payloads[0], ensure_ascii=False)
    assert "80–120%" in critic_text and "paragraph" in critic_text and "details" in critic_text


def test_jev_plans_from_real_baseline_before_one_writer_pass_and_checks_all_drafts(monkeypatch):
    import json
    import httpx
    from tribe_service import llm_layer
    original = 'benimle çilekeş konserine gelmelisin harika bi grup çok eğlencez'
    drafts = ['Çilekeş konserine benimle gelir misin? Yanımda sen ol istiyorum.',
              'Çilekeş zevkime güvenmediğini biliyorum, eşlik etme teklifim hâlâ var. Beraber gidelim mi?',
              'İki bilet aldım, cuma günü konsere gidelim mi?']
    events = []
    winner = ['c2']
    served_model = 'z-ai/glm-5.3-flash:served-version'
    monkeypatch.setattr(llm_layer, 'OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setattr(llm_layer, 'OPENROUTER_REASONING_EFFORT', 'high')
    def measure(text):
        events.append(('measure', text))
        return [[.2]]
    def analysis(*args, **kwargs):
        return ({}, {'segments': 4, 'voxel_count': 20484, 'temporal_trace': [.2, .5, .1, .3],
                     'temporal_trace_basis': 'synthetic_word_order'},
                {key: 50.0 for key in service_app.PERSUASION_SIGNAL_LABELS})
    def post(url, **kwargs):
        payload = kwargs['json']
        if url.endswith('/decisions'):
            state, questions = payload['state'], payload['questions']
            events.append(('jev', state))
            if 'relationship' in questions:
                assert events[0] == ('measure', original)
                assert state['baseline']['voxel_count'] == 20484
                assert state['baseline']['temporal_trace'] == [.2, .5, .1, .3]
                selected = {'relationship': 'romantic', 'objection': 'taste', 'angle': 'company',
                            'tone': 'warm', 'repair': 'recipient'}
            else:
                assert [row['message'] for row in state['measurements']] == [original, *drafts]
                selected = {'winner': winner[0]}
            answers = {}
            for name, q in questions.items():
                if q['type'] == 'choice':
                    choice = selected.get(name, next(iter(q['criteria'])))
                    answers[name] = {'type': 'choice', 'choice': choice, 'confidence': .95,
                                     'probabilities': {key: float(key == choice) for key in q['criteria']}}
                elif q['type'] == 'noul':
                    # A mistaken editor cannot authorize fabricated ticket/date facts.
                    answers[name] = {'type': 'noul', 'noul': .99}
                else:
                    score = 2.0 if name.startswith('original_') else 3.9
                    answers[name] = {'type': 'score', 'score': score, 'confidence': .9,
                                     'probabilities': {str(i): (float(i == 2) if score == 2 else .1 if i == 3 else .9 if i == 4 else 0) for i in range(5)}}
            body = {'model': 'typesafe/jev-1.13-20260917', 'answers': answers}
        else:
            events.append(('writer', payload))
            assert payload['model'] == 'z-ai/glm-5.3-flash'
            assert payload['provider'] == {'order': ['baseten/fp8', 'fireworks', 'coreweave/nvfp4'],
                                           'allow_fallbacks': True, 'require_parameters': True}
            assert 'models' not in payload
            assert payload['reasoning'] == {'effort': 'low', 'exclude': True}
            assert payload['max_tokens'] == 1536
            assert kwargs['timeout'] == llm_layer.OPENROUTER_TIMEOUT
            prompt = payload['messages'][1]['content']
            assert 'company' in prompt and 'approximate_word_order' in prompt
            assert 'continuity' in prompt
            assert 'temporal_trace' not in prompt
            assert '15 saniyelik klip' not in prompt
            plans = [{'anchor': 'çilekeş', 'idea': idea, 'message': draft}
                     for idea, draft in zip(['Kendine takılan davet', 'Birlikte deneyim', 'Yalın davet'], drafts)]
            body = {'model': served_model, 'provider': 'BaseTen',
                    'usage': {'completion_tokens': 300, 'cost': .0002},
                    'choices': [{'message': {'content': json.dumps({'drafts': plans})}}]}
        return httpx.Response(200, json=body, request=httpx.Request('POST', url))
    monkeypatch.setattr(service_app, 'score_text', measure)
    monkeypatch.setattr(service_app, 'analyze_predictions', analysis)
    monkeypatch.setattr(llm_layer.httpx, 'post', post)
    response = client.post('/refine', json={'message': original, 'persona': 'çilekeşi sevmeyen flörtüm',
        'openRouterModel': 'google/gemini-3.5-flash-lite', 'jevStrategy': True, 'forceRewrite': True,
        'suggestions': ['15 saniyelik klip var, izlesin.']})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['model'] == served_model
    assert result['refined_message'] == drafts[1]
    assert [kind for kind, _ in events] == ['measure', 'jev', 'writer', 'measure', 'measure', 'measure', 'jev']
    assert result['decision_strategy']['choices']['angle'] == 'company'
    proof = result['tribe_guidance']
    assert proof['writer_passes'] == 1
    assert proof['writer_call']['requested_model'] == 'z-ai/glm-5.3-flash'
    assert proof['writer_call']['served_model'] == served_model
    assert proof['writer_call']['provider'] == 'BaseTen'
    assert proof['writer_call']['reasoning'] == {'effort': 'low', 'exclude': True}
    assert proof['writer_call']['reasoning_effort'] == 'low'
    assert proof['writer_call']['usage'] == {'completion_tokens': 300, 'cost': .0002}
    assert proof['quality_model'].startswith('typesafe/jev-')
    assert proof['evaluations'][3]['eligible'] is False
    assert result['methodology'] == 'tribe_jev_planned_refinement'
    # Even Jev's preferred candidate cannot bypass the deterministic fact guard.
    events.clear()
    winner[0] = 'c3'
    response = client.post('/refine', json={'message': original, 'persona': 'çilekeşi sevmeyen flörtüm',
        'openRouterModel': 'google/gemini-3.5-flash-lite', 'jevStrategy': True, 'forceRewrite': True})
    assert response.status_code == 200
    assert response.json()['refined_message'] in drafts[:2]
    assert sum(kind == 'writer' for kind, _ in events) == 1


class TestScoreLog:
    MESSAGE = "Hi Jordan, our platform cut deploy rollbacks 31% for teams like yours. Worth a look Tuesday?"
    PERSONA = "Jordan Park, Staff Engineer at a fintech, hates flaky deploys"

    def _score(self):
        res = client.post("/score", json={"message": self.MESSAGE, "persona": self.PERSONA})
        assert res.status_code == 200
        return res.json()["report"]

    def test_off_by_default_and_without_a_long_key(self, monkeypatch, tmp_path):
        log = tmp_path / "scores.jsonl"
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_PATH", str(log))
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_KEY", "short")
        report = self._score()
        assert not log.exists()
        assert "score_log_id" not in report["robustness"]
        assert client.get("/health").json()["score_log_enabled"] is False

    def test_writes_numbers_only_with_keyed_ids(self, monkeypatch, tmp_path):
        import json
        log = tmp_path / "scores.jsonl"
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_PATH", str(log))
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_KEY", "k" * 32)
        report = self._score()

        line = log.read_text(encoding="utf-8")
        record = json.loads(line)
        assert "Jordan" not in line and "fintech" not in line  # no message or persona text
        assert record["pitch_id"] == report["robustness"]["score_log_id"] and len(record["pitch_id"]) == 64
        assert record["final_score"] == report["robustness"]["final_score"]
        assert record["word_count"] == 16 and record["b1_tribe_informed"] is True

    def test_persona_prior_shadow_and_labels_logged_without_facts(self, monkeypatch, tmp_path):
        import json
        from tribe_service import llm_layer
        log = tmp_path / "scores.jsonl"
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_PATH", str(log))
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_KEY", "k" * 32)
        monkeypatch.setattr(llm_layer, "_call_openrouter", lambda *a, **k: {"persuasion_score": 65,
            "context_fit": {**{key: {"score": 60} for key in llm_layer.CONTEXT_FIT_KEYS},
                "recipient_stance": "skeptical", "personalization_level": "recipient_specific",
                "unused_persona_facts": ["hates flaky deploys"]}})
        report = self._score()
        line = log.read_text()
        record = json.loads(line)
        assert record["recipient_stance"] == "skeptical"
        assert record["personalization_level"] == "recipient_specific"
        assert record["unused_persona_fact_count"] == 1
        assert record["persona_read_prior"] == report["robustness"]["persona_read_prior"]
        assert record["score_terms"] == report["robustness"]["score_terms"]
        assert "hates flaky deploys" not in line and "Jordan" not in line

    def test_refine_prior_shadow_logged_without_prose(self, monkeypatch, tmp_path):
        import json
        log = tmp_path / "scores.jsonl"
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_PATH", str(log))
        monkeypatch.setenv("PITCHCHECK_SCORE_LOG_KEY", "k" * 32)
        service_app._log_refine_candidates(self.MESSAGE, {"recipient_stance": "skeptical", "selected_id": "c1",
            "evaluations": [{"id": "c1", "message": self.MESSAGE, "semantic_score": 63.,
                             "semantic_score_without_prior": 60., "personalization_level": "recipient_specific",
                             "issues": ["private critic prose"]}]}, jev=False)
        line = log.read_text()
        record = json.loads(line)
        assert record["recipient_stance"] == "skeptical"
        assert record["personalization_level"] == "recipient_specific"
        assert record["semantic_score_without_prior"] == 60.
        assert "private critic prose" not in line and "Jordan" not in line

@pytest.mark.parametrize('weight,compatible,expected', [(0.15, True, 0.15), (0.15, False, 0.075), (0, True, None)])
def test_full_width_roi_http_score(monkeypatch, weight, compatible, expected):
    import numpy as np
    from tribe_service import engine, persuasion_features, llm_layer
    monkeypatch.setattr(service_app, 'score_text', lambda message: np.random.default_rng(5).normal(size=(6, 20484)).astype(np.float32))
    monkeypatch.setattr(persuasion_features, 'ROI_PRIOR_WEIGHT', weight)
    monkeypatch.setattr(persuasion_features, 'ROI_ENCODER_MISMATCH_FACTOR', 0.5)
    monkeypatch.setattr(engine, '_loaded_runtime_config', {'text_model': engine.TRIBE_CHECKPOINT_TEXT_MODEL if compatible else 'NousResearch/Hermes-3-Llama-3.2-3B'})
    monkeypatch.setattr(llm_layer, '_call_openrouter', lambda *args, **kwargs: None)
    response = client.post('/score', json={'message': 'Our deployment workflow helps engineering teams inspect and restore failed releases.', 'persona': 'An engineering manager'})
    assert response.status_code == 200
    report = response.json()['report']
    assert isinstance(report['fmri_output']['roi_summary'], dict)
    assert set(report['fmri_output']['roi_summary']) >= {'rois', 'locations'}
    assert isinstance(report['persuasion_score'], int)
    assert report['persuasion_evidence']['methodology_version'] == 'neural_only_v2.3'
    if expected is None:
        assert 'roi_prior' not in report['persuasion_evidence']
        assert report['robustness']['roi_prior'] is None
    else:
        prior = report['robustness']['roi_prior']
        assert prior['applied'] and prior['weight'] == expected
        assert prior['provenance'].endswith(';encoder_mismatch') == (not compatible)
        r = report['robustness']
        assert abs(r['final_score'] - r['final_score_without_literature_priors']) <= 6.05
        assert r['score_terms']['roi_prior']['provenance'] == prior['provenance']


@pytest.mark.parametrize("enabled,mock", [(False, False), (True, True)])
def test_specificity_disabled_and_mock_make_no_extra_calls(monkeypatch, enabled, mock):
    from tribe_service import engine, llm_layer
    calls = []
    def score(text):
        calls.append(text)
        return engine.np.ones((5, 20))
    monkeypatch.setattr(service_app, "SPECIFICITY_ENABLED", enabled)
    monkeypatch.setattr(service_app, "TRIBE_ALLOW_MOCK", mock)
    monkeypatch.setattr(service_app, "score_text", score)
    monkeypatch.setattr(service_app, "make_specificity_twin", lambda *a: pytest.fail("Twin must not run"))
    monkeypatch.setattr(service_app, "measure_specificity", lambda *a: pytest.fail("Contrast must not run"))
    monkeypatch.setattr(llm_layer, "_call_openrouter", lambda *a, **k: None)
    response = client.post("/score", json={"message": "Our platform reduces deployment time for teams handling frequent releases.", "persona": "An engineering manager"})
    assert response.status_code == 200 and len(calls) == 1
    assert "specificity_contrast" not in response.json()["report"]["robustness"]


@pytest.mark.parametrize("compatible,factor,expected", [(True, .5, 3), (False, .5, 1.5), (False, 0, 0)])
def test_specificity_http_signed_prior_encoder_and_explicit_reason(monkeypatch, compatible, factor, expected):
    import numpy as np
    from tribe_service import engine, llm_layer, persuasion_features as pf, roi_features, specificity as sp
    from tribe_service.tests.test_specificity import MESSAGE, PERSONA, SPANS
    calls, twin_calls = [], []
    assets = roi_features.load_roi_assets()
    a = np.random.default_rng(17).normal(-1, .1, size=(5, 20484))
    b = a.copy()
    for key in sp.SPECIFICITY_ROI_KEYS[:2]:
        b[:, assets[3][key]] -= .2
    def score(text):
        calls.append(text)
        return a if text == MESSAGE else b
    def twin(*args):
        twin_calls.append(True)
        return sp.build_specificity_twin(MESSAGE, PERSONA, SPANS)
    monkeypatch.setattr(service_app, "SPECIFICITY_ENABLED", True)
    monkeypatch.setattr(service_app, "TRIBE_ALLOW_MOCK", False)
    monkeypatch.setattr(service_app, "ROI_ENCODER_MISMATCH_FACTOR", factor)
    monkeypatch.setattr(llm_layer, "ROI_ENCODER_MISMATCH_FACTOR", factor)
    monkeypatch.setattr(pf, "ROI_PRIOR_WEIGHT", 0)
    monkeypatch.setattr(service_app, "score_text", score)
    monkeypatch.setattr(sp, "score_text", score)
    monkeypatch.setattr(service_app, "make_specificity_twin", twin)
    monkeypatch.setattr(engine, "_text_feature_metadata", lambda: {
        "text_feature_model": "llama" if compatible else "hermes", "text_feature_compatible": compatible})
    monkeypatch.setattr(llm_layer, "_call_openrouter", lambda *a, **k: None)
    response = client.post("/score", json={"message": MESSAGE, "persona": PERSONA})
    assert response.status_code == 200
    report = response.json()["report"]
    r = report["robustness"]
    c = r["specificity_contrast"]
    assert c["prior_points"] == expected
    if expected:
        assert c["available"] and len(calls) == 2 and len(twin_calls) == 1
        assert report["persuasion_score"] == round(r["score_without_specificity_prior"] + expected)
        assert (";encoder_mismatch" in r["score_terms"]["specificity_prior"]["provenance"]) == (not compatible)
    else:
        assert c["reason"] == "encoder_mismatch_disabled" and len(calls) == 1 and not twin_calls


def test_specificity_optional_timeout_is_http_200(monkeypatch):
    from tribe_service import engine, llm_layer
    from tribe_service.tests.test_specificity import MESSAGE, PERSONA, SPANS
    from tribe_service.specificity import build_specificity_twin
    monkeypatch.setattr(service_app, "SPECIFICITY_ENABLED", True)
    monkeypatch.setattr(service_app, "TRIBE_ALLOW_MOCK", False)
    monkeypatch.setattr(service_app, "score_text", lambda text: engine.np.ones((5, 20)))
    monkeypatch.setattr(service_app, "make_specificity_twin", lambda *a: build_specificity_twin(MESSAGE, PERSONA, SPANS))
    monkeypatch.setattr(llm_layer, "_call_openrouter", lambda *a, **k: None)
    run = service_app._run_with_backpressure
    async def timeout(function, *args, **kwargs):
        if function is service_app.measure_specificity:
            raise service_app.ScoreQueueTimeoutError
        return await run(function, *args, **kwargs)
    monkeypatch.setattr(service_app, "_run_with_backpressure", timeout)
    response = client.post("/score", json={"message": MESSAGE, "persona": PERSONA})
    assert response.status_code == 200
    assert response.json()["report"]["robustness"]["specificity_contrast"]["reason"] == "skipped_ScoreQueueTimeoutError"


def test_specificity_never_queues_ahead_of_primary_scores(monkeypatch):
    """Two primary /score calls keep their slot; optional contrast yields under contention."""
    from tribe_service import engine, llm_layer
    from tribe_service.tests.test_specificity import MESSAGE, PERSONA, SPANS
    from tribe_service.specificity import build_specificity_twin
    entered, finish, finish_second = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def score(text):
        calls.append(text)
        if len(calls) == 1:
            entered.set()
            assert finish.wait(2)
        else:
            assert finish_second.wait(2)
        return engine.np.ones((5, 20))
    def measure(*args):
        assert finish_second.is_set(), "Optional work cannot run ahead of a primary score"
        return {"available": False, "reason": "roi_unavailable"}
    async def run_case():
        monkeypatch.setattr(service_app, "_score_lock", asyncio.Semaphore(1))
        monkeypatch.setattr(service_app, "_llm_lock", asyncio.Semaphore(2))
        monkeypatch.setattr(service_app, "_pipeline_lock", asyncio.Lock())
        monkeypatch.setattr(service_app, "SPECIFICITY_ENABLED", True)
        monkeypatch.setattr(service_app, "TRIBE_ALLOW_MOCK", False)
        monkeypatch.setattr(service_app, "TRIBE_SCORE_QUEUE_TIMEOUT_SECONDS", .5)
        monkeypatch.setattr(service_app, "score_text", score)
        monkeypatch.setattr(service_app, "make_specificity_twin", lambda *a: build_specificity_twin(MESSAGE, PERSONA, SPANS))
        monkeypatch.setattr(service_app, "measure_specificity", measure)
        monkeypatch.setattr(llm_layer, "_call_openrouter", lambda *a, **k: None)
        request = service_app.PitchScoreRequest(message=MESSAGE, persona=PERSONA)
        first = asyncio.create_task(service_app.score_pitch(request, _="test"))
        assert await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(service_app.score_pitch(request, _="test"))
        # Wait until the second primary owns a queued acquisition, then release the first.
        for _ in range(100):
            if service_app._score_lock._waiters:
                break
            await asyncio.sleep(.001)
        assert service_app._score_lock._waiters
        finish.set()
        # Keep the second primary busy until the first has skipped its optional contrast.
        first_report = await first
        assert first_report["report"]["robustness"]["specificity_contrast"]["reason"] == "skipped_busy"
        finish_second.set()
        second_report = await second
        assert second_report["report"]["robustness"]["specificity_contrast"]["available"] is False
        assert len(calls) == 2
    asyncio.run(run_case())


def test_specificity_refine_penalty_zero_skips_twin_calls(monkeypatch):
    monkeypatch.setattr(service_app, "SPECIFICITY_ENABLED", True)
    monkeypatch.setattr(service_app, "TRIBE_ALLOW_MOCK", False)
    monkeypatch.setattr(service_app, "SPECIFICITY_REFINE_PENALTY", 0)
    monkeypatch.setattr(service_app, "make_specificity_twin", lambda *a: pytest.fail("Refine twin must not run"))
    monkeypatch.setattr(service_app, "refine_pitch_message", lambda **k: {
        "model": "test", "needs_clarification": True, "questions": []})
    response = client.post("/refine", json={"message": "Our platform reduces deployment time for teams handling frequent releases.",
                                           "persona": "An engineering manager"})
    assert response.status_code == 200 and response.json()["needs_clarification"]
