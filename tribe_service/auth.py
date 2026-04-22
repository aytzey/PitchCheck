"""Lightweight optional auth for the shared PitchServer runtime."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HASH_ITERATIONS = 210_000
MIN_PASSWORD_LENGTH = 8
DEFAULT_SESSION_TTL_SECONDS = 24 * 60 * 60


class AuthError(Exception):
    """Base auth error."""


class AuthConfigurationError(AuthError):
    """Auth is enabled but the seed/state is missing."""


class InvalidCredentialsError(AuthError):
    """Username/password or token is invalid."""


class InvalidCredentialUpdateError(AuthError):
    """Credential update payload is invalid."""


@dataclass
class Session:
    username: str
    expires_at: float


def auth_required() -> bool:
    return _env_bool("PITCHSERVER_AUTH_REQUIRED")


def auth_file_path() -> Path:
    return Path(os.getenv("PITCHSERVER_AUTH_FILE", "/auth/pitchserver_auth.json"))


def _env_bool(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in {"1", "true", "yes", "on"}


def _now() -> float:
    return time.time()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _session_ttl_seconds() -> int:
    raw = os.getenv("PITCHSERVER_SESSION_TTL_SECONDS", "").strip()
    if not raw:
        return DEFAULT_SESSION_TTL_SECONDS
    try:
        return max(300, int(raw))
    except ValueError:
        return DEFAULT_SESSION_TTL_SECONDS


def _normalize_username(username: str | None) -> str:
    value = (username or "").strip()
    if not 3 <= len(value) <= 64:
        raise InvalidCredentialUpdateError("Username must be 3-64 characters.")
    if any(character.isspace() for character in value):
        raise InvalidCredentialUpdateError("Username cannot contain whitespace.")
    return value


def _validate_password(password: str | None) -> str:
    value = password or ""
    if len(value) < MIN_PASSWORD_LENGTH:
        raise InvalidCredentialUpdateError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )
    if any(character in value for character in "\r\n"):
        raise InvalidCredentialUpdateError("Password cannot contain line breaks.")
    return value


def _hash_password(password: str, salt: bytes | None = None) -> dict[str, Any]:
    salt = salt or secrets.token_bytes(24)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, HASH_ITERATIONS)
    return {
        "algorithm": "pbkdf2_sha256",
        "iterations": HASH_ITERATIONS,
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(digest).decode("ascii"),
    }


def _verify_password(password: str, state: dict[str, Any]) -> bool:
    try:
        salt = base64.b64decode(str(state["salt"]))
        expected = base64.b64decode(str(state["hash"]))
        iterations = int(state.get("iterations") or HASH_ITERATIONS)
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


class AuthStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}

    def status(self) -> dict[str, Any]:
        configured = auth_file_path().exists() or bool(
            os.getenv("PITCHSERVER_AUTH_SEED_USERNAME")
            and os.getenv("PITCHSERVER_AUTH_SEED_PASSWORD")
        )
        return {"required": auth_required(), "configured": configured}

    def login(self, username: str, password: str) -> dict[str, Any]:
        if not auth_required():
            token = self._create_session("dev")
            return self._session_response("dev", token)
        with self._lock:
            state = self._ensure_state()
            if username.strip() != state.get("username") or not _verify_password(password, state):
                raise InvalidCredentialsError("Invalid PitchServer username or password.")
            token = self._create_session(str(state["username"]))
            return self._session_response(str(state["username"]), token)

    def verify_token(self, token: str | None) -> str:
        if not auth_required():
            return "dev"
        if not token:
            raise InvalidCredentialsError("Missing PitchServer auth token.")
        with self._lock:
            self._prune_expired_sessions()
            session = self._sessions.get(token)
            if not session:
                raise InvalidCredentialsError("Invalid or expired PitchServer auth token.")
            return session.username

    def change_credentials(
        self,
        *,
        token: str | None,
        current_password: str,
        new_username: str,
        new_password: str,
    ) -> dict[str, Any]:
        if not auth_required():
            raise AuthConfigurationError("PitchServer auth is not enabled.")
        username = self.verify_token(token)
        next_username = _normalize_username(new_username)
        next_password = _validate_password(new_password)
        with self._lock:
            state = self._ensure_state()
            if username != state.get("username") or not _verify_password(current_password, state):
                raise InvalidCredentialsError("Current PitchServer password is incorrect.")
            now = _now_iso()
            updated = {
                "username": next_username,
                **_hash_password(next_password),
                "created_at": state.get("created_at") or now,
                "updated_at": now,
            }
            self._write_state(updated)
            self._sessions.clear()
            new_token = self._create_session(next_username)
            return self._session_response(next_username, new_token)

    def _ensure_state(self) -> dict[str, Any]:
        path = auth_file_path()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        username = _normalize_username(os.getenv("PITCHSERVER_AUTH_SEED_USERNAME"))
        password = _validate_password(os.getenv("PITCHSERVER_AUTH_SEED_PASSWORD"))
        now = _now_iso()
        state = {
            "username": username,
            **_hash_password(password),
            "created_at": now,
            "updated_at": now,
        }
        self._write_state(state)
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        path = auth_file_path()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(tmp, flags, 0o600), "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = Session(
            username=username,
            expires_at=_now() + _session_ttl_seconds(),
        )
        return token

    def _session_response(self, username: str, token: str) -> dict[str, Any]:
        session = self._sessions[token]
        return {
            "ok": True,
            "username": username,
            "token": token,
            "expires_at": int(session.expires_at),
        }

    def _prune_expired_sessions(self) -> None:
        now = _now()
        expired = [token for token, session in self._sessions.items() if session.expires_at <= now]
        for token in expired:
            self._sessions.pop(token, None)


AUTH_STORE = AuthStore()
