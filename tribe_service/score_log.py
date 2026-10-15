"""Opt-in, numbers-only score log for later outcome joins (off unless path AND key are set).

Never writes message, persona or any LLM prose; texts are reduced to keyed HMAC ids so
outcomes (reply, meeting, rating) can be joined later by pitch_id.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import unicodedata

LOG_SCHEMA = 1
_LOCK = threading.Lock()  # ponytail: process lock; uvicorn runs one worker. Use fcntl.flock if workers > 1.
LOGGER = logging.getLogger(__name__)


def _key() -> bytes | None:
    key = os.getenv("PITCHCHECK_SCORE_LOG_KEY", "")
    return key.encode() if len(key) >= 32 else None  # refuse unkeyed (plain sha256) ids


def enabled() -> bool:
    return bool(os.getenv("PITCHCHECK_SCORE_LOG_PATH")) and _key() is not None


def pitch_id(text: str) -> str | None:
    key = _key()
    norm = " ".join(unicodedata.normalize("NFC", text).split())
    return hmac.new(key, norm.encode(), hashlib.sha256).hexdigest() if key else None


def append(record: dict) -> None:
    path = os.getenv("PITCHCHECK_SCORE_LOG_PATH")
    if not path or _key() is None:
        return
    try:
        line = json.dumps({"log_schema": LOG_SCHEMA, **record}, separators=(",", ":"), allow_nan=False)
        with _LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError) as exc:
        LOGGER.warning("Score log write failed (%s)", type(exc).__name__)  # never fails /score
