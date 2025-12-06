"""
privacy_module.py
Implements:
- ephemeral session store with TTL (in-memory; swap to Redis for production),
- auto-delete mechanism,
- no-permanent-storage policy enforcement,
- input redaction helpers.
"""

from typing import Any, Dict, Optional, Callable
import time
import threading
import re
import uuid
from dataclasses import dataclass, field

# --- Configuration ---
DEFAULT_TTL_SECONDS = 300  # 5 minutes default session lifetime; adjust per policy
CLEANUP_INTERVAL_SECONDS = 60

# --- Simple in-memory TTL store (thread-safe) ---
@dataclass
class EphemeralEntry:
    value: Any
    created_at: float = field(default_factory=lambda: time.time())
    ttl: int = DEFAULT_TTL_SECONDS


class EphemeralStore:
    def __init__(self):
        self._store: Dict[str, EphemeralEntry] = {}
        self._lock = threading.RLock()
        self._stop = False
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            self._store[key] = EphemeralEntry(value=value, ttl=ttl or DEFAULT_TTL_SECONDS)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            e = self._store.get(key)
            if not e:
                return None
            if time.time() - e.created_at > e.ttl:
                del self._store[key]
                return None
            return e.value

    def delete(self, key: str):
        with self._lock:
            if key in self._store:
                del self._store[key]

    def keys(self):
        with self._lock:
            return list(self._store.keys())

    def _cleanup_loop(self):
        while not self._stop:
            now = time.time()
            with self._lock:
                expired = [k for k, v in self._store.items() if now - v.created_at > v.ttl]
                for k in expired:
                    del self._store[k]
            time.sleep(CLEANUP_INTERVAL_SECONDS)

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1)


# instantiate a module-level store (import-safe)
ephemeral_store = EphemeralStore()


# --- Privacy utilities ---

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\+?\d[\d\-\s]{6,}\d"),
    # Add more patterns for names, ssn, etc., as needed
}


def redact_pii(text: str, *, placeholder: str = "[REDACTED]") -> str:
    """
    Redact common PII using regex patterns. This is a conservative approach:
    replace detected tokens with a placeholder to avoid leakage to logs or external LLMs.
    """
    if text is None:
        return text
    out = text
    for _, pat in PII_PATTERNS.items():
        out = pat.sub(placeholder, out)
    return out


def enforce_no_storage_policy(content: str) -> bool:
    """
    Return True if content is allowed to be sent to external services; False if it must be blocked.
    This is a simple policy gate: e.g., block explicit requests to store PII or to 'remember' long-term.
    Extend with enterprise rules as needed.
    """
    lower = (content or "").lower()
    # simple heuristics
    if "remember this" in lower or "store my" in lower or "save my" in lower:
        return False
    return True


# --- Session / request lifecycle helpers ---

def create_session(ttl_seconds: Optional[int] = None) -> str:
    session_id = str(uuid.uuid4())
    ephemeral_store.set(session_id, {"created_at": time.time()}, ttl=ttl_seconds)
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    raw = ephemeral_store.get(session_id)
    if raw is None:
        return None
    return raw


def touch_session(session_id: str, ttl_seconds: Optional[int] = None):
    data = ephemeral_store.get(session_id)
    if data is not None:
        ephemeral_store.set(session_id, data, ttl=ttl_seconds)


def delete_session(session_id: str):
    ephemeral_store.delete(session_id)


# --- Auto-delete hooks and decorators ---

def auto_delete_after(ttl_seconds: int):
    """
    Decorator factory: attaches a session entry with the given TTL, and ensures deletion after TTL.
    Useful for functions that generate temporary artifacts.
    """
    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            session_id = kwargs.get("session_id") or create_session(ttl_seconds)
            try:
                result = fn(*args, **{**kwargs, "session_id": session_id})
                return result
            finally:
                # schedule deletion (non-blocking)
                threading.Timer(ttl_seconds, lambda: ephemeral_store.delete(session_id)).start()
        return wrapper
    return decorator


# --- Example redaction helper used before sending payloads externally ---

def prepare_payload_for_llm(user_text: str) -> dict:
    """
    Apply redaction and policy checks before dispatching content to LLM APIs.
    Returns dict with 'allowed' flag and 'payload' (redacted).
    """
    if not enforce_no_storage_policy(user_text):
        return {"allowed": False, "reason": "request_violates_no_storage_policy"}

    redacted = redact_pii(user_text)
    return {"allowed": True, "payload": redacted}


# Cleanup helper to call on shutdown
def shutdown():
    ephemeral_store.stop()
