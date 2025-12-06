"""
error_handler.py
Centralized error handling, classification (technical vs semantic), structured logging,
and integration hooks for external error reporting (Sentry, etc).
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, Optional, Tuple
from enum import Enum
import time
import uuid

logger = logging.getLogger("app.error_handler")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(correlation_id)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    TECHNICAL = "technical"   # system exceptions, infra, timeouts, dependencies
    SEMANTIC = "semantic"     # user input ambiguous, nonsensical, hallucination, inconsistent
    VALIDATION = "validation" # client input validation
    AUTH = "auth"             # authentication/authorization


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.TECHNICAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.correlation_id = self.metadata.get("correlation_id") or str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }


# --- Error classification helpers ---

SEMANTIC_HINTS = [
    "unclear",
    "ambiguous",
    "contradiction",
    "nonsensical",
    "can't determine",
    "unknown",
    "hallucination",
]


def classify_semantic_vs_technical(error_message: str, context: Optional[str] = None) -> ErrorCategory:
    """
    Lightweight heuristic classifier. Returns ErrorCategory.SEMANTIC if
    we detect clues that the problem arises from ambiguous or nonsensical input.
    Otherwise defaults to TECHNICAL.
    NOTE: replace or extend with an ML-based classifier if you need high accuracy.
    """
    text = (error_message + " " + (context or "")).lower()
    for hint in SEMANTIC_HINTS:
        if hint in text:
            return ErrorCategory.SEMANTIC
    # common technical phrases
    if any(kw in text for kw in ("traceback", "exception", "timeout", "connection refused", "500", "503", "error code")):
        return ErrorCategory.TECHNICAL
    # fallback
    return ErrorCategory.TECHNICAL


# --- Logging & reporting API ---


def _log_structured(level: ErrorSeverity, msg: str, correlation_id: Optional[str], **extra):
    # Ensure correlation_id present for tracing
    extra_record = {"correlation_id": correlation_id or str(uuid.uuid4())}
    extra_record.update(extra)
    # use logger.bind-like pattern by injecting into record via adapter
    adapter = logging.LoggerAdapter(logger, {"correlation_id": extra_record["correlation_id"]})
    if level == ErrorSeverity.INFO:
        adapter.info(msg, extra=extra_record)
    elif level == ErrorSeverity.WARNING:
        adapter.warning(msg, extra=extra_record)
    elif level == ErrorSeverity.ERROR:
        adapter.error(msg, extra=extra_record)
    elif level == ErrorSeverity.CRITICAL:
        adapter.critical(msg, extra=extra_record)


def report_exception(exc: Exception, *, correlation_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """
    Central place to send exceptions to logs and optional external services.
    Keep this lightweight to avoid masking root cause.
    """
    if isinstance(exc, AppError):
        category = exc.category
        severity = exc.severity
        msg = exc.message
        metadata = {**(metadata or {}), **exc.metadata}
        cid = exc.correlation_id
    else:
        msg = str(exc)
        category = classify_semantic_vs_technical(msg)
        severity = ErrorSeverity.ERROR
        metadata = metadata or {}
        cid = correlation_id or str(uuid.uuid4())

    # Minimal traceback capture (do not include user data)
    tb = traceback.format_exc()
    metadata_summary = {"metadata_keys": list(metadata.keys())} if metadata else {}

    _log_structured(severity, f"{category.value.upper()} - {msg} | tb-snippet: {tb.splitlines()[-1]}", cid, **metadata_summary)

    # Optional: send to Sentry / external service
    # try:
    #     import sentry_sdk
    #     sentry_sdk.capture_exception(exc)
    # except Exception:
    #     pass

    # Return structured payload for API responses
    return {
        "error": {
            "message": "internal_error" if category == ErrorCategory.TECHNICAL else msg,
            "category": category.value,
            "severity": severity.value,
            "correlation_id": cid,
        }
    }


# Decorator for catching errors in async/sync handlers
def safe_execute(func):
    if asyncio.iscoroutinefunction(func):
        async def wrapper(*args, **kwargs):
            cid = kwargs.get("correlation_id") or str(uuid.uuid4())
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                report = report_exception(exc, correlation_id=cid, metadata={"fn": func.__name__})
                # Re-raise an AppError to be handled by the framework, or return the report
                raise AppError("A server error occurred", category=ErrorCategory.TECHNICAL, metadata={"correlation_id": report["error"]["correlation_id"]})
        return wrapper
    else:
        def wrapper(*args, **kwargs):
            cid = kwargs.get("correlation_id") or str(uuid.uuid4())
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                report = report_exception(exc, correlation_id=cid, metadata={"fn": func.__name__})
                raise AppError("A server error occurred", category=ErrorCategory.TECHNICAL, metadata={"correlation_id": report["error"]["correlation_id"]})
        return wrapper
