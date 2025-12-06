"""
api_server.py
FastAPI server exposing /summarize, /explain, /emotion, /topic, /adaptive-response.
Integrates with error_handler and privacy_module, and provides an LLM adapter abstraction.
"""

import os
import asyncio
from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import time

from error_handler import report_exception, AppError, ErrorCategory, safe_execute, ErrorSeverity
from privacy_module import prepare_payload_for_llm, create_session, get_session, delete_session, redact_pii

# --- LLM adapter abstraction ---

class LLMAdapterInterface:
    async def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class DummyLocalAdapter(LLMAdapterInterface):
    """
    Placeholder. Replace with real client for OpenAI/HuggingFace/Groq/local runtime.
    This returns deterministic dummy responses for unit testing.
    """
    async def generate(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(0.05)
        # Simple rule-based outputs for endpoints
        if "summarize" in prompt.lower():
            return "TL;DR: (dummy) This is a summary."
        if "emotion" in prompt.lower():
            return "neutral"
        if "topic" in prompt.lower():
            return "technology"
        if "explain" in prompt.lower():
            return "Explanation: (dummy) Here's a simple explanation."
        return "Adaptive response (dummy)."


# Choose adapter via env var
ADAPTER = os.getenv("LLM_ADAPTER", "dummy")
if ADAPTER == "dummy":
    llm_adapter = DummyLocalAdapter()
else:
    # Import real adapters here (OpenAIAdapter, HFAdapter, etc.)
    llm_adapter = DummyLocalAdapter()


# --- FastAPI app setup ---
app = FastAPI(title="LLM Backend API with Privacy & Error Handling")

# --- Request/Response schemas ---

class BaseRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    result: str
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None


# Utility: generate correlation id
def make_correlation_id(provided: Optional[str] = None) -> str:
    return provided or str(uuid.uuid4())


# Dependency to prepare payload and enforce privacy
async def prepare_request(req: BaseRequest):
    correlation_id = make_correlation_id(req.correlation_id)
    session_id = req.session_id or create_session()
    # Apply privacy checks
    prep = prepare_payload_for_llm(req.text)
    if not prep["allowed"]:
        raise AppError("Request violates privacy policy", category=ErrorCategory.VALIDATION)
    payload = prep["payload"]
    return {"text": payload, "session_id": session_id, "correlation_id": correlation_id, "metadata": req.metadata or {}}


# --- API endpoints ---


@app.post("/summarize", response_model=LLMResponse)
@safe_execute
async def summarize(req: BaseRequest, prepared: dict = Depends(prepare_request)):
    corr = prepared["correlation_id"]
    try:
        prompt = f"summarize:\n\n{prepared['text']}"
        raw = await llm_adapter.generate(prompt, task="summarize", correlation_id=corr)
        return LLMResponse(result=raw, session_id=prepared["session_id"], correlation_id=corr)
    except AppError:
        raise
    except Exception as e:
        report = report_exception(e, correlation_id=corr, metadata={"endpoint": "/summarize"})
        raise HTTPException(status_code=502, detail=report["error"])


@app.post("/explain", response_model=LLMResponse)
@safe_execute
async def explain(req: BaseRequest, prepared: dict = Depends(prepare_request)):
    corr = prepared["correlation_id"]
    try:
        prompt = f"explain (simple):\n\n{prepared['text']}"
        raw = await llm_adapter.generate(prompt, task="explain", correlation_id=corr)
        return LLMResponse(result=raw, session_id=prepared["session_id"], correlation_id=corr)
    except Exception as e:
        report = report_exception(e, correlation_id=corr, metadata={"endpoint": "/explain"})
        raise HTTPException(status_code=502, detail=report["error"])


@app.post("/emotion", response_model=LLMResponse)
@safe_execute
async def emotion(req: BaseRequest, prepared: dict = Depends(prepare_request)):
    corr = prepared["correlation_id"]
    try:
        prompt = f"detect emotion:\n\n{prepared['text']}"
        raw = await llm_adapter.generate(prompt, task="emotion", correlation_id=corr)
        return LLMResponse(result=raw, session_id=prepared["session_id"], correlation_id=corr)
    except Exception as e:
        report = report_exception(e, correlation_id=corr, metadata={"endpoint": "/emotion"})
        raise HTTPException(status_code=502, detail=report["error"])


@app.post("/topic", response_model=LLMResponse)
@safe_execute
async def topic(req: BaseRequest, prepared: dict = Depends(prepare_request)):
    corr = prepared["correlation_id"]
    try:
        prompt = f"extract topic:\n\n{prepared['text']}"
        raw = await llm_adapter.generate(prompt, task="topic", correlation_id=corr)
        return LLMResponse(result=raw, session_id=prepared["session_id"], correlation_id=corr)
    except Exception as e:
        report = report_exception(e, correlation_id=corr, metadata={"endpoint": "/topic"})
        raise HTTPException(status_code=502, detail=report["error"])


@app.post("/adaptive-response", response_model=LLMResponse)
@safe_execute
async def adaptive_response(req: BaseRequest, prepared: dict = Depends(prepare_request)):
    """
    Adaptive response: could return short/long/explain/emotion combined based on metadata flags.
    Example metadata: {"style":"concise","include_explanation":true}
    """
    corr = prepared["correlation_id"]
    try:
        metadata = req.metadata or {}
        mode = metadata.get("style", "balanced")
        include_expl = metadata.get("include_explanation", False)
        prompt = f"adaptive mode={mode}; include_explanation={include_expl}\n\n{prepared['text']}"
        raw = await llm_adapter.generate(prompt, task="adaptive", correlation_id=corr)
        return LLMResponse(result=raw, session_id=prepared["session_id"], correlation_id=corr)
    except Exception as e:
        report = report_exception(e, correlation_id=corr, metadata={"endpoint": "/adaptive-response"})
        raise HTTPException(status_code=502, detail=report["error"])


# --- Admin / privacy controls (safe endpoints) ---

@app.post("/session/delete")
async def api_delete_session(session_id: str):
    """
    Explicit session deletion endpoint. Caller must be authenticated in production.
    """
    try:
        delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        report = report_exception(e, metadata={"endpoint": "/session/delete"})
        raise HTTPException(status_code=500, detail=report["error"])


# --- Graceful shutdown hook to ensure ephemeral store cleanup ---
@app.on_event("shutdown")
def shutdown_event():
    try:
        from privacy_module import shutdown as privacy_shutdown
        privacy_shutdown()
    except Exception:
        pass
