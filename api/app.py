import time
import asyncio
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from src.batch_inference import load_policy_and_session, score_candidates_batch
from src.pipeline import run_audit_pipeline, apply_catalog_guardrails

# ---------------------------------------------------------
# Pydantic Schemas for Request & Response Contracts
# ---------------------------------------------------------

class AnalyzeUrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="Target webpage URL to audit for deceptive UI")

class AnalyzeSnippetsRequest(BaseModel):
    snippets: List[str] = Field(..., min_length=1, description="List of raw UI text strings to evaluate")

class DeceptiveFinding(BaseModel):
    tag: Optional[str] = None
    selector: Optional[str] = None
    text: str
    prob_dark: float
    confidence: float
    verdict: str

class PipelineTimings(BaseModel):
    dom_extraction_ms: Optional[float] = None
    heuristic_pruning_ms: Optional[float] = None
    batch_inference_ms: Optional[float] = None
    total_pipeline_ms: float

class AuditSummary(BaseModel):
    total_dom_nodes: int
    pruned_candidates: int
    model_scored: int
    guardrail_suppressed: int
    flagged_count: int

class UrlAuditResponse(BaseModel):
    target_url: str
    timings: PipelineTimings
    summary: AuditSummary
    findings: List[DeceptiveFinding]

class SnippetAuditResponse(BaseModel):
    scored_count: int
    processing_time_ms: float
    results: List[DeceptiveFinding]

# ---------------------------------------------------------
# FastAPI App Initialization & CORS
# ---------------------------------------------------------

app = FastAPI(
    title="Dark Pattern Detection API",
    description="Asynchronous microservice for auditing deceptive UI/UX using Playwright, heuristic pruning, and quantized INT8 DistilRoBERTa.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Helper: Thread-isolated Playwright runner for Windows
# ---------------------------------------------------------

def _run_sync_pipeline_wrapper(target_url: str) -> Dict[str, Any]:
    """Runs Playwright in an isolated event loop to bypass Windows loop conflicts."""
    return asyncio.run(run_audit_pipeline(target_url))

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    try:
        session, _, policy = load_policy_and_session()
        return {
            "status": "healthy",
            "model_initialized": session is not None,
            "optimal_threshold": policy.get("optimal_threshold", 0.65),
            "tri_state_policy": policy.get("tri_state_policy", {})
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model loading failure: {str(exc)}")


@app.post("/analyze-url", response_model=UrlAuditResponse, tags=["Audit"])
async def analyze_url(payload: AnalyzeUrlRequest):
    try:
        # Offload Playwright pipeline to an isolated OS thread
        report = await asyncio.to_thread(_run_sync_pipeline_wrapper, str(payload.url))
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit pipeline error: {str(exc)}")


@app.post("/analyze-snippets", response_model=SnippetAuditResponse, tags=["Inference"])
def analyze_snippets(payload: AnalyzeSnippetsRequest):
    t0 = time.perf_counter()
    session, tokenizer, policy = load_policy_and_session()

    raw_candidates = [
        {"node_id": idx, "tag": "text", "selector": "raw", "text": text}
        for idx, text in enumerate(payload.snippets)
    ]

    guarded_candidates = apply_catalog_guardrails(raw_candidates)
    model_candidates = [c for c in guarded_candidates if not c.get("guardrail_override")]

    scored_items = []
    if model_candidates:
        scored_items = score_candidates_batch(model_candidates, session, tokenizer, policy, batch_size=16)

    for c in guarded_candidates:
        if c.get("guardrail_override"):
            scored_items.append({
                **c,
                "prob_dark": 0.0001,
                "confidence": 0.9999,
                "verdict": "SAFE_CATALOG_OVERRIDE"
            })

    scored_items.sort(key=lambda x: x["node_id"])

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    findings = [
        DeceptiveFinding(
            tag=item.get("tag"),
            selector=item.get("selector"),
            text=item.get("text", ""),
            prob_dark=round(item.get("prob_dark", 0.0), 4),
            confidence=round(item.get("confidence", 0.0), 4),
            verdict=item.get("verdict", "BENIGN")
        )
        for item in scored_items
    ]

    return SnippetAuditResponse(
        scored_count=len(findings),
        processing_time_ms=elapsed_ms,
        results=findings
    )