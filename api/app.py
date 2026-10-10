import sys
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer

from src.batch_inference import load_policy_and_session, score_candidates_batch
from src.pipeline import run_audit_pipeline, apply_catalog_guardrails

# ---------------------------------------------------------------------------
# 1. Lifespan Manager (Preload ONNX Graph & Tokenizer into RAM)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[Lifespan Startup] Preloading quantized ONNX model and tokenizer...")
    session, tokenizer, policy = load_policy_and_session()
    
    app.state.session = session
    app.state.tokenizer = tokenizer
    app.state.policy = policy
    print("[Lifespan Startup] Model cached in memory. Ready for zero-cold-start inference.\n")
    
    yield
    
    print("\n[Lifespan Shutdown] Releasing runtime resources...")
    app.state.session = None
    app.state.tokenizer = None
    app.state.policy = None


app = FastAPI(
    title="Dark Pattern Detection Engine API",
    description="Asynchronous microservice for deceptive UI detection via cached INT8 ONNX and isolated Playwright auditing.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 2. Pydantic Schemas (Aligned with src/pipeline.py)
# ---------------------------------------------------------------------------
class SnippetRequest(BaseModel):
    snippets: List[str]

class DeceptiveFinding(BaseModel):
    tag: Optional[str] = None
    selector: Optional[str] = None
    text: str
    prob_dark: float
    confidence: float
    verdict: str

class SnippetResponse(BaseModel):
    total_evaluated: int
    findings: List[DeceptiveFinding]

class AnalyzeUrlRequest(BaseModel):
    url: HttpUrl

class TimingMetrics(BaseModel):
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
    timings: TimingMetrics
    summary: AuditSummary
    findings: List[DeceptiveFinding]


# ---------------------------------------------------------------------------
# 3. Synchronous Thread Helpers
# ---------------------------------------------------------------------------
def _score_cached_snippets(session, tokenizer, policy, snippets: List[str]):
    raw_candidates = [
        {"node_id": idx, "tag": "text", "selector": "raw", "text": text}
        for idx, text in enumerate(snippets)
    ]
    
    guarded = apply_catalog_guardrails(raw_candidates)
    
    model_candidates = [c for c in guarded if not c.get("guardrail_override")]
    override_candidates = [c for c in guarded if c.get("guardrail_override")]
    
    scored_items = []
    if model_candidates:
        scored_items = score_candidates_batch(
            candidates=model_candidates,
            session=session,
            tokenizer=tokenizer,
            policy=policy,
            batch_size=16,
        )
        
    all_results = scored_items + override_candidates
    all_results.sort(key=lambda x: x["node_id"])
    return all_results

def _run_sync_pipeline_wrapper(target_url: str):
    return asyncio.run(run_audit_pipeline(target_url))


# ---------------------------------------------------------------------------
# 4. REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check(request: Request):
    is_ready = bool(
        getattr(request.app.state, "session", None) is not None
        and getattr(request.app.state, "tokenizer", None) is not None
    )
    policy = getattr(request.app.state, "policy", {})
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_initialized": is_ready,
        "optimal_threshold": policy.get("optimal_threshold", 0.65),
        "tri_state_policy": policy.get("tri_state_policy", {}),
    }


@app.post("/analyze-snippets", response_model=SnippetResponse, tags=["Inference"])
async def analyze_snippets(payload: SnippetRequest, request: Request):
    if not payload.snippets:
        raise HTTPException(status_code=400, detail="Snippets list cannot be empty.")
    
    session = request.app.state.session
    tokenizer = request.app.state.tokenizer
    policy = request.app.state.policy

    scored = await asyncio.to_thread(
        _score_cached_snippets, session, tokenizer, policy, payload.snippets
    )

    findings = [
        DeceptiveFinding(
            tag=item.get("tag"),
            selector=item.get("selector"),
            text=item.get("text", ""),
            prob_dark=round(item.get("prob_dark", 0.0), 4),
            confidence=round(item.get("confidence", 0.0), 4),
            verdict=item.get("verdict", "BENIGN"),
        )
        for item in scored
    ]
    return SnippetResponse(total_evaluated=len(findings), findings=findings)


@app.post("/analyze-url", response_model=UrlAuditResponse, tags=["Audit"])
async def analyze_url(payload: AnalyzeUrlRequest):
    try:
        report = await asyncio.to_thread(_run_sync_pipeline_wrapper, str(payload.url))
        return report
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit pipeline error: {str(exc)}")