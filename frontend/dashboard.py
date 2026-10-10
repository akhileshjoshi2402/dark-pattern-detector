import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Dark Pattern Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .badge-dark {
        background-color: #ffebee;
        color: #c62828;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-review {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-safe {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar: System Health & Threshold Policy ---
with st.sidebar:
    st.title("🛡️ Engine Telemetry")
    st.caption("Backend Status & Model Calibration")
    
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success("API Status: Online")
            st.write(f"**Model Ready:** {health_data.get('model_initialized', False)}")
            st.write(f"**Calibrated Tau (τ*):** `{health_data.get('optimal_threshold', 0.65)}`")
            
            tri = health_data.get("tri_state_policy", {})
            st.markdown(f"""
            **Tri-State Decision Tiers:**
            - **Safe:** `p < {tri.get('tau_low', 0.30)}`
            - **Review:** `{tri.get('tau_low', 0.30)} ≤ p < {tri.get('tau_high', 0.70)}`
            - **Deceptive:** `p ≥ {tri.get('tau_high', 0.70)}`
            """)
        else:
            st.error(f"API Degraded (Status {health_resp.status_code})")
    except requests.exceptions.RequestException:
        st.error("API Offline: Ensure `uvicorn api.app:app` is running on port 8000.")


# --- Main Dashboard ---
st.title("Deceptive UI/UX Detection Engine")
st.markdown("Live inspection tool targeting manipulative copy, forced urgency, and confirmshaming via quantized INT8 DistilRoBERTa.")

tabs = st.tabs(["🌐 Live URL Audit", "✍️ Ad-Hoc Snippet Inspector"])


# ==============================================================================
# TAB 1: Live Webpage Audit
# ==============================================================================
with tabs[0]:
    st.subheader("Automated Headless Webpage Audit")
    target_url = st.text_input("Enter Webpage URL:", value="http://books.toscrape.com/")
    
    col_run, _ = st.columns([1, 4])
    with col_run:
        run_audit = st.button("Run Live Audit", type="primary", use_container_width=True)

    if run_audit:
        if not target_url.startswith("http://") and not target_url.startswith("https://"):
            st.warning("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Executing Playwright scraping, DOM pruning, and ONNX batch scoring..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/analyze-url",
                        json={"url": target_url},
                        timeout=90
                    )
                    if res.status_code == 200:
                        report = res.json()
                        st.success(f"Audit completed for: {report.get('target_url')}")
                        
                        # Pipeline Metrics Row
                        timings = report.get("timings", {})
                        summary = report.get("summary", {})
                        
                        # Around lines 105-110 in frontend/dashboard.py:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Latency", f"{timings.get('total_pipeline_ms', 0):.0f} ms")
                        m2.metric("DOM Elements Extracted", summary.get("total_dom_nodes", 0))
                        m3.metric("Scored by Model", summary.get("model_scored", 0))
                        m4.metric("Deceptive Elements Flagged", summary.get("flagged_count", 0))

                        # Timings Breakdown Expander
                        with st.expander("⏱️ Pipeline Latency Telemetry"):
                            st.write({
                                "Headless DOM Scraping": f"{timings.get('dom_extraction_ms', 0):.2f} ms",
                                "Heuristic Pruning": f"{timings.get('heuristic_pruning_ms', 0):.2f} ms",
                                "Batch ONNX Inference": f"{timings.get('batch_inference_ms', 0):.2f} ms",
                                "Catalog Guardrails Suppressed": summary.get("guardrail_suppressed", 0)
                            })

                        # Findings Table
                        st.markdown("### Confirmed Deceptive Elements")
                        findings = report.get("findings", [])
                        if findings:
                            for idx, f in enumerate(findings, 1):
                                st.markdown(f"""
                                <div class="metric-card" style="margin-bottom: 12px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <span class="badge-dark">{f.get('verdict')}</span>
                                        <span><b>P(Dark):</b> {f.get('prob_dark', 0.0):.4f} | <b>Confidence:</b> {f.get('confidence', 0.0):.4f}</span>
                                    </div>
                                    <p style="margin: 8px 0 4px 0; font-size: 1.1em;"><b>Text:</b> "{f.get('text')}"</p>
                                    <small style="color: #6c757d;">Tag: <code>&lt;{f.get('tag')}&gt;</code> | Selector: <code>{f.get('selector')}</code></small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No dark patterns detected on the inspected page elements.")
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as exc:
                    st.error(f"Failed to communicate with API server: {str(exc)}")


# ==============================================================================
# TAB 2: Ad-Hoc Snippet Inspector
# ==============================================================================
with tabs[1]:
    st.subheader("Manual Copy & Snippet Inspection")
    default_text = "Hurry! Only 1 item left in stock!\nIn stock. Delivered by tomorrow.\nNo thanks, I prefer paying full price\nFree shipping on orders over £25"
    snippet_input = st.text_area("Enter UI text strings (one per line):", value=default_text, height=140)

    if st.button("Evaluate Snippets", type="primary"):
        lines = [line.strip() for line in snippet_input.split("\n") if line.strip()]
        if not lines:
            st.warning("Please provide at least one text snippet to evaluate.")
        else:
            with st.spinner("Scoring candidate batch..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/analyze-snippets",
                        json={"snippets": lines},
                        timeout=10
                    )
                    if res.status_code == 200:
                        results = res.json().get("findings", [])
                        
                        data = []
                        for r in results:
                            data.append({
                                "Snippet": r.get("text"),
                                "Verdict": r.get("verdict"),
                                "Probability (Dark)": f"{r.get('prob_dark'):.4f}",
                                "Confidence": f"{r.get('confidence'):.4f}"
                            })
                        
                        st.markdown("### Evaluation Results")
                        df = pd.DataFrame(data)
                        
                        # Color coding display
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as exc:
                    st.error(f"Failed to connect to API: {str(exc)}")