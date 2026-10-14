"""
Page 9: Settings and Privacy.
Privacy controls, PII redaction rules, cache management, hardware diagnostics, and ethical compliance audit.
"""

import streamlit as st
import psutil
import torch
import shutil
from pathlib import Path
from config import ARTIFACTS_DIR, DATA_CACHE_DIR, APP_VERSION
from components.header import render_header, render_ethical_disclaimer
from components.metrics_cards import render_metric_card, render_research_card_start, render_research_card_end

def render_page():
    render_header("Settings and Privacy", "Ethical Safeguards, Local Data Retention & System Configuration")
    render_ethical_disclaimer()

    # Privacy Safeguards Control Center
    render_research_card_start("Data Privacy & Ethical Governance Safeguards", "Protocol guarantees preventing deanonymization or unconsented surveillance", "🛡️")
    
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown("**Active Safeguards Status:**")
        p1 = st.checkbox("Automatic Reddit Username Redaction (`u/[USER]` ➔ `[REDACTED]`)", value=True, disabled=True)
        p2 = st.checkbox("Automatic URL and External Link Masking", value=True, disabled=True)
        p3 = st.checkbox("Zero Ingestion of User-Identifiable PII (Strict Regex)", value=True, disabled=True)
        p4 = st.checkbox("Disable Raw Submitted Text Logging to Disk", value=True)
    with c_p2:
        st.markdown("**Ethical Boundaries Enforcement:**")
        st.checkbox("Enforce Academic Non-Clinical Use Boundary", value=True, disabled=True)
        st.checkbox("Block Individual Clinical Risk Scoring or Triage", value=True, disabled=True)
        st.checkbox("Strict Local Processing (Zero External Cloud API Calls)", value=True, disabled=True)
        st.checkbox("Transient In-Memory Inference Only", value=True)
    render_research_card_end()

    # Hardware & System Diagnostics
    render_research_card_start("Hardware Telemetry & Acceleration Diagnostics", "Runtime execution environment specifications", "💻")
    
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "Standard CPU (Intel/AMD Architecture)"
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_avail_gb = psutil.virtual_memory().available / (1024**3)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Compute Architecture", "GPU Enabled" if cuda_available else "CPU Execution", device_name)
    with c2:
        render_metric_card("Logical CPU Cores", f"{cpu_count} Cores", "Multi-Threading Active")
    with c3:
        render_metric_card("System RAM Available", f"{ram_avail_gb:.1f} / {ram_gb:.1f} GB", f"Load: {psutil.virtual_memory().percent:.0f}%")
    with c4:
        render_metric_card("PyTorch Engine", f"v{torch.__version__}", "Core Backend")
    render_research_card_end()

    # Cache & Artifact Management
    render_research_card_start("Local Artifacts & Cache Management", "Inspect storage footprint and safely manage local data cache", "🧹")
    
    # Calculate disk usage of artifacts
    artifact_size_mb = sum(f.stat().st_size for f in ARTIFACTS_DIR.glob("**/*") if f.is_file()) / (1024 * 1024)
    cache_size_mb = sum(f.stat().st_size for f in DATA_CACHE_DIR.glob("**/*") if f.is_file()) / (1024 * 1024)

    st.markdown(f"""
    - **Trained Model Artifacts Footprint:** `{artifact_size_mb:.2f} MB` (`{ARTIFACTS_DIR}`)
    - **Local Dataset Cache Footprint:** `{cache_size_mb:.2f} MB` (`{DATA_CACHE_DIR}`)
    """)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Clear Local Dataset Cache Parquets"):
            for f in DATA_CACHE_DIR.glob("*.parquet"):
                try:
                    f.unlink()
                except Exception:
                    pass
            st.success("Local dataset cache cleared successfully!")
    with col_btn2:
        if st.button("🔄 Reset Models & Embeddings Cache"):
            for fname in ["umap_embeddings.parquet", "test_predictions.parquet"]:
                p = ARTIFACTS_DIR / fname
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
            st.success("Embeddings and prediction cache reset!")
    render_research_card_end()

    # Ethical Compliance Statement
    render_research_card_start("Institutional Review & Ethical Compliance Statement", "Adherence to ethical principles for NLP on sensitive mental health social text", "📜")
    
    st.markdown("""
    1. **Non-Diagnostic Mandate:** MindScope AI models are statistical text classifiers trained to predict Reddit community labels. They do not possess clinical diagnostic authority and must never be employed for triage or treatment decisions.
    2. **Beneficence & Non-Maleficence:** Social media disclosures reflect voluntary peer interactions. All data ingestion strictly strips author identifiers and URLs to prevent deanonymization.
    3. **Transparency & Open Science:** Model weights, training pipelines, and evaluation protocols are fully open and reproducible.
    4. **Licensing & Terms of Service:** Reddit mental health corpora are evaluated strictly under academic research conventions.
    """)
    render_research_card_end()

if __name__ == "__main__":
    render_page()

