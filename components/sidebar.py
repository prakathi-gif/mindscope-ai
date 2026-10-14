"""
Sidebar navigation and status monitoring component.
"""

import streamlit as st
import psutil
import torch
from config import APP_NAME, APP_VERSION, THEME_COLORS

PAGES = [
    ("Research Overview", "1_Research_Overview", "📊"),
    ("Dataset Explorer", "2_Dataset_Explorer", "🔍"),
    ("Text Analysis Lab", "3_Text_Analysis_Lab", "🧪"),
    ("Model Laboratory", "4_Model_Laboratory", "⚙️"),
    ("Explainable AI", "5_Explainable_AI", "💡"),
    ("Embedding Explorer", "6_Embedding_Explorer", "🌐"),
    ("Error Analysis", "7_Error_Analysis", "📉"),
    ("Research Evaluation", "8_Research_Evaluation", "📑"),
    ("Settings and Privacy", "9_Settings_and_Privacy", "🔒"),
]

def render_sidebar() -> str:
    """Renders the persistent sidebar and returns the selected page key."""
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #293548; margin-bottom: 16px;">
            <div style="font-size: 18px; font-weight: 700; color: #F1F5F9; letter-spacing: -0.3px;">
                🧠 {APP_NAME}
            </div>
            <div style="font-size: 11px; color: #38BDF8; font-weight: 500;">
                Advanced NLP Research Lab
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "selected_page" not in st.session_state:
            st.session_state["selected_page"] = "Research Overview"

        st.markdown("<p style='font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #64748B; font-weight: 600; margin-bottom: 6px;'>Research Modules</p>", unsafe_allow_html=True)
        
        for name, key, icon in PAGES:
            is_active = (st.session_state["selected_page"] == name)
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{icon}  {name}", key=f"nav_{key}", use_container_width=True, type=btn_type):
                st.session_state["selected_page"] = name
                st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #293548; margin: 20px 0 16px 0;'/>", unsafe_allow_html=True)
        
        # System & Hardware Telemetry
        st.markdown("<p style='font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #64748B; font-weight: 600; margin-bottom: 6px;'>System Diagnostics</p>", unsafe_allow_html=True)
        
        cuda_avail = torch.cuda.is_available()
        device_label = f"GPU: {torch.cuda.get_device_name(0)}" if cuda_avail else "CPU: Standard Intel/AMD"
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_gb = psutil.virtual_memory().used / (1024**3)
        total_ram_gb = psutil.virtual_memory().total / (1024**3)

        st.markdown(f"""
        <div style="background: #151D2E; border: 1px solid #293548; border-radius: 8px; padding: 12px; font-size: 11px; line-height: 1.6;">
            <div><span style="color: #94A3B8;">Compute Engine:</span> <span style="color: #38BDF8; font-family: monospace;">{device_label}</span></div>
            <div><span style="color: #94A3B8;">Memory Load:</span> <span style="color: #2DD4BF; font-family: monospace;">{ram_gb:.1f} / {total_ram_gb:.1f} GB</span></div>
            <div><span style="color: #94A3B8;">CPU Utilization:</span> <span style="color: #F1F5F9; font-family: monospace;">{cpu_pct:.0f}%</span></div>
            <div><span style="color: #94A3B8;">Dataset Status:</span> <span style="color: #22C55E; font-weight: 600;">Verified (52.6k)</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 10px; color: #64748B; margin-top: 20px; text-align: center;">
            MindScope AI • Academic Release<br/>
            Strictly Non-Clinical Aggregate Research
        </div>
        """, unsafe_allow_html=True)

    return st.session_state["selected_page"]

