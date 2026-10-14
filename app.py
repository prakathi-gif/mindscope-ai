"""
MindScope AI — An Explainable Transformer–Ensemble Framework for Mental Health-Related Text Classification.
Main Application Entrypoint & Streamlit Page Router.
"""

import streamlit as st
from pathlib import Path
from config import BASE_DIR, STYLES_DIR, APP_NAME, APP_SUBTITLE
from components.sidebar import render_sidebar

# Configure Streamlit page settings
st.set_page_config(
    page_title=f"{APP_NAME} — {APP_SUBTITLE}",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Scientific Dark Theme CSS
def inject_custom_styles():
    css_path = STYLES_DIR / "main.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

inject_custom_styles()

# Render Navigation Sidebar
active_page = render_sidebar()

# Page Routing Engine via importlib
import importlib

PAGE_MODULE_MAP = {
    "Research Overview": "pages.1_Research_Overview",
    "Dataset Explorer": "pages.2_Dataset_Explorer",
    "Text Analysis Lab": "pages.3_Text_Analysis_Lab",
    "Model Laboratory": "pages.4_Model_Laboratory",
    "Explainable AI": "pages.5_Explainable_AI",
    "Embedding Explorer": "pages.6_Embedding_Explorer",
    "Error Analysis": "pages.7_Error_Analysis",
    "Research Evaluation": "pages.8_Research_Evaluation",
    "Settings and Privacy": "pages.9_Settings_and_Privacy",
}

module_path = PAGE_MODULE_MAP.get(active_page, "pages.1_Research_Overview")
page_module = importlib.import_module(module_path)

# Execute Page Rendering
page_module.render_page()
