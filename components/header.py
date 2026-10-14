"""
Header and branding component for MindScope AI.
Features an inline SVG geometric logo, system status indicator, and academic metadata.
"""

import streamlit as st
from config import APP_NAME, APP_SUBTITLE, APP_TAGLINE, APP_VERSION

def render_svg_logo(size: int = 42) -> str:
    """Generates a modern geometric SVG logo representing neural networks and analytical insight."""
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="48" height="48" rx="10" fill="#151D2E" stroke="#293548" stroke-width="1.5"/>
        <!-- Hexagonal lattice -->
        <polygon points="24,10 36,17 36,31 24,38 12,31 12,17" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="2 2" fill="none" opacity="0.6"/>
        <!-- Central Prism Core -->
        <circle cx="24" cy="24" r="5" fill="#2DD4BF" opacity="0.9"/>
        <!-- Nodes -->
        <circle cx="24" cy="12" r="3" fill="#38BDF8"/>
        <circle cx="34.4" cy="18" r="3" fill="#38BDF8"/>
        <circle cx="34.4" cy="30" r="3" fill="#38BDF8"/>
        <circle cx="24" cy="36" r="3" fill="#38BDF8"/>
        <circle cx="13.6" cy="30" r="3" fill="#38BDF8"/>
        <circle cx="13.6" cy="18" r="3" fill="#38BDF8"/>
        <!-- Synaptic connections -->
        <line x1="24" y1="24" x2="24" y2="12" stroke="#38BDF8" stroke-width="1.2" opacity="0.8"/>
        <line x1="24" y1="24" x2="34.4" y2="18" stroke="#38BDF8" stroke-width="1.2" opacity="0.8"/>
        <line x1="24" y1="24" x2="34.4" y2="30" stroke="#38BDF8" stroke-width="1.2" opacity="0.8"/>
        <line x1="24" y1="24" x2="24" y2="36" stroke="#38BDF8" stroke-width="1.2" opacity="0.8"/>
        <line x1="24" y1="24" x2="13.6" y2="30" stroke="#38BDF8" stroke-width="1.2" opacity="0.8"/>
        <line x1="24" y1="24" x2="13.6" y2="18" stroke="#38BDF8" stroke-width="1.2" opacity="0.8"/>
    </svg>
    """

def render_header(current_page: str = "Overview", status_text: str = "System Online • Research Pipeline Active"):
    """Renders the top branding header with logo, title, and live status badge."""
    logo_svg = render_svg_logo(48)
    
    header_html = f"""
    <div class="brand-header">
        <div>{logo_svg}</div>
        <div style="flex-grow: 1;">
            <div style="display: flex; align-items: baseline; justify-content: space-between;">
                <h1 class="brand-title">{APP_NAME} <span style="font-size: 14px; color: #64748B; font-weight: 400;">{APP_VERSION}</span></h1>
                <div class="system-status-badge">
                    <span class="status-dot"></span>
                    <span>{status_text}</span>
                </div>
            </div>
            <div class="brand-subtitle">{APP_SUBTITLE} — <span style="color: #94A3B8;">{current_page}</span></div>
            <div class="brand-tagline">{APP_TAGLINE}</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_ethical_disclaimer():
    """Renders mandatory academic & ethical disclaimer as required by prompt guidelines."""
    disclaimer_html = """
    <div class="disclaimer-banner">
        <strong>⚠️ Academic Research & Ethical Safeguard Notice:</strong>
        This software is strictly designed for aggregate NLP linguistic analysis and computational research.
        It is <strong>not</strong> a medical device, diagnostic instrument, or clinical decision support system.
        Predictions reflect machine learning pattern associations across social media corpora and must not be interpreted as psychiatric evaluations or used for triage/emergency response.
    </div>
    """
    st.markdown(disclaimer_html, unsafe_allow_html=True)

