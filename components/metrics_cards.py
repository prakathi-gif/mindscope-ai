"""
Reusable UI cards and metric visualization components.
"""

import streamlit as st
from typing import Optional, Dict, Any

def render_metric_card(label: str, value: str, subtext: Optional[str] = None, delta_text: Optional[str] = None):
    """Renders a single metric box."""
    sub_html = f'<div class="metric-desc">{subtext}</div>' if subtext else ""
    delta_html = f'<span style="font-size: 11px; color: #38BDF8; margin-left: 6px;">{delta_text}</span>' if delta_text else ""
    card_html = f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-val">{value} {delta_html}</div>
        {sub_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_research_card_start(title: str, subtitle: Optional[str] = None, icon: str = "🔬"):
    """Starts a research container card."""
    sub_html = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    card_html = f"""
    <div class="research-card">
        <div class="card-title"><span>{icon}</span> {title}</div>
        {sub_html}
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_research_card_end():
    """Closes a research container card."""
    st.markdown("</div>", unsafe_allow_html=True)

def render_token_pill(token: str, score: float, threshold: float = 0.05) -> str:
    """Renders an attribution pill with styling corresponding to positive or negative contribution."""
    clean_token = token.replace("Ġ", " ").replace(" ", " ")
    if score > threshold:
        cls = "token-pos"
        sign = "+"
    elif score < -threshold:
        cls = "token-neg"
        sign = ""
    else:
        cls = "token-neu"
        sign = ""
    return f'<span class="token-badge {cls}" title="Attribution: {score:+.4f}">{clean_token} <small>({sign}{score:.2f})</small></span>'

