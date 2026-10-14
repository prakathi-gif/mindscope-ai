"""
Page 2: Dataset Explorer.
Comprehensive dataset analysis, class imbalance metrics, length distributions, and privacy-filtered inspection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from config import CATEGORY_COLORS, LABEL_NAMES, DATA_CACHE_DIR, PATH_PROCESSED_DATASET
from data.loader import load_verified_reddit_dataset, redact_sensitive_pii, inspect_dataset_schema
from components.header import render_header, render_ethical_disclaimer
from components.metrics_cards import render_metric_card, render_research_card_start, render_research_card_end

def compute_entropy(proportions):
    """Computes Shannon entropy (bits) of class distribution."""
    props = proportions[proportions > 0]
    return -np.sum(props * np.log2(props))

def render_page():
    render_header("Dataset Explorer", "Reddit Mental Health Corpus Analysis • 52,642 Verified Posts")
    render_ethical_disclaimer()

    # Data Source Selector
    st.sidebar.markdown("### 📁 Data Source Selection")
    data_source = st.sidebar.selectbox(
        "Source Provider",
        ["Verified Hugging Face Dataset (mavinsao/reddit-mental-illness-82)", "Local Parquet Cache", "Upload Custom CSV/JSON/JSONL"]
    )

    # Load Data
    df = None
    if "Upload Custom" in data_source:
        uploaded_file = st.sidebar.file_uploader("Upload Mental Health Text File", type=["csv", "json", "jsonl"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_json(uploaded_file)
                st.sidebar.success(f"Uploaded {len(df):,} records successfully!")
            except Exception as e:
                st.sidebar.error(f"Error parsing file: {e}")
    else:
        # Load verified dataset
        try:
            train_df, val_df, test_df, meta = load_verified_reddit_dataset(sample_limit=None)
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        except Exception as e:
            st.warning(f"Connecting to dataset cache: {e}")
            # Fallback to local files if needed
            c_train = DATA_CACHE_DIR / "train.parquet"
            if c_train.exists():
                df = pd.read_parquet(c_train)

    if df is None or len(df) == 0:
        st.info("Loading Reddit mental-health dataset... Please ensure cache or data pipeline is initialized.")
        return

    # Basic statistics
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label_name" if "label_name" in df.columns else ("label" if "label" in df.columns else df.columns[1])

    # Word and char lengths
    if "word_count" not in df.columns:
        df["word_count"] = df[text_col].astype(str).str.split().str.len()
        df["char_count"] = df[text_col].astype(str).str.len()

    total_records = len(df)
    unique_classes = df[label_col].nunique()
    class_counts = df[label_col].value_counts()
    imbalance_ratio = class_counts.max() / max(1, class_counts.min())
    shannon_entropy = compute_entropy(class_counts.values / total_records)
    max_entropy = np.log2(unique_classes)
    evenness = shannon_entropy / max_entropy if max_entropy > 0 else 1.0

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Corpus Posts", f"{total_records:,}", "Verified Reddit Disclosures")
    with col2:
        render_metric_card("Categories", f"{unique_classes}", "Mutually Exclusive Classes")
    with col3:
        render_metric_card("Imbalance Ratio", f"{imbalance_ratio:.2f} : 1", f"Max: {class_counts.max()} | Min: {class_counts.min()}")
    with col4:
        render_metric_card("Shannon Evenness", f"{evenness:.3f}", f"Entropy: {shannon_entropy:.2f} bits")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Distribution Charts
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        render_research_card_start("Annotation Category Distribution", "Post counts across all 8 classes in the corpus", "📊")
        fig_bar = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            color=class_counts.index,
            color_discrete_map=CATEGORY_COLORS,
            labels={"x": "Annotation Category", "y": "Observation Count", "color": "Category"}
        )
        fig_bar.update_layout(
            paper_bgcolor="#151D2E",
            plot_bgcolor="#0B1020",
            font=dict(color="#F1F5F9", family="Inter, sans-serif"),
            showlegend=False,
            margin=dict(l=30, r=20, t=20, b=40),
            xaxis=dict(gridcolor="#293548", tickfont=dict(color="#94A3B8")),
            yaxis=dict(gridcolor="#293548", tickfont=dict(color="#94A3B8")),
            height=340
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        render_research_card_end()

    with c_right:
        render_research_card_start("Text Length Distribution (Word Count)", "Token density and document length variance", "📏")
        fig_hist = px.histogram(
            df,
            x="word_count",
            nbins=50,
            color_discrete_sequence=["#38BDF8"],
            labels={"word_count": "Words per Post"}
        )
        p50 = float(df["word_count"].median())
        p95 = float(np.percentile(df["word_count"], 95))
        fig_hist.add_vline(x=p50, line_dash="dash", line_color="#2DD4BF", annotation_text=f"Median: {p50:.0f}w", annotation_font_color="#2DD4BF")
        fig_hist.add_vline(x=p95, line_dash="dot", line_color="#F59E0B", annotation_text=f"P95: {p95:.0f}w", annotation_font_color="#F59E0B")
        
        fig_hist.update_layout(
            paper_bgcolor="#151D2E",
            plot_bgcolor="#0B1020",
            font=dict(color="#F1F5F9", family="Inter, sans-serif"),
            margin=dict(l=30, r=20, t=20, b=40),
            xaxis=dict(gridcolor="#293548", range=[0, min(1000, p95 * 1.5)], tickfont=dict(color="#94A3B8")),
            yaxis=dict(gridcolor="#293548", tickfont=dict(color="#94A3B8")),
            height=340
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        render_research_card_end()

    # Lexical Analysis: Distinctive N-Grams per Category
    render_research_card_start("Distinctive Lexical N-Grams per Category", "Top TF-IDF distinctive phrases associated with each annotation class", "🔤")
    
    selected_lex_cat = st.selectbox("Select Mental Health Category for Lexical Profile:", sorted(df[label_col].unique().tolist()))
    cat_texts = df[df[label_col] == selected_lex_cat][text_col].dropna().tolist()[:1000]

    if len(cat_texts) > 0:
        vec = TfidfVectorizer(max_features=15, stop_words="english", ngram_range=(1, 2))
        try:
            X_cat = vec.fit_transform(cat_texts)
            scores = np.asarray(X_cat.mean(axis=0)).ravel()
            top_words = pd.DataFrame({
                "N-Gram Phrase": vec.get_feature_names_out(),
                "Mean TF-IDF Weight": scores
            }).sort_values(by="Mean TF-IDF Weight", ascending=True)

            fig_lex = px.bar(
                top_words,
                x="Mean TF-IDF Weight",
                y="N-Gram Phrase",
                orientation="h",
                color_discrete_sequence=[CATEGORY_COLORS.get(selected_lex_cat, "#38BDF8")]
            )
            fig_lex.update_layout(
                paper_bgcolor="#151D2E",
                plot_bgcolor="#0B1020",
                font=dict(color="#F1F5F9", family="Inter, sans-serif"),
                margin=dict(l=40, r=20, t=20, b=30),
                xaxis=dict(gridcolor="#293548", tickfont=dict(color="#94A3B8")),
                yaxis=dict(gridcolor="#293548", tickfont=dict(color="#94A3B8")),
                height=320
            )
            st.plotly_chart(fig_lex, use_container_width=True)
        except Exception as e:
            st.info(f"Computing n-grams: {e}")
    render_research_card_end()

    # Privacy-Conscious Data Table Inspector
    render_research_card_start("Privacy-Conscious Corpus Data Viewer", "PII-redacted post text with interactive filters and search", "🔎")
    
    col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
    with col_f1:
        search_kw = st.text_input("Search Keyword / Token:", placeholder="e.g., panic, focus, medication...")
    with col_f2:
        filter_cat = st.multiselect("Filter by Categories:", sorted(df[label_col].unique().tolist()))
    with col_f3:
        max_rows = st.slider("Display Records Limit:", 10, 200, 50)

    filtered_df = df.copy()
    if search_kw:
        filtered_df = filtered_df[filtered_df[text_col].astype(str).str.contains(search_kw, case=False, na=False)]
    if filter_cat:
        filtered_df = filtered_df[filtered_df[label_col].isin(filter_cat)]

    st.markdown(f"**Showing {min(len(filtered_df), max_rows)} of {len(filtered_df):,} matching posts**")
    
    # Redact display text
    display_df = filtered_df[[label_col, text_col, "word_count"]].head(max_rows).copy()
    display_df[text_col] = display_df[text_col].apply(redact_sensitive_pii)
    display_df = display_df.rename(columns={label_col: "Category", text_col: "Redacted Text", "word_count": "Words"})

    st.dataframe(display_df, use_container_width=True, height=350)
    render_research_card_end()

if __name__ == "__main__":
    render_page()

