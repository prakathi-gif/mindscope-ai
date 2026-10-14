"""
Configuration and constants for MindScope AI.
Defines color themes, paths, model specifications, label mapping, and experiment settings.
"""

from pathlib import Path
import os

# Project root
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
STYLES_DIR = BASE_DIR / "styles"
DATA_CACHE_DIR = ARTIFACTS_DIR / "data_cache"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# App Metadata
APP_NAME = "MindScope AI"
APP_SUBTITLE = "Explainable Mental Health Text Analytics"
APP_TAGLINE = "Understanding Language Patterns Through Responsible AI"
APP_VERSION = "v1.0.0-academic"

# Dark Theme Design System Colors (Hex)
THEME_COLORS = {
    "bg_main": "#0B1020",
    "bg_sidebar": "#111827",
    "bg_card_primary": "#151D2E",
    "bg_card_secondary": "#1B2638",
    "accent_primary": "#38BDF8",     # Cyan / Sky
    "accent_secondary": "#2DD4BF",   # Mint / Teal
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "border": "#293548",
    "status_positive": "#22C55E",
    "status_warning": "#F59E0B",
    "status_error": "#F87171",
}

# Distinct, Colorblind-Safe Category Palette
CATEGORY_COLORS = {
    "adhd": "#38BDF8",        # Sky Blue
    "anxiety": "#F59E0B",     # Amber
    "bipolar": "#A855F7",     # Violet
    "bpd": "#EC4899",         # Pink
    "depression": "#6366F1",  # Indigo
    "none": "#64748B",        # Slate Gray (Control)
    "ocd": "#14B8A6",         # Teal
    "ptsd": "#F97316",        # Orange
}

# Master Label Mapping for Reddit Mental Illness Dataset
# mavinsao/reddit-mental-illness-82 has 8 classes (7 conditions + 1 control)
LABEL_NAMES = ["adhd", "anxiety", "bipolar", "bpd", "depression", "none", "ocd", "ptsd"]
NUM_CLASSES = len(LABEL_NAMES)
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

# Clinical & Research Descriptions for Categories
CATEGORY_DESCRIPTIONS = {
    "adhd": "Attention-Deficit/Hyperactivity Disorder: Disclosures related to focus, executive function, dopamine seeking, and hyperactivity.",
    "anxiety": "Generalized & Social Anxiety: Disclosures of excessive worry, panic attacks, somatic anxiety, and apprehension.",
    "bipolar": "Bipolar Affective Spectrum: Expressions of cyclical mood shifts, manic/hypomanic episodes, and depressive lows.",
    "bpd": "Borderline Personality Features: Discussions of emotional dysregulation, fear of abandonment, splitting, and intense relationships.",
    "depression": "Major Depressive Expressions: Expressions of anhedonia, fatigue, pervasive despair, hopelessness, and psychomotor slowing.",
    "none": "Control Baseline: Non-clinical informal Reddit discourse with no overt mental health condition focus.",
    "ocd": "Obsessive-Compulsive Patterns: Intrusive thoughts, contamination obsessions, repetitive checking, and ritualistic compulsions.",
    "ptsd": "Post-Traumatic Stress: Intrusive trauma recollections, hypervigilance, emotional numbing, and avoidance triggers."
}

# Transformer Checkpoints
ROBERTA_CHECKPOINT = "roberta-base"
# MentalBERT / MentalRoBERTa verified public checkpoints
MENTAL_ROBERTA_CHECKPOINT = "mental/mental-roberta-base"
MENTAL_BERT_CHECKPOINT = "mental/mental-bert-base-uncased"

# Default Model Paths
PATH_LR_MODEL = ARTIFACTS_DIR / "model_lr.joblib"
PATH_TFIDF_VECTORIZER = ARTIFACTS_DIR / "vectorizer_tfidf.joblib"
PATH_XGB_MODEL = ARTIFACTS_DIR / "model_xgb.joblib"
PATH_ROBERTA_WEIGHTS = ARTIFACTS_DIR / "model_roberta.pt"
PATH_ENSEMBLE_CONFIG = ARTIFACTS_DIR / "ensemble_config.json"
PATH_EVALUATION_METRICS = ARTIFACTS_DIR / "evaluation_metrics.json"
PATH_UMAP_EMBEDDINGS = ARTIFACTS_DIR / "umap_embeddings.parquet"
PATH_TEST_PREDICTIONS = ARTIFACTS_DIR / "test_predictions.parquet"
PATH_PROCESSED_DATASET = DATA_CACHE_DIR / "processed_reddit_mental_health.parquet"

