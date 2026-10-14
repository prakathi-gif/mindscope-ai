"""
Data loading, verification, and caching module for MindScope AI.
Supports verified Hugging Face datasets, local cached Parquet, and CSV/JSON/JSONL uploads.
"""

import os
import re
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset
from config import (
    DATA_CACHE_DIR, PATH_PROCESSED_DATASET, LABEL_NAMES,
    LABEL2ID, ID2LABEL, NUM_CLASSES
)

def redact_sensitive_pii(text: str) -> str:
    """
    Applies privacy and ethical safeguards to Reddit text:
    - Redacts Reddit usernames (u/username or /u/username)
    - Redacts URLs
    - Redacts email addresses
    - Normalizes excessive whitespace
    """
    if not isinstance(text, str):
        return ""
    # Mask Reddit usernames
    text = re.sub(r'(?:/?u/)[A-Za-z0-9_-]+', '[USER_REDACTED]', text)
    # Mask URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL_REDACTED]', text)
    # Mask emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def inspect_dataset_schema(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> Dict[str, Any]:
    """Computes academic validation metrics and structural integrity of the dataset."""
    total_obs = len(df)
    missing_text = df[text_col].isna().sum() if text_col in df.columns else 0
    empty_text = (df[text_col].astype(str).str.strip() == "").sum() if text_col in df.columns else 0
    duplicates = df[text_col].duplicated().sum() if text_col in df.columns else 0
    
    label_counts = df[label_col].value_counts().to_dict() if label_col in df.columns else {}
    unique_labels = sorted(list(label_counts.keys()))
    
    char_lens = df[text_col].astype(str).str.len()
    word_lens = df[text_col].astype(str).str.split().str.len()
    
    return {
        "total_observations": total_obs,
        "text_column": text_col,
        "label_column": label_col,
        "num_classes": len(unique_labels),
        "unique_labels": unique_labels,
        "missing_text_count": int(missing_text),
        "empty_text_count": int(empty_text),
        "duplicate_text_count": int(duplicates),
        "label_distribution": label_counts,
        "char_length_stats": {
            "mean": float(char_lens.mean()),
            "median": float(char_lens.median()),
            "std": float(char_lens.std()),
            "p95": float(np.percentile(char_lens, 95)),
            "max": int(char_lens.max())
        },
        "word_length_stats": {
            "mean": float(word_lens.mean()),
            "median": float(word_lens.median()),
            "p95": float(np.percentile(word_lens, 95)),
            "max": int(word_lens.max())
        }
    }

def load_verified_reddit_dataset(
    dataset_name: str = "mavinsao/reddit-mental-illness-82",
    force_reload: bool = False,
    sample_limit: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads and validates the verified Reddit mental-health dataset.
    Returns (train_df, val_df, test_df, metadata_dict).
    Uses local cached Parquet files if available.
    """
    cache_train = DATA_CACHE_DIR / "train.parquet"
    cache_val = DATA_CACHE_DIR / "val.parquet"
    cache_test = DATA_CACHE_DIR / "test.parquet"
    
    if not force_reload and cache_train.exists() and cache_val.exists() and cache_test.exists():
        train_df = pd.read_parquet(cache_train)
        val_df = pd.read_parquet(cache_val)
        test_df = pd.read_parquet(cache_test)
        for s_df in [train_df, val_df, test_df]:
            if "label_name" not in s_df.columns and "label_id" in s_df.columns:
                s_df["label_name"] = s_df["label_id"].map(ID2LABEL)
            elif "label_id" not in s_df.columns and "label_name" in s_df.columns:
                s_df["label_id"] = s_df["label_name"].map(LABEL2ID)
    else:
        # Load from Hugging Face
        ds = load_dataset(dataset_name)
        
        # Convert splits to pandas
        raw_train = ds["train"].to_pandas()
        raw_val = ds["validation"].to_pandas()
        raw_test = ds["test"].to_pandas()
        
        # Format labels
        label_map = LABEL_NAMES
        if ds["train"].features["label"].dtype.startswith("int") or hasattr(ds["train"].features["label"], "names"):
            class_names = getattr(ds["train"].features["label"], "names", LABEL_NAMES)
            raw_train["label_name"] = raw_train["label"].map(lambda x: class_names[x] if x < len(class_names) else "unknown")
            raw_val["label_name"] = raw_val["label"].map(lambda x: class_names[x] if x < len(class_names) else "unknown")
            raw_test["label_name"] = raw_test["label"].map(lambda x: class_names[x] if x < len(class_names) else "unknown")
        else:
            raw_train["label_name"] = raw_train["label"].astype(str).str.lower()
            raw_val["label_name"] = raw_val["label"].astype(str).str.lower()
            raw_test["label_name"] = raw_test["label"].astype(str).str.lower()
            
        # Clean and redact text
        processed_splits = []
        for s_df in [raw_train, raw_val, raw_test]:
            s_df = s_df.copy()
            s_df["clean_text"] = s_df["text"].apply(redact_sensitive_pii)
            s_df.dropna(subset=["clean_text"], inplace=True)
            s_df = s_df[s_df["clean_text"].str.strip() != ""]
            s_df["label_id"] = s_df["label_name"].map(LABEL2ID)
            s_df.dropna(subset=["label_id"], inplace=True)
            s_df["label_id"] = s_df["label_id"].astype(int)
            out_df = s_df[["clean_text", "label_name", "label_id"]].rename(columns={"clean_text": "text"})
            processed_splits.append(out_df.reset_index(drop=True))
            
        train_df, val_df, test_df = processed_splits[0], processed_splits[1], processed_splits[2]
        
        # Save to cache
        train_df.to_parquet(cache_train)
        val_df.to_parquet(cache_val)
        test_df.to_parquet(cache_test)

    if sample_limit:
        train_df = train_df.groupby("label_name", group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_limit // NUM_CLASSES), random_state=42)
        ).reset_index(drop=True)
        val_df = val_df.groupby("label_name", group_keys=False).apply(
            lambda x: x.sample(min(len(x), (sample_limit // 4) // NUM_CLASSES), random_state=42)
        ).reset_index(drop=True)
        test_df = test_df.groupby("label_name", group_keys=False).apply(
            lambda x: x.sample(min(len(x), (sample_limit // 4) // NUM_CLASSES), random_state=42)
        ).reset_index(drop=True)

    metadata = {
        "dataset_name": dataset_name,
        "source": "Hugging Face Datasets",
        "task_type": "Multi-Class Mental Health Text Classification (8 Classes)",
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "total_size": len(train_df) + len(val_df) + len(test_df),
        "classes": LABEL_NAMES,
        "license": "Academic / Non-Commercial Research Only",
        "validation_metrics": inspect_dataset_schema(train_df)
    }

    return train_df, val_df, test_df, metadata
