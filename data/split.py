"""
Dataset splitting, stratification, and cross-split leakage auditor.
"""

from typing import Tuple, Dict, Any, Set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def audit_data_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str = "text") -> Dict[str, Any]:
    """
    Performs leakage detection across splits:
    - Exact duplicate text intersection
    - Split overlap counts
    - Stratification preservation
    """
    train_texts: Set[str] = set(train_df[text_col].dropna().str.lower().str.strip())
    val_texts: Set[str] = set(val_df[text_col].dropna().str.lower().str.strip())
    test_texts: Set[str] = set(test_df[text_col].dropna().str.lower().str.strip())

    train_val_overlap = len(train_texts.intersection(val_texts))
    train_test_overlap = len(train_texts.intersection(test_texts))
    val_test_overlap = len(val_texts.intersection(test_texts))
    total_leakage = train_val_overlap + train_test_overlap + val_test_overlap

    return {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "total_overlap_count": total_leakage,
        "is_leakage_free": (total_leakage == 0),
        "train_classes": train_df["label_name"].value_counts().to_dict() if "label_name" in train_df else {},
        "val_classes": val_df["label_name"].value_counts().to_dict() if "label_name" in val_df else {},
        "test_classes": test_df["label_name"].value_counts().to_dict() if "label_name" in test_df else {}
    }

def create_stratified_splits(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    test_pct: float = 0.15,
    random_state: int = 42,
    stratify_col: str = "label_id"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates guaranteed stratified splits if official splits need reconstruction or custom uploaded data is provided.
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-4, "Splits must sum to 1.0"
    
    # First split: Train vs Temp (Val + Test)
    temp_size = val_pct + test_pct
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df[stratify_col],
        random_state=random_state
    )
    
    # Second split: Val vs Test
    val_relative_ratio = val_pct / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        stratify=temp_df[stratify_col],
        random_state=random_state
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

