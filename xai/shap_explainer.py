"""
Classical Model Explainability Module (SHAP for XGBoost and Logistic Regression).
Computes token feature contributions with positive and negative polarities.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import shap
from config import LABEL_NAMES

class ClassicalSHAPExplainer:
    """SHAP Explainer for sparse TF-IDF text models."""

    def __init__(self, model, preprocessor, model_type: str = "xgboost"):
        self.model = model
        self.preprocessor = preprocessor
        self.model_type = model_type
        self.feature_names = np.array(self.preprocessor.get_feature_names())
        
        # Build explainer
        if model_type == "xgboost":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Linear model explainer with background sample
            self.explainer = None

    def explain_instance(
        self,
        text: str,
        target_class_idx: int,
        top_k: int = 15
    ) -> Dict[str, Any]:
        """
        Calculates feature attributions for a single document text on target class.
        Returns top positive and negative contributing tokens.
        """
        X_sparse = self.preprocessor.transform([text])
        
        if self.model_type == "xgboost":
            shap_values = self.explainer.shap_values(X_sparse)
            # shap_values shape: (1, n_features, n_classes) or list of (1, n_features)
            if isinstance(shap_values, list):
                class_shap = shap_values[target_class_idx][0]
            elif len(shap_values.shape) == 3:
                class_shap = shap_values[0, :, target_class_idx]
            else:
                class_shap = shap_values[0]
        else:
            # For Logistic Regression, exact linear attribution is w_c * x_i
            coef = self.model.coef_[target_class_idx]
            x_dense = X_sparse.toarray()[0]
            class_shap = coef * x_dense

        # Find non-zero feature indices in this document
        non_zero_indices = X_sparse.nonzero()[1]
        if len(non_zero_indices) == 0:
            return {
                "top_positive": [],
                "top_negative": [],
                "target_class": LABEL_NAMES[target_class_idx]
            }

        scores = class_shap[non_zero_indices]
        words = self.feature_names[non_zero_indices]
        
        # Sort by importance
        sorted_order = np.argsort(scores)[::-1]
        sorted_words = words[sorted_order]
        sorted_scores = scores[sorted_order]

        positive_features = []
        negative_features = []

        for w, s in zip(sorted_words, sorted_scores):
            if s > 0 and len(positive_features) < top_k:
                positive_features.append({"feature": str(w), "score": float(s)})
            elif s < 0 and len(negative_features) < top_k:
                negative_features.append({"feature": str(w), "score": float(s)})

        return {
            "target_class": LABEL_NAMES[target_class_idx],
            "target_class_idx": target_class_idx,
            "top_positive": positive_features,
            "top_negative": negative_features[::-1], # Most negative first
            "total_active_features": len(non_zero_indices)
        }

