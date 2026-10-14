"""
Model 5: Hybrid Transformer–XGBoost Probability Ensemble.
Optimizes fusion weight alpha strictly on validation split and evaluates on test set.
"""

import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import log_loss, f1_score
from models.evaluator import evaluate_classification_predictions
from config import PATH_ENSEMBLE_CONFIG, LABEL_NAMES

class HybridProbabilityEnsemble:
    """
    Ensemble combining RoBERTa contextual probabilities and TF-IDF XGBoost probabilities:
    P_ensemble = alpha * P_RoBERTa + (1 - alpha) * P_XGBoost
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.optimal_alpha: float = self.alpha
        self.val_scores: Dict[str, float] = {}

    def fit_validation_weights(
        self,
        val_probs_roberta: np.ndarray,
        val_probs_xgb: np.ndarray,
        y_val: np.ndarray,
        metric: str = "log_loss",
        num_candidates: int = 101
    ) -> float:
        """
        Performs grid search strictly on validation set to find optimal fusion parameter alpha*.
        Prevents test data leakage.
        """
        assert val_probs_roberta.shape == val_probs_xgb.shape, "Probability shape mismatch"
        alphas = np.linspace(0.0, 1.0, num_candidates)
        best_score = float("inf") if metric == "log_loss" else -float("inf")
        best_alpha = 0.5

        for a in alphas:
            fused_probs = a * val_probs_roberta + (1.0 - a) * val_probs_xgb
            fused_probs = np.clip(fused_probs, 1e-15, 1 - 1e-15)
            # Re-normalize rows
            fused_probs = fused_probs / fused_probs.sum(axis=1, keepdims=True)

            if metric == "log_loss":
                score = log_loss(y_val, fused_probs)
                if score < best_score:
                    best_score = score
                    best_alpha = a
            else:
                preds = np.argmax(fused_probs, axis=1)
                score = f1_score(y_val, preds, average="macro", zero_division=0)
                if score > best_score:
                    best_score = score
                    best_alpha = a

        self.optimal_alpha = float(best_alpha)
        self.alpha = float(best_alpha)
        self.val_scores = {
            "metric": metric,
            "best_validation_score": float(best_score),
            "optimal_alpha": float(best_alpha)
        }
        return self.optimal_alpha

    def combine_probabilities(
        self,
        probs_roberta: np.ndarray,
        probs_xgb: np.ndarray,
        custom_alpha: Optional[float] = None
    ) -> np.ndarray:
        """Fuses probability distributions using specified or fitted alpha."""
        a = self.alpha if custom_alpha is None else float(np.clip(custom_alpha, 0.0, 1.0))
        fused = a * probs_roberta + (1.0 - a) * probs_xgb
        # Re-normalize to ensure valid simplex distribution
        return fused / (np.sum(fused, axis=1, keepdims=True) + 1e-15)

    def evaluate_all_regimes(
        self,
        test_probs_roberta: np.ndarray,
        test_probs_xgb: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates the four required regimes:
        1. RoBERTa-only (alpha=1.0)
        2. XGBoost-only (alpha=0.0)
        3. Equal-weight ensemble (alpha=0.5)
        4. Validation-selected weighted ensemble (alpha=optimal_alpha)
        """
        # 1. RoBERTa-only
        res_roberta = evaluate_classification_predictions(
            y_test, test_probs_roberta, LABEL_NAMES, "RoBERTa-only (alpha=1.0)"
        )
        # 2. XGBoost-only
        res_xgb = evaluate_classification_predictions(
            y_test, test_probs_xgb, LABEL_NAMES, "XGBoost-only (alpha=0.0)"
        )
        # 3. Equal-weight
        probs_equal = self.combine_probabilities(test_probs_roberta, test_probs_xgb, custom_alpha=0.5)
        res_equal = evaluate_classification_predictions(
            y_test, probs_equal, LABEL_NAMES, "Equal-Weight Ensemble (alpha=0.5)"
        )
        # 4. Optimal validation-selected
        probs_optimal = self.combine_probabilities(test_probs_roberta, test_probs_xgb, custom_alpha=self.optimal_alpha)
        res_optimal = evaluate_classification_predictions(
            y_test, probs_optimal, LABEL_NAMES, f"Validation-Selected Ensemble (alpha={self.optimal_alpha:.2f})"
        )

        return {
            "roberta_only": res_roberta,
            "xgboost_only": res_xgb,
            "equal_weight": res_equal,
            "validation_selected": res_optimal,
            "optimal_alpha": self.optimal_alpha,
            "val_optimization_details": self.val_scores
        }

    def save(self, filepath=PATH_ENSEMBLE_CONFIG):
        with open(filepath, "w") as f:
            json.dump({
                "optimal_alpha": self.optimal_alpha,
                "current_alpha": self.alpha,
                "validation_tuning": self.val_scores
            }, f, indent=2)

    def load(self, filepath=PATH_ENSEMBLE_CONFIG):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.optimal_alpha = data.get("optimal_alpha", 0.5)
            self.alpha = self.optimal_alpha
            self.val_scores = data.get("validation_tuning", {})
        return self

