"""
Unit tests for the Hybrid Transformer–XGBoost Probability Ensemble.
"""

import pytest
import numpy as np
from models.ensemble import HybridProbabilityEnsemble

def test_ensemble_probability_simplex():
    ensemble = HybridProbabilityEnsemble(alpha=0.6)
    
    # 3 samples, 8 classes
    p_rob = np.array([
        [0.7, 0.1, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03],
        [0.05, 0.8, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02],
        [0.1, 0.1, 0.6, 0.05, 0.05, 0.02, 0.04, 0.04]
    ])
    p_xgb = np.array([
        [0.5, 0.2, 0.1, 0.05, 0.05, 0.02, 0.04, 0.04],
        [0.1, 0.7, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03],
        [0.2, 0.1, 0.5, 0.05, 0.05, 0.02, 0.04, 0.04]
    ])

    fused = ensemble.combine_probabilities(p_rob, p_xgb)
    assert fused.shape == (3, 8)
    # Check probabilities sum to 1
    assert np.allclose(fused.sum(axis=1), 1.0)
    # Check all probabilities in [0, 1]
    assert np.all(fused >= 0.0) and np.all(fused <= 1.0)

def test_ensemble_validation_optimization():
    ensemble = HybridProbabilityEnsemble()
    p_rob = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ])
    p_xgb = np.array([
        [0.5, 0.5],
        [0.4, 0.6]
    ])
    y_val = np.array([0, 1])

    best_alpha = ensemble.fit_validation_weights(p_rob, p_xgb, y_val, metric="log_loss")
    # RoBERTa is more confident and correct, so optimal alpha should be high (> 0.5)
    assert best_alpha >= 0.5

