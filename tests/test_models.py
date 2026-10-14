"""
Unit tests for classical models and the evaluation engine.
"""

import pytest
import numpy as np
from models.baseline_lr import BaselineLogisticRegressionModel
from models.xgboost_model import TFIDFXGBoostModel
from models.evaluator import evaluate_classification_predictions, compute_expected_calibration_error

def test_evaluator_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    # Near-perfect probabilities for 3 classes
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.85, 0.1, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.1, 0.85]
    ])
    res = evaluate_classification_predictions(y_true, y_prob, ["c0", "c1", "c2"], "TestModel")
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    assert res["log_loss"] < 0.5
    assert res["ece"] < 0.2

def test_baseline_logistic_regression():
    texts = [
        "sad depressed anhedonia tired hopeless",
        "happy joyful excited energetic thrilled",
        "panic heart attack racing chest tight dread",
        "unfocused distracted daydreaming hyperactive executive"
    ]
    labels = np.array([0, 1, 2, 3])
    
    model = BaselineLogisticRegressionModel(max_features=50)
    model.fit(texts, labels)
    probs = model.predict_proba(texts)
    
    assert probs.shape == (4, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)

