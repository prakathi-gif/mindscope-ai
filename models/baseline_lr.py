"""
Model 1: Classical Baseline NLP Model (TF-IDF + Logistic Regression).
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from data.preprocessor import ClassicalPreprocessor
from models.evaluator import evaluate_classification_predictions
from config import PATH_LR_MODEL, PATH_TFIDF_VECTORIZER, LABEL_NAMES

class BaselineLogisticRegressionModel:
    """TF-IDF + Regularized Multinomial Logistic Regression Baseline."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 500,
        class_weight: Optional[str] = "balanced",
        ngram_range: tuple = (1, 2),
        max_features: int = 10000,
        random_state: int = 42
    ):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.preprocessor = ClassicalPreprocessor(
            ngram_range=ngram_range,
            max_features=max_features
        )
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver="lbfgs",
            random_state=self.random_state
        )
        self.is_trained = False
        self.train_time_sec = 0.0

    def fit(self, train_texts: List[str], train_labels: np.ndarray):
        """Fits vectorizer and multinomial logistic regression."""
        t0 = time.time()
        X_train = self.preprocessor.fit_transform(train_texts)
        self.classifier.fit(X_train, train_labels)
        self.train_time_sec = time.time() - t0
        self.is_trained = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Transforms text and outputs probability distributions."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        X = self.preprocessor.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: List[str]) -> np.ndarray:
        prob = self.predict_proba(texts)
        return np.argmax(prob, axis=1)

    def evaluate(self, test_texts: List[str], test_labels: np.ndarray) -> Dict[str, Any]:
        """Evaluates model on held-out test split."""
        probs = self.predict_proba(test_texts)
        res = evaluate_classification_predictions(
            test_labels, probs, class_names=LABEL_NAMES, model_name="TF-IDF + Logistic Regression"
        )
        res["training_time_sec"] = self.train_time_sec
        return res

    def save(self, model_path=PATH_LR_MODEL, vec_path=PATH_TFIDF_VECTORIZER):
        """Saves model and preprocessor artifacts."""
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.preprocessor, vec_path)

    def load(self, model_path=PATH_LR_MODEL, vec_path=PATH_TFIDF_VECTORIZER):
        """Loads model and preprocessor artifacts."""
        self.classifier = joblib.load(model_path)
        self.preprocessor = joblib.load(vec_path)
        self.is_trained = True
        return self
