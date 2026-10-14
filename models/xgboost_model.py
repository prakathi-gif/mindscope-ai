"""
Model 2: Classical Nonlinear NLP Model (TF-IDF + XGBoost).
Optimized for sparse text matrices with multi:softprob objective.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import joblib
from xgboost import XGBClassifier
from data.preprocessor import ClassicalPreprocessor
from models.evaluator import evaluate_classification_predictions
from config import PATH_XGB_MODEL, PATH_TFIDF_VECTORIZER, LABEL_NAMES, NUM_CLASSES

class TFIDFXGBoostModel:
    """TF-IDF + Gradient Boosted Decision Trees (XGBoost) Classifier."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        ngram_range: tuple = (1, 2),
        max_features: int = 8000,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        
        self.preprocessor = ClassicalPreprocessor(
            ngram_range=ngram_range,
            max_features=max_features
        )
        self.classifier = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist"  # Fast histogram algorithm for sparse data
        )
        self.is_trained = False
        self.train_time_sec = 0.0

    def fit(self, train_texts: List[str], train_labels: np.ndarray, val_texts: Optional[List[str]] = None, val_labels: Optional[np.ndarray] = None):
        """Fits TF-IDF and XGBoost with optional early stopping on validation split."""
        t0 = time.time()
        X_train = self.preprocessor.fit_transform(train_texts)
        
        eval_set = None
        if val_texts is not None and val_labels is not None:
            X_val = self.preprocessor.transform(val_texts)
            eval_set = [(X_train, train_labels), (X_val, val_labels)]
            self.classifier.fit(X_train, train_labels, eval_set=eval_set, verbose=False)
        else:
            self.classifier.fit(X_train, train_labels, verbose=False)
            
        self.train_time_sec = time.time() - t0
        self.is_trained = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Transforms text and predicts class probability distribution."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        X = self.preprocessor.transform(texts)
        return self.classifier.predict_proba(X)

    def predict(self, texts: List[str]) -> np.ndarray:
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    def evaluate(self, test_texts: List[str], test_labels: np.ndarray) -> Dict[str, Any]:
        probs = self.predict_proba(test_texts)
        res = evaluate_classification_predictions(
            test_labels, probs, class_names=LABEL_NAMES, model_name="TF-IDF + XGBoost"
        )
        res["training_time_sec"] = self.train_time_sec
        return res

    def save(self, model_path=PATH_XGB_MODEL):
        joblib.dump({"classifier": self.classifier, "preprocessor": self.preprocessor}, model_path)

    def load(self, model_path=PATH_XGB_MODEL):
        data = joblib.load(model_path)
        self.classifier = data["classifier"]
        self.preprocessor = data["preprocessor"]
        self.is_trained = True
        return self

