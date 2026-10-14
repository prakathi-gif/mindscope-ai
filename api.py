"""
MindScope AI — Production-Grade FastAPI Inference Service.
Provides RESTful real-time inference, batch prediction, and explainability endpoints
with strict Pydantic input validation, PII redaction, and error handling.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any
import numpy as np
import psutil
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    APP_NAME, APP_VERSION, LABEL_NAMES, NUM_CLASSES,
    CATEGORY_DESCRIPTIONS, PATH_LR_MODEL, PATH_TFIDF_VECTORIZER,
    PATH_XGB_MODEL, PATH_ROBERTA_WEIGHTS, PATH_ENSEMBLE_CONFIG,
    ROBERTA_CHECKPOINT
)
from data.loader import redact_sensitive_pii
from data.preprocessor import clean_academic_text

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_available_models()
    yield

app = FastAPI(
    title=f"{APP_NAME} Inference API",
    description=(
        "Production-grade RESTful API for mental-health text analytics, multi-paradigm "
        "model inference (Logistic Regression, XGBoost, RoBERTa, Late-Fusion Ensemble), "
        "and explainability attribution."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# ---------------------------------------------------------
# Model Cache & Lazy Loader
# ---------------------------------------------------------
class ModelRegistry:
    def __init__(self):
        self.lr_model = None
        self.xgb_model = None
        self.xgb_preprocessor = None
        self.roberta_clf = None
        self.ensemble = None
        self.is_loaded = False

    def load_available_models(self):
        """Loads available trained models from disk."""
        import joblib
        # 1. Logistic Regression
        if PATH_LR_MODEL.exists() and PATH_TFIDF_VECTORIZER.exists():
            try:
                from models.baseline_lr import BaselineLogisticRegressionModel
                self.lr_model = BaselineLogisticRegressionModel()
                self.lr_model.load(PATH_LR_MODEL, PATH_TFIDF_VECTORIZER)
            except Exception as e:
                print(f"[Warning] Failed loading Logistic Regression: {e}")

        # 2. XGBoost
        if PATH_XGB_MODEL.exists():
            try:
                from models.xgboost_model import TFIDFXGBoostModel
                self.xgb_model = TFIDFXGBoostModel()
                self.xgb_model.load(PATH_XGB_MODEL)
            except Exception as e:
                print(f"[Warning] Failed loading XGBoost: {e}")

        # 3. RoBERTa Transformer
        if PATH_ROBERTA_WEIGHTS.exists():
            try:
                import torch
                from models.transformer_model import TransformerClassifier
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.roberta_clf = TransformerClassifier(checkpoint=ROBERTA_CHECKPOINT, device=device)
                self.roberta_clf.load(PATH_ROBERTA_WEIGHTS)
            except Exception as e:
                print(f"[Warning] Failed loading RoBERTa: {e}")

        # 4. Hybrid Ensemble
        if PATH_ENSEMBLE_CONFIG.exists():
            try:
                from models.ensemble import HybridProbabilityEnsemble
                self.ensemble = HybridProbabilityEnsemble()
                self.ensemble.load(PATH_ENSEMBLE_CONFIG)
            except Exception as e:
                print(f"[Warning] Failed loading Ensemble: {e}")

        self.is_loaded = True

    def get_status(self) -> Dict[str, bool]:
        return {
            "logistic_regression": self.lr_model is not None,
            "xgboost": self.xgb_model is not None,
            "roberta": self.roberta_clf is not None,
            "hybrid_ensemble": (self.ensemble is not None and self.xgb_model is not None)
        }

registry = ModelRegistry()


# ---------------------------------------------------------
# Pydantic Schemas & Input Validation
# ---------------------------------------------------------
ModelChoice = Literal["hybrid_ensemble", "roberta", "xgboost", "logistic_regression"]

class SingleInferenceRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=15000,
        description="The clinical disclosure text to classify.",
        examples=["I have been unable to sleep for four days and feel an overwhelming sense of dread and racing thoughts."]
    )
    model: ModelChoice = Field(
        default="hybrid_ensemble",
        description="Target model architecture for inference."
    )
    return_probabilities: bool = Field(
        default=True,
        description="If True, includes calibrated probabilities across all 8 classes."
    )
    redact_pii: bool = Field(
        default=True,
        description="If True, scrubs usernames, emails, and URLs prior to tokenization."
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Input text must contain non-whitespace characters.")
        return v


class BatchInferenceRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of text samples to classify (Max 50 items per batch)."
    )
    model: ModelChoice = Field(
        default="hybrid_ensemble",
        description="Target model architecture for batch inference."
    )
    redact_pii: bool = Field(
        default=True,
        description="If True, sanitizes all inputs."
    )

    @field_validator("texts")
    @classmethod
    def validate_texts_list(cls, v: List[str]) -> List[str]:
        for idx, text in enumerate(v):
            if not isinstance(text, str) or len(text.strip()) < 3:
                raise ValueError(f"Item at index {idx} must be a non-empty string of at least 3 characters.")
        return v


class SingleInferenceResponse(BaseModel):
    predicted_label: str = Field(..., description="Highest-probability mental health category")
    predicted_id: int = Field(..., description="Zero-indexed class ID (0-7)")
    confidence: float = Field(..., description="Softmax confidence score (0.0 to 1.0)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probability distribution across all classes")
    model_used: str = Field(..., description="Model identifier used for this prediction")
    sanitized_text: str = Field(..., description="Preprocessed/PII-redacted text processed by model")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp")


class BatchInferenceResponse(BaseModel):
    total_samples: int
    model_used: str
    total_latency_ms: float
    predictions: List[SingleInferenceResponse]


class FeatureAttributionItem(BaseModel):
    feature: str
    weight: float
    direction: Literal["positive (driving)", "negative (inhibiting)"]


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=15000, description="Input text to interpret")
    target_class: Optional[str] = Field(
        default=None,
        description="Specific target class to explain. Defaults to predicted class if omitted."
    )
    top_k: int = Field(default=8, ge=1, le=25, description="Number of top influential features to return")


class ExplainResponse(BaseModel):
    predicted_label: str
    target_class_explained: str
    explanation_method: str
    top_attributions: List[FeatureAttributionItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    uptime_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    loaded_models: Dict[str, bool]


# ---------------------------------------------------------
# Core Helper Logic
# ---------------------------------------------------------
def run_model_inference(text: str, model_type: str) -> tuple[np.ndarray, str]:
    """Computes probability distribution for single text input."""
    # Ensure loaded
    if not registry.is_loaded:
        registry.load_available_models()

    if model_type == "logistic_regression":
        if registry.lr_model is None:
            raise HTTPException(status_code=503, detail="Logistic Regression model artifact is not loaded.")
        probs = registry.lr_model.predict_proba([text])[0]
        return probs, "TF-IDF + Logistic Regression"

    elif model_type == "xgboost":
        if registry.xgb_model is None:
            raise HTTPException(status_code=503, detail="XGBoost model artifact is not loaded.")
        probs = registry.xgb_model.predict_proba([text])[0]
        return probs, "TF-IDF + XGBoost"

    elif model_type == "roberta":
        if registry.roberta_clf is None:
            # Fallback to XGBoost if RoBERTa weights not present
            if registry.xgb_model is not None:
                probs = registry.xgb_model.predict_proba([text])[0]
                return probs, "XGBoost (Fallback for RoBERTa)"
            raise HTTPException(status_code=503, detail="RoBERTa model weights are not loaded.")
        probs = registry.roberta_clf.predict_proba([text])[0]
        return probs, "Fine-Tuned RoBERTa"

    elif model_type == "hybrid_ensemble":
        # Needs RoBERTa + XGBoost, or fallback to XGBoost
        if registry.xgb_model is None and registry.lr_model is None:
            raise HTTPException(status_code=503, detail="No models available for ensemble.")
        
        # XGBoost probs
        p_xgb = registry.xgb_model.predict_proba([text])[0] if registry.xgb_model else registry.lr_model.predict_proba([text])[0]
        
        if registry.roberta_clf is not None:
            p_rob = registry.roberta_clf.predict_proba([text])[0]
            alpha = registry.ensemble.optimal_alpha if registry.ensemble else 0.65
            fused = alpha * p_rob + (1.0 - alpha) * p_xgb
            fused /= fused.sum()
            return fused, f"Hybrid RoBERTa+XGBoost Ensemble (α={alpha:.2f})"
        else:
            return p_xgb, "XGBoost (Standalone Ensemble Mode)"

    raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/", tags=["System"])
def root_endpoint():
    """API overview and interactive docs redirect."""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "documentation": "/docs",
        "redoc": "/redoc",
        "health_check": "/health",
        "supported_classes": LABEL_NAMES,
        "status": "Operational"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns service health, loaded model inventory, and server memory metrics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info().rss / (1024 * 1024)
    return HealthResponse(
        status="healthy",
        app_name=APP_NAME,
        version=APP_VERSION,
        uptime_seconds=round(time.time() - START_TIME, 2),
        memory_usage_mb=round(mem_info, 2),
        cpu_percent=round(psutil.cpu_percent(interval=None), 2),
        loaded_models=registry.get_status()
    )


@app.get("/classes", tags=["Metadata"])
def get_classes_metadata():
    """Returns information on the 8 mental-health classification categories."""
    return {
        "num_classes": NUM_CLASSES,
        "classes": [
            {
                "id": i,
                "label": name,
                "description": CATEGORY_DESCRIPTIONS.get(name, "")
            }
            for i, name in enumerate(LABEL_NAMES)
        ]
    }


@app.get("/models", tags=["Metadata"])
def list_models():
    """Lists available model architectures and their loaded state."""
    status_dict = registry.get_status()
    return {
        "available_architectures": [
            {
                "id": "hybrid_ensemble",
                "name": "Hybrid Probability Ensemble",
                "description": "Late-fusion probability weighting of RoBERTa and XGBoost minimizing Log-Loss.",
                "loaded": status_dict["hybrid_ensemble"]
            },
            {
                "id": "roberta",
                "name": "Fine-Tuned RoBERTa",
                "description": "Bidirectional transformer fine-tuned on Reddit mental health discourse.",
                "loaded": status_dict["roberta"]
            },
            {
                "id": "xgboost",
                "name": "TF-IDF + XGBoost",
                "description": "Gradient-boosted decision trees over unigram/bigram sublinear TF-IDF features.",
                "loaded": status_dict["xgboost"]
            },
            {
                "id": "logistic_regression",
                "name": "TF-IDF + Logistic Regression",
                "description": "Regularized multinomial linear baseline with sub-millisecond inference.",
                "loaded": status_dict["logistic_regression"]
            }
        ]
    }


@app.post("/predict", response_model=SingleInferenceResponse, tags=["Inference"])
def predict_single(req: SingleInferenceRequest):
    """
    Real-time inference endpoint for a single text disclosure.
    Performs PII redaction, feature extraction, and model inference with strict validation.
    """
    t0 = time.time()
    
    # 1. PII Redaction & Normalization
    processed_text = redact_sensitive_pii(req.text) if req.redact_pii else req.text
    cleaned = clean_academic_text(processed_text)
    if not cleaned.strip():
        cleaned = processed_text.strip()

    # 2. Model Inference
    probs, model_tag = run_model_inference(cleaned, req.model)
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    latency_ms = round((time.time() - t0) * 1000, 2)

    prob_dict = {LABEL_NAMES[i]: round(float(probs[i]), 4) for i in range(len(LABEL_NAMES))} if req.return_probabilities else None

    return SingleInferenceResponse(
        predicted_label=LABEL_NAMES[pred_idx],
        predicted_id=pred_idx,
        confidence=round(conf, 4),
        probabilities=prob_dict,
        model_used=model_tag,
        sanitized_text=cleaned,
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/predict/batch", response_model=BatchInferenceResponse, tags=["Inference"])
def predict_batch(req: BatchInferenceRequest):
    """
    Batch inference endpoint handling up to 50 items per payload.
    """
    t0_batch = time.time()
    results: List[SingleInferenceResponse] = []

    for text_sample in req.texts:
        t0 = time.time()
        processed_text = redact_sensitive_pii(text_sample) if req.redact_pii else text_sample
        cleaned = clean_academic_text(processed_text)
        if not cleaned.strip():
            cleaned = processed_text.strip()

        probs, model_tag = run_model_inference(cleaned, req.model)
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        latency_ms = round((time.time() - t0) * 1000, 2)

        prob_dict = {LABEL_NAMES[i]: round(float(probs[i]), 4) for i in range(len(LABEL_NAMES))}
        results.append(
            SingleInferenceResponse(
                predicted_label=LABEL_NAMES[pred_idx],
                predicted_id=pred_idx,
                confidence=round(conf, 4),
                probabilities=prob_dict,
                model_used=model_tag,
                sanitized_text=cleaned,
                latency_ms=latency_ms,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        )

    total_latency_ms = round((time.time() - t0_batch) * 1000, 2)
    return BatchInferenceResponse(
        total_samples=len(results),
        model_used=req.model,
        total_latency_ms=total_latency_ms,
        predictions=results
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_text(req: ExplainRequest):
    """
    Explainability endpoint: Computes token/n-gram feature attributions for a given text.
    Uses TF-IDF + Logistic Regression / XGBoost linear attributions for transparent interpretability.
    """
    t0 = time.time()
    cleaned = clean_academic_text(redact_sensitive_pii(req.text))
    if not cleaned.strip():
        cleaned = req.text.strip()

    # Predict class
    probs, _ = run_model_inference(cleaned, "logistic_regression")
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]

    # Target class to explain
    target_label = req.target_class if (req.target_class and req.target_class in LABEL_NAMES) else pred_label
    target_idx = LABEL_NAMES.index(target_label)

    # Attribution extraction from Logistic Regression / Vectorizer
    if registry.lr_model is None or not hasattr(registry.lr_model, "classifier"):
        raise HTTPException(status_code=503, detail="Baseline linear explainer artifact not loaded.")

    clf = registry.lr_model.classifier
    vec = registry.lr_model.preprocessor.vectorizer
    feature_names = vec.get_feature_names_out()

    # Sparse representation of the single text
    X_single = vec.transform([cleaned])
    non_zero_indices = X_single.indices

    attributions: List[FeatureAttributionItem] = []
    if len(non_zero_indices) > 0:
        coefs = clf.coef_[target_idx]
        scores = []
        for idx in non_zero_indices:
            feat_name = feature_names[idx]
            tfidf_val = X_single[0, idx]
            weight = float(coefs[idx] * tfidf_val)
            scores.append((feat_name, weight))

        # Sort by absolute impact
        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        top_scores = scores[:req.top_k]

        for feat, score in top_scores:
            direction = "positive (driving)" if score > 0 else "negative (inhibiting)"
            attributions.append(FeatureAttributionItem(
                feature=feat,
                weight=round(score, 4),
                direction=direction
            ))

    latency_ms = round((time.time() - t0) * 1000, 2)
    return ExplainResponse(
        predicted_label=pred_label,
        target_class_explained=target_label,
        explanation_method="TF-IDF Linear Attribution (Margin Contribution)",
        top_attributions=attributions,
        latency_ms=latency_ms
    )


if __name__ == "__main__":
    import uvicorn
    print(f"Starting {APP_NAME} FastAPI server on http://127.0.0.1:8000...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
