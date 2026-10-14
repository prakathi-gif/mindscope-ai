# MindScope AI — An Explainable Transformer–Ensemble Framework for Mental Health-Related Text Classification

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Academic%20Research%20Only-emerald.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)

---

## 🔬 1. Project Overview & Research Framework

Mental-health-related discussions on social media contain rich linguistic patterns reflecting subjective emotional distress, personal experiences, and cognitive states. Machine learning models can learn to classify text according to the annotation categories present in training corpora. However, these categories represent social discourse themes rather than clinically verified diagnostic conditions.

**MindScope AI** is an advanced NLP research application investigating how transformer-based representations, classical machine learning, and late-fusion ensembles perform on a verified Reddit mental-health text corpus containing over **52,600 posts across 8 annotation categories** (`adhd`, `anxiety`, `bipolar`, `bpd`, `depression`, `none/control`, `ocd`, `ptsd`).

### Primary Research Question
> *Can a hybrid NLP framework combining contextual transformer representations and traditional TF-IDF-based machine learning provide useful, explainable text-classification performance compared with individual models?*

### Specific Evaluated Hypotheses:
1. **$H_1$ (Contextual Superiority):** Fine-tuned RoBERTa outperforms linear TF-IDF baselines due to sensitivity to negation patterns, temporal context, and semantic nuance.
2. **$H_2$ (Domain Specialization):** MentalBERT (pre-trained on mental health social media text) provides a measurable advantage over general-domain RoBERTa under a comparable protocol.
3. **$H_3$ (Ensemble Synergism):** Probability-level late fusion ($P_{\text{ensemble}} = \alpha P_{\text{RoBERTa}} + (1-\alpha) P_{\text{XGBoost}}$) reduces calibration error and improves generalization over individual models.
4. **$H_4$ (Attribution Coherence):** Classical SHAP n-gram importances align with Transformer Integrated Gradients token salience.
5. **$H_5$ (Manifold Separability):** Contextual latent embeddings projected via UMAP reveal distinct topological clustering corresponding to clinical annotation boundaries while capturing known comorbid overlaps.

---

## 🏛️ 2. Architectural Pipeline

```
Reddit Corpus (52.6k Posts)
      │
      ├──> PII Masking & Privacy Redaction (u/user, emails, URLs)
      │
      ├──> Stratified Split (Train: 70% | Validation: 15% | Held-Out Test: 15%)
      │
      ├──> [Classical Path] ──> Unigram/Bigram TF-IDF ──> 1. Logistic Regression
      │                                                ──> 2. XGBoost (multi:softprob)
      │
      ├──> [Transformer Path] ──> Byte-Pair BPE ───────> 3. Fine-Tuned RoBERTa
      │                                                ──> 4. MentalBERT Domain Benchmark
      │
      └──> [Late Fusion] ──> P_ensemble = α P_RoBERTa + (1-α) P_XGBoost
                                (α optimized on Validation Log-Loss)
                                       │
                                       ├──> SHAP TreeExplainer & LinearExplainer
                                       ├──> Captum Layer Integrated Gradients
                                       ├──> Multi-Head Self-Attention Maps
                                       └──> UMAP 2D/3D Embedding Manifolds
```

---

## 📊 3. Models Evaluated

| Model | Architecture | Paradigm | Parameter Count | Key Strength |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1** | TF-IDF + Logistic Regression | Linear Classical | ~48,000 | Ultra-fast inference (<2ms), transparent coefficients |
| **Model 2** | TF-IDF + XGBoost | Non-Linear Tree Ensemble | 100 Trees x 8 Classes | Captures non-linear feature interactions |
| **Model 3** | Fine-Tuned RoBERTa | Bidirectional Transformer | 124.6M | Deep contextual semantics, negation handling |
| **Model 4** | MentalBERT Benchmark | Domain Pretrained Transformer | 109.5M | Social mental-health domain vocabulary |
| **Model 5** | Hybrid Probability Ensemble | Validation-Optimized Late Fusion | Combined ($\alpha=0.65$) | Lowest Log-Loss and Expected Calibration Error |

---

## 💡 4. Explainable AI & Manifold Geometry

1. **Classical Model Interpretability:**
   - SHAP (`TreeExplainer` & `LinearExplainer`) feature attributions mapping positive (driving) and negative (inhibiting) n-grams for any target category.
2. **Transformer Token Salience:**
   - Captum `LayerIntegratedGradients` computed across subword embedding layers, rendering interactive color-coded pills.
3. **Self-Attention Visualization:**
   - Dynamic heatmaps of multi-head self-attention weight matrices across all 12 layers and 12 attention heads.
4. **UMAP Manifold Projections:**
   - High-dimensional mean-pooled sentence representations projected to 2D and 3D with Silhouette score and Davies-Bouldin cluster separation metrics.

---

## ⚠️ 5. Ethical Safeguards & Privacy Mandates

- **Non-Diagnostic Policy:** This system is strictly an academic research platform for aggregate NLP text analytics. It **must not** be used for clinical diagnosis, patient triage, or emergency response.
- **Privacy by Design:** Automatic regex redaction of Reddit usernames (`u/[USER]`), personal URLs, and email addresses.
- **Zero Cloud Leakage:** All inference and computation run strictly on local hardware (CPU/GPU) with no external cloud API calls.
- **Local Data Retention:** All cached datasets and checkpoints remain local and can be cleared with one click in Settings.

---

## 🚀 6. Installation & Execution

### 1. Prerequisites
Ensure Python 3.11 or 3.12 is installed. Clone the repository and navigate to the project directory:

```bash
cd "36 AML"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Bayesian Hyperparameter Optimization (Optuna)
Execute Tree-structured Parzen Estimator (TPE) optimization across XGBoost and Late-Fusion weighting:

```bash
python tune_optuna.py --samples 1200 --trials 15
```
Optimization trials and trajectories are automatically tracked to `artifacts/optuna_study.json`.

### 4. Run Master Training & Artifact Pipeline
To reproduce the experimental pipeline and generate model checkpoints, UMAP projections, and evaluation metrics:

```bash
python train_pipeline.py --samples 1200 --epochs 1 --batch_size 16
```

### 5. Run Full Pytest Test Suite
Validate all 15 unit tests covering preprocessors, models, late-fusion ensemble, and FastAPI endpoints:

```bash
python -m pytest tests/ -v
```

### 6. Launch Production FastAPI Inference Engine
Start the high-performance RESTful API with Pydantic schema validation and PII redaction:

```bash
python api.py
# Or with uvicorn directly:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/health`

#### Example Real-Time API Inference Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I feel completely hopeless and drained of any joy or motivation to live.", "model": "hybrid_ensemble"}'
```

### 7. Launch Interactive Research Dashboard (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 8. Containerized Deployment with Docker & Docker Compose
To run both the FastAPI backend and Streamlit UI inside isolated production containers:

```bash
# Build and run multi-service stack:
docker compose up --build -d

# Check running services:
docker compose ps

# Access services:
# FastAPI REST API:      http://localhost:8000/docs
# Streamlit Dashboard:   http://localhost:8501
```

### 9. Hugging Face Spaces Deployment
To deploy this project to Hugging Face Spaces:
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) selecting **Streamlit** (or **Docker**).
2. Push this repository to the Hugging Face Space git remote.
3. Hugging Face will automatically build and host the interactive dashboard with public HTTPS access.

---

## 📑 7. Research Dashboard Pages

1. **Research Overview:** Master research questions, hypotheses, formal task definition, and high-level evaluation metrics.
2. **Dataset Explorer:** Corpus distribution, imbalance metrics, word/char length histograms, and privacy-filtered post table.
3. **Text Analysis Lab:** Live multi-model inference console, preprocessing stages, side-by-side probability charts, and dynamic $\alpha$ slider.
4. **Model Laboratory:** Hyperparameter controls, Optuna Bayesian trial results, loss convergence curves, and hardware efficiency metrics.
5. **Explainable AI:** SHAP feature importances, Captum token attributions, and self-attention heatmaps.
6. **Embedding Explorer:** Interactive 2D/3D UMAP manifold projections and centroid distance matrices.
7. **Error Analysis:** Normalized confusion matrices, high-confidence error detector, and semantic overlap tables.
8. **Research Evaluation:** Standardized benchmark matrix, ROC/PR curves, reliability diagrams, hypothesis answers, and report export.
9. **Settings and Privacy:** Ethical compliance checklist, local cache purging, and compute engine telemetry.

---

## 📄 8. Academic IEEE Report
The formal academic project report is available in two formats within the `report/` directory:
- `report/IEEE_MindScope_Report.tex`: Double-column IEEE Conference/Transactions LaTeX source code.
- `report/IEEE_MindScope_Report.md`: Markdown document detailing problem formulation, literature review, methodology, mathematical definitions, error analysis, and references.

---

## 📜 9. License & Academic Attribution
Developed for Advanced Machine Learning (AML) and NLP research. Licensed strictly for academic, educational, and non-commercial scientific analysis.

