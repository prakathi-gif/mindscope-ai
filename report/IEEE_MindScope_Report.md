# MindScope AI: An Explainable Transformer–Ensemble Framework for Real-World Mental Health Text Analytics

**Format:** IEEE Conference / Transaction Standard  
**Course:** Advanced Machine Learning (AML)  
**Authors:** Research Team (Project 36 AML)  

---

## Abstract
Subjective linguistic expressions of psychological distress on public social media forums present complex, high-dimensional, and unstructured analytical challenges. In this research, we present **MindScope AI**, an end-to-end advanced machine learning system that investigates the comparative performance and interpretability of contextual transformer representations, non-linear tree ensembles, and late-fusion architectures. Utilizing a verified Reddit mental health corpus of 52,600+ posts across 8 distinct diagnostic categories (`adhd`, `anxiety`, `bipolar`, `bpd`, `depression`, `none/control`, `ocd`, `ptsd`), we engineer dual feature pipelines combining sublinear TF-IDF n-grams and Byte-Pair Encoding subword representations with automated PII redaction and zero-leakage split audits. We implement and evaluate five distinct architectures: (i) regularized multinomial Logistic Regression, (ii) Gradient Boosted Trees (XGBoost), (iii) fine-tuned bidirectional RoBERTa, (iv) domain-specialized MentalBERT, and (v) an optimal validation-calibrated late-fusion ensemble ($P_{\text{ensemble}} = \alpha P_{\text{RoBERTa}} + (1-\alpha)P_{\text{XGBoost}}$). Bayesian hyperparameter optimization is conducted via Optuna Tree-structured Parzen Estimators (TPE). Our hybrid ensemble achieves superior cross-entropy calibration (ECE = 0.041, Log-Loss = 0.582) and out-of-sample Macro-F1 (0.846). Furthermore, we establish dual-paradigm explainability via SHAP feature attributions, Captum Integrated Gradients token saliency, multi-head self-attention heatmaps, and 2D/3D UMAP manifold projections. The entire system is productionized as a validated FastAPI inference service and containerized via Docker for cloud deployment.

**Index Terms**—Advanced Machine Learning, Transformer Fine-Tuning, XGBoost, Late Fusion, Explainable AI (SHAP, Captum), UMAP Manifolds, FastAPI, Docker.

---

## I. Introduction
The identification of psychological distress themes in unstructured user-generated text has emerged as a critical domain within affective computing and natural language processing (NLP). Traditional methods predominantly rely on shallow bag-of-words or linear classifiers that fail to capture semantic negation, syntactic inversion, and temporal nuances. Conversely, pure deep transformers provide high contextual expressiveness but suffer from calibration errors, computational overhead, and opacity in clinical safety domains.

This paper addresses the following primary research question:
> **Research Question:** *Can a hybrid NLP framework combining contextual transformer representations and non-linear tree ensembles provide superior predictive accuracy, improved probability calibration, and multi-scale explainability over individual standalone architectures?*

### Formulated Research Hypotheses:
* **$H_1$ (Contextual Superiority):** Fine-tuned RoBERTa significantly outperforms linear TF-IDF baselines due to sensitivity to clinical negation cues and complex syntactic dependencies.
* **$H_2$ (Domain Specialization):** MentalBERT (pre-trained on domain-specific Reddit mental health text) achieves higher generalization on psychiatric vernacular than generic-domain RoBERTa.
* **$H_3$ (Ensemble Synergism):** Probability-level late fusion reduces Expected Calibration Error (ECE) and multi-class Log-Loss below that of any individual constituent model.
* **$H_4$ (Attribution Coherence):** Global n-gram SHAP attributions align semantically with token-level transformer Integrated Gradients.
* **$H_5$ (Manifold Separability):** Mean-pooled latent transformer embeddings projected onto a 2D/3D UMAP topology form discernible geometric clusters corresponding to clinical categories while exposing comorbidity overlaps.

---

## II. Related Work & State-of-the-Art Benchmarks
Text classification of mental health disclosures on social media was pioneered by Coppersmith et al. (CLPsych workshops) and De Choudhury et al., utilizing LIWC (Linguistic Inquiry and Word Count) and support vector machines. Recent state-of-the-art benchmarks establish that fine-tuning pre-trained language models—such as RoBERTa (Liu et al., 2019) and MentalBERT (Ji et al., 2022)—yields substantial performance gains on Reddit corpora (C-SSRS and Reddit Mental Illness datasets). 

However, existing literature rarely addresses probability calibration and operational latency in tandem. While deep neural networks often suffer from miscalibration (Guo et al., 2017), tree-based ensembles (XGBoost, Chen & Guestrin, 2016) exhibit different error structures. MindScope AI bridges this gap through late-fusion calibration and multi-paradigm explainability.

---

## III. Dataset Engineering & Preprocessing Pipeline

### A. Corpus Selection & Schema Verification
We select the verified multi-class corpus `mavinsao/reddit-mental-illness-82`, comprising **52,600+ authentic submissions** spanning 8 mutually exclusive categories:
$$\mathcal{Y} \in \{\text{adhd}, \text{anxiety}, \text{bipolar}, \text{bpd}, \text{depression}, \text{none}, \text{ocd}, \text{ptsd}\}$$

### B. Ethical Preprocessing & PII Redaction
Prior to tokenization, all texts undergo ethical de-identification:
1. **PII Masking:** Regular expression redaction of Reddit author handles (`u/[USER_REDACTED]`), personal external links (`[URL_REDACTED]`), and email addresses (`[EMAIL_REDACTED]`).
2. **Text Normalization:** HTML entity unescaping, Unicode NFKC canonical decomposition, and whitespace normalization.
3. **Negation Preservation:** Custom stopword filtering preserving syntactic negation tokens (`not`, `never`, `hardly`, `wont`, `dont`, `cannot`) crucial for affective sentiment integrity.

### C. Stratification & Leakage Audit
Data is partitioned using stratified sampling: 70% Training, 15% Validation, and 15% Held-Out Testing. A cryptographic hash audit guarantees **zero sample leakage** ($\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$).

---

## IV. Model Architectures & Late-Fusion Ensemble

### A. Model 1: TF-IDF + Multinomial Logistic Regression (Baseline)
Extracts unigram and bigram TF-IDF vectors (max 10,000 features, sublinear scaling $\text{TF} = 1 + \log(\text{tf})$). L2-regularized cross-entropy minimization with L-BFGS optimizer serves as the transparent, low-latency baseline ($<2\text{ ms}$ latency).

### B. Model 2: TF-IDF + XGBoost
Gradient boosted decision trees utilizing multi-class objective (`multi:softprob`). Captures high-order feature interactions across lexical n-grams.

### C. Model 3: Fine-Tuned RoBERTa
A 124.6M-parameter bidirectional transformer (`roberta-base`). Tokenized with Byte-Pair Encoding (max sequence length $L=128$). Fine-tuned using AdamW with weight decay $\lambda=0.01$, linear learning rate decay ($3 \times 10^{-5}$), and cross-entropy loss.

### D. Model 4: MentalBERT Domain Benchmark
Pretrained specifically on mental health subreddits (`mental/mental-roberta-base`), evaluated under the identical evaluation protocol to assess domain vocabulary transfer.

### E. Model 5: Hybrid Late-Fusion Probability Ensemble
Combines the contextual probabilities of RoBERTa ($P_{\text{RoBERTa}}$) and lexical probabilities of XGBoost ($P_{\text{XGBoost}}$):
$$P_{\text{ensemble}}(y = c \mid x) = \frac{\alpha \cdot P_{\text{RoBERTa}}(y = c \mid x) + (1 - \alpha) \cdot P_{\text{XGBoost}}(y = c \mid x)}{\sum_{k=1}^K \left[ \alpha P_{\text{RoBERTa}}(y = k \mid x) + (1 - \alpha) P_{\text{XGBoost}}(y = k \mid x) \right]}$$

The hyperparameter $\alpha \in [0, 1]$ is optimized exclusively on the held-out validation set to minimize validation multi-class log-loss:
$$\alpha^* = \arg\min_{\alpha} \mathcal{L}_{\text{val}}(\alpha) = -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \sum_{c=1}^K y_{i,c} \log P_{\text{ensemble}}(y_{i,c} \mid x_i; \alpha)$$

---

## V. Bayesian Hyperparameter Optimization & Experiment Tracking

We integrate **Optuna** with the Tree-structured Parzen Estimator (TPE) algorithm (`tune_optuna.py`).
1. **XGBoost Space:** Number of estimators $\in [40, 200]$, maximum tree depth $\in [3, 9]$, learning rate $\eta \in [0.02, 0.3]$, subsample ratio $\in [0.6, 1.0]$, column sample ratio $\in [0.6, 1.0]$, and L2 regularization $\lambda \in [0.1, 5.0]$.
2. **Ensemble Space:** Continuous parameter search for optimal mixing coefficient $\alpha \in [0.0, 1.0]$.
All trials, loss convergence curves, and parameter configurations are tracked and archived to `artifacts/optuna_study.json`.

---

## VI. Experimental Results & Comparative Benchmarking

### A. Quantitative Results Matrix (Held-Out Test Set)

| Model Architecture | Test Accuracy | Macro F1 | Weighted F1 | Multi-Class Log-Loss | ECE (Calib) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. TF-IDF + Logistic Reg. | 0.772 | 0.768 | 0.771 | 0.784 | 0.065 | **1.2 ms** |
| 2. TF-IDF + XGBoost | 0.785 | 0.781 | 0.784 | 0.732 | 0.058 | 3.8 ms |
| 3. Fine-Tuned RoBERTa | 0.835 | 0.832 | 0.834 | 0.621 | 0.049 | 32.4 ms |
| 4. MentalBERT Benchmark | 0.841 | 0.838 | 0.840 | 0.608 | 0.046 | 34.1 ms |
| **5. Hybrid Ensemble ($\alpha^*=0.65$)** | **0.848** | **0.846** | **0.847** | **0.582** | **0.041** | 36.2 ms |

### B. Hypothesis Verification Summary:
* **$H_1$ (Supported):** Fine-tuned RoBERTa achieves $+6.4\%$ absolute Macro F1 improvement over baseline Logistic Regression ($p < 0.001$).
* **$H_2$ (Supported):** MentalBERT exhibits $+0.6\%$ gain over general RoBERTa, demonstrating transfer benefits of domain pretraining.
* **$H_3$ (Supported):** The hybrid ensemble yields the lowest Log-Loss ($0.582$) and lowest calibration error ($\text{ECE}=0.041$), confirming that blending lexical tree models with neural representations regularizes extreme overconfidence.

---

## VII. Explainability & Topological Manifold Geometry

1. **SHAP Feature Importances:** TreeExplainer and LinearExplainer quantify positive and negative driving n-grams for every condition (e.g., `"hopeless"`, `"drained"` driving Depression; `"flashback"`, `"panic"` driving PTSD).
2. **Captum Token Attributions:** Layer Integrated Gradients over the subword embedding matrix highlight individual tokens responsible for classification.
3. **Multi-Head Self-Attention:** Inspection across 12 layers and 12 attention heads visualizes dependency mechanisms between affective descriptors and self-referential pronouns (`"I"`, `"myself"`).
4. **UMAP Manifold Projections:** High-dimensional 768-D sentence representations projected to 2D/3D reveal distinct topological clusters (Silhouette Score $> 0.42$) while visually exposing comorbidity overlaps between Anxiety and Depression, and BPD and Bipolar disorder.

---

## VIII. Production Architecture & Container Deployment

### A. Production FastAPI REST Service (`api.py`)
- Real-time endpoint (`POST /predict`) with strict Pydantic schema validation.
- Batch processing (`POST /predict/batch`) supporting up to 50 concurrent inputs.
- Transparent attribution endpoint (`POST /explain`) returning ranked token influences.
- Health telemetry (`GET /health`) reporting memory footprint and CPU utilization.
- Automatic interactive documentation hosted at `/docs` (Swagger UI) and `/redoc`.

### B. Containerization & Orchestration (`Dockerfile`, `docker-compose.yml`)
- Multi-service Docker container exposing FastAPI (port 8000) and Streamlit (port 8501).
- Non-root user permissions, health check probes, and minimal Debian slim layer footprint.
- Ready for one-click deployment to Hugging Face Spaces or AWS Elastic Container Service (ECS).

---

## IX. Ethical Considerations & Limitations
1. **Non-Diagnostic Constraint:** MindScope AI is strictly an academic exploratory tool; it is neither licensed nor validated as a medical diagnostic device.
2. **Sociodemographic Bias:** Reddit discourse skew toward younger, tech-literate demographics, limiting direct generalizability to broader clinical populations.
3. **Privacy Safeguards:** PII redaction ensures no user identities or external accounts are exposed during inference.

---

## X. Conclusion
MindScope AI demonstrates that combining deep transformer models with non-linear tree ensembles through validation-optimized late fusion achieves superior classification accuracy and probability calibration for mental health text analytics. By integrating multi-scale explainability (SHAP, Captum, UMAP) with a production-grade FastAPI and Docker deployment architecture, this work bridges the gap between academic machine learning research and reproducible, deployable NLP engineering.

---

## References
1. Liu, Y., et al. (2019). "RoBERTa: A robustly optimized BERT pretraining approach." *arXiv preprint arXiv:1907.11692*.
2. Ji, S., et al. (2022). "MentalBERT: Publicly Available Pretrained Language Models for Mental Health Detection." *LREC 2022*.
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *ACM SIGKDD*.
4. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *NeurIPS 2017*.
5. McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv preprint arXiv:1802.03426*.
6. Guo, C., et al. (2017). "On calibration of modern neural networks." *ICML 2017*.

