# AI Hallucination and Misinformation Detection System

M.Tech project demonstrating two complementary approaches to detecting AI-generated
hallucinations and factually incorrect information.

---

## System Overview

| Component | Description |
|-----------|-------------|
| **TF-IDF Baseline** | Supervised binary classifier (hallucinated vs. non-hallucinated) trained on a dialogue dataset |
| **Evidence-Based NLI** | Retrieval-augmented verification: Wikipedia IR + DeBERTa NLI cross-encoder |
| **Streamlit App** | Interactive web UI combining both systems |
| **Test Suite** | Pytest with mocked unit tests + real integration tests |
| **Benchmark Evaluator** | Curated 25-claim benchmark with P/R/F1/coverage metrics |

---

## Project Structure

```
hallucination_detection/
├── dialogue_data.csv              # 10,000-row paired dialogue dataset
├── train.py                       # TF-IDF baseline training pipeline
├── predict.py                     # Single-response baseline prediction CLI
├── evaluate.py                    # Baseline metrics display
├── fact_checker.py                # Evidence-based NLI verification engine (v2)
├── app.py                         # Streamlit web application
├── evaluate_fact_checker.py       # NLI system benchmark evaluation
├── requirements.txt               # Python dependencies
├── models/                        # Saved scikit-learn pipelines (.joblib)
├── processed_data/                # Labelled train/val/test splits
├── results/                       # Training reports, metrics, predictions
└── tests/
    ├── __init__.py
    └── test_fact_checker.py       # Pytest suite (unit + integration)
```

---

## Setup (Windows)

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train baseline models if not already trained
python train.py --data dialogue_data.csv --seed 42
```

---

## Running the Applications

### Streamlit Web App (recommended for demo)
```powershell
streamlit run app.py
```
Opens at `http://localhost:8501`. Three tabs:
- **Verify Claim** — enter any factual claim and get an evidence-based verdict
- **Baseline Comparison** — compare TF-IDF models side-by-side
- **About & Limitations** — system explanation and caveats

### Evidence-Based Fact Checker (CLI)
```powershell
python fact_checker.py
# Enter: "The Sun is a star."
# Enter: "The Sun revolves around the Earth."
```

### TF-IDF Baseline Prediction
```powershell
# Response-only model
python predict.py --response "The example response to classify."

# Context + response model
python predict.py \
  --model context_plus_response__logistic_regression \
  --knowledge "Relevant background facts" \
  --dialogue-history "Previous conversation" \
  --response "Response to classify"
```

### View Baseline Metrics
```powershell
python evaluate.py
```

---

## Running Tests

```powershell
# Unit tests only (fast, no network, no model download — uses mocks)
python -m pytest tests/ -v

# Unit tests + integration tests (requires internet + ~30s model load)
python -m pytest tests/ -v --integration

# Run a specific test class
python -m pytest tests/test_fact_checker.py::TestInputValidation -v
```

### Test coverage
| Class | Tests | Network required? |
|-------|-------|-------------------|
| `TestInputValidation` | 3 | No |
| `TestWikipediaRetrieval` | 6 | No (mocked) |
| `TestSentenceRanking` | 5 | No |
| `TestNLIOutputParsing` | 6 | No (mocked) |
| `TestEvidenceAggregation` | 7 | No |
| `TestCheckClaimEndToEnd` | 10 | No (fully mocked) |
| `TestIntegration` | 4 | **Yes** (`--integration`) |

---

## Benchmark Evaluation (NLI System)

```powershell
# Run against 25-claim curated benchmark
python evaluate_fact_checker.py

# With custom thresholds
python evaluate_fact_checker.py --nli-threshold 0.65 --min-tfidf 0.05

# Verbose (shows evidence and sources per claim)
python evaluate_fact_checker.py --verbose

# Save results to JSON
python evaluate_fact_checker.py --output-json results/nli_benchmark.json
```

Metrics reported: Accuracy, Precision, Recall, F1 (per class + macro),
Coverage, Decisive Accuracy, Confusion Matrix.

---

## Existing Baseline Results (TF-IDF)

| Feature Set | Classifier | Test Acc | Test F1 | Val F1 | Paired Acc |
|---|---|---|---|---|---|
| Response Only | Logistic Regression | 80.30% | 80.11% | 78.17% | 89.67% |
| Response Only | Linear SVC | 80.20% | 79.85% | **79.34%** | 88.93% |
| Context + Response | Logistic Regression | 73.43% | 71.65% | 68.85% | 91.00% |
| Context + Response | Linear SVC | 74.23% | 73.00% | 70.97% | 91.20% |

**Best model by validation F1**: `response_only__linear_svc`

> ⚠️ These metrics measure agreement with the dataset's supplied hallucination
> labels. They do **not** establish real-world misinformation detection accuracy.

---

## NLI Pipeline Configuration

Configurable constants in `fact_checker.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `NLI_CONFIDENCE_THRESHOLD` | 0.60 | Min score for a decisive NLI verdict |
| `MIN_TFIDF_SIMILARITY` | 0.05 | Min cosine similarity to include a sentence |
| `TOP_K_SENTENCES` | 5 | Sentences evaluated per Wikipedia page |
| `MAX_WIKI_PAGES` | 3 | Wikipedia pages fetched per claim |
| `WIKI_SEARCH_LIMIT` | 8 | Candidate titles from search API |

All thresholds are **experimental parameters** — evaluate on a held-out
benchmark before treating them as calibrated.

---

## Key Bug Fixes (v1 → v2)

| Bug | Impact | Fix |
|-----|--------|-----|
| `text-classification` pipeline missing `top_k=None` | Only top-1 label returned; full score distribution unavailable | Added `top_k=None`; all three label scores now inspected |
| First non-empty Wikipedia page accepted regardless of relevance | Irrelevant articles could drive incorrect verdicts | Multi-page retrieval (up to 3); all pages evaluated |
| TF-IDF threshold `> 0` too permissive | Nearly any sentence qualified as evidence | Raised to configurable `MIN_TFIDF_SIMILARITY = 0.05` |
| No conflicting-evidence detection | Contradicting sources could silently override each other | Conflict detection added; INSUFFICIENT EVIDENCE returned when both entailment and contradiction are strong |
| Orphaned `except Exception` block | Structurally confusing, potentially masking errors | Rewritten with clean exception hierarchy |
| No structured result with provenance | No source URLs, no evidence sentences in output | `FactCheckResult` dataclass with full provenance, `to_dict()` for JSON export |

---

## Interpretation Warnings

1. **The baseline and NLI metrics are not comparable.** They measure performance
   on fundamentally different tasks (supervised classification vs. open-domain
   claim verification) and must not be combined into a single accuracy figure.

2. **SUPPORTED does not mean true.** It means the retrieved Wikipedia evidence
   appears to entail the claim under the NLI model. Wikipedia may be wrong,
   the NLI model may be wrong, or an irrelevant article may have been retrieved.

3. **INSUFFICIENT EVIDENCE is a legitimate outcome**, not always an error. When
   evidence is absent, ambiguous, or below the confidence threshold, this is the
   correct conservative verdict.

4. **The NLI confidence threshold is not calibrated.** Changing it without
   re-evaluating on a held-out benchmark invalidates any reported metrics.

5. **This is a research prototype**, not a production fact-checking system.
