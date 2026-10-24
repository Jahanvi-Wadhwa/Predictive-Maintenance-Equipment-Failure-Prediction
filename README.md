# Predictive Maintenance - Equipment Failure Prediction

An end-to-end machine learning project that estimates the risk of industrial
machine failure from operating conditions (temperatures, rotational speed,
torque, tool wear), with reproducible training, model comparison,
explainability, automated tests and an interactive Streamlit dashboard.

> **What this is:** a student portfolio project that runs entirely locally
> with free, open-source tools. It scores readings you type or upload - it is
> **not** a live sensor system, not a real-time deployment, and not a
> safety-instrumented system.

---

## 1. Problem statement

Unplanned equipment failure stops production lines, damages hardware and can
endanger people. Traditional maintenance is either *reactive* (fix it after
it breaks) or *calendar-based* (service it on a fixed schedule regardless of
condition). Both waste money.

**Predictive maintenance** asks a different question: given what the machine
is doing *right now*, how likely is it to fail? If that probability can be
estimated reliably, maintenance can be scheduled **before** a breakdown at
the lowest possible cost.

This project answers that question with a complete, reproducible ML workflow:

- Can five common operating sensors predict machine failure?
- Which model does the job best, and how do we prove it without cheating
  (data leakage, test-set peeking, misleading accuracy)?
- Which operating conditions drive each prediction, and can a non-technical
  user understand the result?

## 2. Dataset

**AI4I 2020 Predictive Maintenance Dataset** - UCI Machine Learning Repository
(dataset id 601):

- Source: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- 10,000 simulated records of industrial machines
- **339 failures (3.39%)** - a strongly imbalanced target

| Column | Type | Used as input? | Notes |
|---|---|---|---|
| `Air temperature [K]` | numeric | yes | ambient air temperature |
| `Process temperature [K]` | numeric | yes | process temperature |
| `Rotational speed [rpm]` | numeric | yes | shaft speed |
| `Torque [Nm]` | numeric | yes | shaft torque |
| `Tool wear [min]` | numeric | yes | accumulated tool wear |
| `Type` | categorical | yes | product variant L / M / H |
| `power_kw`, `temp_diff_k` | derived | yes | engineered from the sensors above (no extra data) |
| `Machine failure` | target | **no** | 1 if any failure mode occurred |
| `TWF, HDF, PWF, OSF, RNF` | labels | **no - leakage** | these *compose* the target |
| `UDI`, `Product ID` | ids | **no** | row/product identifiers |

**Leakage decision:** `Machine failure = 1` if *any* of the five failure-mode
columns is 1. Feeding them to the model would hand it the answer, so they are
excluded from all model inputs and used only for exploratory analysis.
Identifiers are excluded as well. `tests/test_features.py` asserts this.

**Setup:** the CSV ships with this repo (`data/raw/ai4i2020.csv`, ~500 KB).
To (re)download it programmatically with a SHA-256 integrity check:

```bash
python -m src.data
```

Manual fallback instructions live in [`data/README.md`](data/README.md).

## 3. Architecture and ML workflow

```
data/raw/ai4i2020.csv
        │
        ▼
┌───────────────────  python -m src.eda  ───────────────────┐
│ quality checks, class imbalance, leakage review, charts   │──► reports/
└───────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────  python -m src.train  ──────────────────┐
│ stratified 80/20 split (seed 42, test set sealed)         │
│                                                           │
│ for each of 3 models:                                     │
│   pipeline = imputer → scaler/one-hot → [SMOTE] → model   │
│   5-fold stratified CV on TRAIN only (ROC-AUC, PR-AUC,    │
│                                       recall, F1)         │
│   fit on full train → evaluate ONCE on test               │
│                                                           │
│ select best by CV average precision (test never decides)  │
│ risk bands from out-of-fold training probabilities        │
│ permutation importance on test                            │
└───────────────────────────────────────────────────────────┘
        │
        ▼
models/failure_model.joblib ──┐   models/artifacts/*.json|csv
reports/figures/*.png ────────┤
                              ▼
                 ┌────  streamlit run app.py  ────┐
                 │ Overview | Prediction | Eval   │
                 │ Feature Insights | CSV batch   │
                 └────────────────────────────────┘
```

Design rules the code enforces:

1. **One prediction path.** The dashboard imports `src/predict.py` - the same
   module the tests use. No duplicated preprocessing or logic.
2. **Leakage-safe by construction.** Imputation, scaling, one-hot encoding and
   SMOTE live *inside* the sklearn/imblearn pipeline, so they refit on every
   training fold and never see validation/test rows.
3. **The test set is sealed.** Model selection and risk-threshold tuning use
   training-set cross-validation only; the test set is scored once at the end.
4. **Deterministic.** `RANDOM_SEED = 42` for the split, CV shuffling, model
   constructors and SMOTE - rerunning `python -m src.train` reproduces every
   number in this README exactly.

## 4. Repository structure

```
├── app.py                     # Streamlit entry point (Overview page)
├── pages/
│   ├── 1_Failure_Risk_Prediction.py
│   ├── 2_Model_Evaluation.py
│   ├── 3_Feature_Insights.py
│   └── 4_Sample_Analysis.py
├── src/
│   ├── config.py              # paths, seeds, columns, training constants
│   ├── data.py                # programmatic download + schema validation
│   ├── eda.py                 # exploratory analysis -> reports/
│   ├── features.py            # feature engineering + preprocessing builder
│   ├── train.py               # CV, model comparison, artifact saving
│   ├── evaluate.py            # metrics + chart utilities
│   ├── explain.py             # permutation importance
│   ├── predict.py             # model loading, validation, prediction
│   └── dashboard_utils.py     # shared dashboard styling/helpers
├── tests/                     # 46 pytest tests
├── data/
│   ├── README.md              # dataset setup instructions
│   └── raw/ai4i2020.csv       # committed dataset (500 KB)
├── models/
│   ├── failure_model.joblib   # saved pipeline of the selected model
│   └── artifacts/
│       ├── metrics.json       # every CV/test metric + curve data
│       ├── permutation_importance.csv
│       └── reference_stats.json   # form defaults & validation ranges
├── reports/
│   ├── eda_report.md
│   └── figures/               # generated charts (regenerate any time)
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Installation

Requires **Python 3.10+** (tested on 3.12).

### Windows (PowerShell / CMD)

```powershell
git clone https://github.com/Jahanvi-Wadhwa/Predictive-Maintenance-Equipment-Failure-Prediction.git
cd Predictive-Maintenance-Equipment-Failure-Prediction

python -m venv .venv
.\.venv\Scripts\activate          # PowerShell
# .venv\Scripts\activate.bat      # CMD

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/Jahanvi-Wadhwa/Predictive-Maintenance-Equipment-Failure-Prediction.git
cd Predictive-Maintenance-Equipment-Failure-Prediction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No paid APIs, cloud services, databases or special hardware are required.

## 6. Dataset setup

```bash
python -m src.data
```

Downloads the archive from UCI, extracts `data/raw/ai4i2020.csv` and verifies
its SHA-256 checksum. If the file is already present (it is committed), this
verifies it and does nothing else. If your network blocks the download,
follow the manual steps in [`data/README.md`](data/README.md).

## 7. Run EDA, training and evaluation

```bash
# 1. Exploratory analysis -> reports/eda_report.md + charts
python -m src.eda

# 2. Train all three models, cross-validate, evaluate, save artifacts
python -m src.train

# 3. Verify the saved model scores a real dataset row
python -m src.predict
```

`python -m src.train` prints the full comparison table and writes
`models/failure_model.joblib` plus `models/artifacts/*`. Typical runtime: **1-2
minutes** on a laptop.

## 8. Launch the dashboard

```bash
streamlit run app.py
```

Then open <http://localhost:8501>.

## 9. Actual model results

> All numbers below come from a real run of `python -m src.train`
> (Python 3.12, scikit-learn 1.9.1, seed 42). They are also stored
> machine-readably in `models/artifacts/metrics.json`, which is what the
> dashboard displays - nothing is hardcoded.

**Split:** stratified 80/20 - train 8,000 rows (271 failures), test 2,000
rows (68 failures). The test set was never used for selection or tuning.

### 9.1 5-fold cross-validation (training set)

| Model | ROC-AUC | Avg. precision (PR-AUC) | Recall | F1 | Imbalance handling |
|---|---|---|---|---|---|
| Logistic Regression | 0.9193 ± 0.0108 | 0.4760 ± 0.0287 | 0.8194 | 0.2791 | SMOTE (in-pipeline) |
| Random Forest | **0.9766 ± 0.0117** | 0.8681 ± 0.0303 | 0.7160 | 0.8138 | balanced class weights |
| **HistGradientBoosting** | 0.9752 ± 0.0104 | **0.8813 ± 0.0162** | 0.8043 | **0.8354** | balanced class weights |

### 9.2 Held-out test set (2,000 rows, 68 real failures)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8600 | 0.1788 | **0.8676** | 0.2965 | 0.9348 | 0.4605 |
| Random Forest | **0.9910** | **0.9464** | 0.7794 | **0.8548** | **0.9761** | 0.8615 |
| **HistGradientBoosting (selected)** | 0.9900 | 0.8636 | 0.8382 | 0.8507 | 0.9748 | **0.8997** |

Confusion matrices (test set):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 1661 | 271 | 9 | 59 |
| Random Forest | 1929 | 3 | 15 | 53 |
| **HistGradientBoosting** | 1923 | 9 | **11** | **57** |

**Selected model: HistGradientBoosting** - chosen *only* on CV average
precision (0.8813, the best of the three), the metric that fits a rare,
imbalanced event. On the test set it caught **57 of 68 failures (recall
0.838)** with only **9 false alarms**, giving the best PR-AUC (0.900) and
effectively tied for best ROC-AUC.

**Reading the errors in maintenance terms:**

- **False negatives (11 missed failures):** machines that really failed but
  were rated low-risk - risk of unplanned downtime. This is the cost we
  minimise, which is why recall is prioritised over raw accuracy.
- **False positives (9 false alarms):** healthy machines flagged for review -
  an unnecessary but cheap inspection. Acceptable trade-off by design.
- **Logistic Regression's** 271 false alarms show what an under-fit linear
  model does on this data: it catches failures but drowns maintenance staff
  in noise (precision 0.18).
- **Random Forest** has the fewest false alarms but misses 4 more real
  failures than the selected model - a worse trade for this use case.

### 9.3 Risk bands used by the dashboard

Derived from 5-fold **out-of-fold training predictions** (never the test set):

| Band | Cut-off | OOF precision | OOF recall | Meaning |
|---|---|---|---|---|
| Low | < 0.12 | - | - | most healthy machines score here |
| Moderate | 0.12 - 0.50 | 0.71 | 0.83 | worth a planned inspection |
| High | >= 0.50 | 0.87 | 0.80 | model's own failure decision point |

The 0.12 cut-off is the *lowest* threshold reaching >= 70% precision while
keeping >= 80% recall on out-of-fold training predictions. These are
**project-defined relative risk bands, not validated industrial safety
limits**, and probabilities are not formally calibrated.

### 9.4 Feature importance (permutation, test set)

Mean drop in average precision when each feature is shuffled
(10 repeats, seed 42):

| Feature | Importance (mean ± std) |
|---|---|
| rotational_speed_rpm | 0.495 ± 0.034 |
| temp_diff_k (derived) | 0.395 ± 0.029 |
| tool_wear_min | 0.294 ± 0.030 |
| power_kw (derived) | 0.208 ± 0.015 |
| torque_nm | 0.045 ± 0.006 |
| type | 0.021 ± 0.009 |
| process_temperature_k | -0.001 ± 0.002 |
| air_temperature_k | -0.009 ± 0.002 |

Plain reading: **speed, the process/air temperature gap, tool wear and the
power proxy dominate**. Failing machines in this dataset tend to run slow
under load, show a shrinking temperature gap and have worn tools. Negative
values (air/process temperature alone) mean shuffling adds noise - those raw
sensors only matter through the derived gap. **Importance shows what the
model relies on; it does not prove causation.**

## 10. Tests

```bash
pytest tests/ -q
```

**46 tests** covering:

- feature engineering maths and leakage guards (`test_features.py`)
- input validation: missing columns, non-numeric values, invalid product
  types, out-of-range readings, extra columns, alias normalization
- model-loading errors: missing file, corrupt file, missing artifacts
- batch prediction shape/columns and risk-band boundaries (`test_predict.py`)
- training logic: stratified split determinism, CV score sanity, pure model
  selection, metrics.json internal consistency (precision/recall/F1 must
  match the stored confusion matrix exactly) (`test_pipeline.py`)
- dashboard: every file compiles, no hardcoded metrics or training logic, and
  **all five pages are executed headlessly with Streamlit's `AppTest`** - the
  Overview page must show the exact numbers from `metrics.json`, and clicking
  *Predict* must produce a probability and a risk badge
  (`test_dashboard.py`, `test_dashboard_app.py`)

Tests that need the trained artifacts skip cleanly with instructions when
`python -m src.train` has not been run yet.

## 11. Screenshots

> Replace the placeholders below with screenshots from your own run
> (`streamlit run app.py`, then capture each page).

| Overview | Failure Risk Prediction |
|---|---|
| ![Overview](docs/screenshots/overview.png) | ![Prediction](docs/screenshots/prediction.png) |

| Model Evaluation | Feature Insights |
|---|---|
| ![Evaluation](docs/screenshots/model_evaluation.png) | ![Insights](docs/screenshots/feature_insights.png) |

| Sample Analysis |
|---|
| ![Sample](docs/screenshots/sample_analysis.png) |

## 12. Limitations

- **Simulated data.** AI4I 2020 is synthetic; real plants have vibration,
  lubrication, humidity, operator and maintenance-history signals that this
  model never sees.
- **No time dimension.** The model estimates *risk*, not *when* a failure
  will occur - there is no survival/time-to-failure modelling and no window
  of prediction.
- **Not calibrated.** `predict_proba` outputs are useful scores for ranking
  and banding, not verified probabilities.
- **Static batch scoring.** Inputs come from the form or an uploaded CSV -
  there is no live sensor integration, streaming, alerting or database.
- **Unseen failure modes.** Only the five AI4I failure modes are represented;
  anything outside that distribution may be scored confidently but wrongly.
- **Risk bands are project conventions**, tuned on this dataset only, and
  must never be treated as safety thresholds.

## 13. Future improvements

- Probability calibration (isotonic/Platt) and a threshold-tuning report
  (precision-recall trade-off curve with an operating-point picker)
- Time-based splitting or a real longitudinal dataset for time-to-failure
  prediction (survival analysis)
- Hyperparameter search with nested cross-validation
- SHAP values for per-prediction explanations
- A drift check that compares new batches against training distributions
- Exporting the batch scorer as a simple REST endpoint (still fully local)

## 14. Skills and technologies

- **Languages & core:** Python 3.12, type hints, modular package layout
- **ML:** scikit-learn (pipelines, ColumnTransformer, StratifiedKFold,
  cross_validate, permutation importance, Logistic Regression, Random Forest,
  HistGradientBoosting), imbalanced-learn (SMOTE, imblearn Pipeline)
- **Data:** pandas, NumPy, Matplotlib
- **Dashboard:** Streamlit (multi-page app, custom CSS design system,
  `streamlit.testing.v1.AppTest`)
- **Engineering:** joblib artifact versioning, JSON/CSV artifact contracts,
  SHA-256 dataset verification, pytest, deterministic seeds, `.gitignore`,
  reproducible command-line entry points (`python -m ...`)
- **Practices:** leakage prevention, stratified evaluation of imbalanced
  targets, CV-only model selection, test-set discipline, model-agnostic
  explainability