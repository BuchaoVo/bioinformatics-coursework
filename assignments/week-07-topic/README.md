<div align="center">

# Week 07 · Thyroid Diagnostic Codes & Model Comparison

**Benchmarking eight classifiers on 9,172 historical UCI thyroid records—from data quality and grouped splits to statistical comparisons and model interpretation.**

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#5-running-the-analysis)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Thyroid_(n%3D9%2C172)-0969DA?style=flat-square)](#2-dataset--experimental-setup)
[![Models](https://img.shields.io/badge/Models-8_Classifiers-2EA043?style=flat-square)](#4-key-results)

[Overview & Key Files](#1-overview--key-files) •
[Dataset & Setup](#2-dataset--experimental-setup) •
[Study Coverage](#3-study-coverage) •
[Key Results](#4-key-results) •
[Running the Analysis](#5-running-the-analysis)

</div>

---

## 1. Overview & Key Files

This folder contains the code, data, figures, and reports for **Week 07: Thyroid Diagnostic Codes and Model Comparison**.

Using **9,172** historical thyroid records, the study extends a Gaussian Naive Bayes baseline to **eight classifiers**, focusing on data quality, validation-based model selection, statistical comparisons, and feature interpretation (with PCA included as an exploratory component).

### Quick Entry Points

| Category | File | Description |
| :--- | :--- | :--- |
| **Reports & Visuals** | [Signed Report (PDF)](report/week07_signed_report.pdf) · [Word](report/week07_signed_report.docx) | Six-page final study write-up with editable Word source |
| **Reports & Visuals** | [Course Research Paper (PDF)](report/thyroid_model_comparison.pdf) | Six-page two-column manuscript version |
| **Reports & Visuals** | [Figure Atlas (PDF)](report/thyroid_analysis_figure_atlas.pdf) | 34-page visual atlas containing a cover and all 33 analysis figures |
| **Code & Data** | [Analysis Runner](code/run_study.py) | End-to-end data parsing, preprocessing, grouped splitting, and 8-model evaluation |
| **Code & Data** | [Field Dictionary](data/data_dictionary.csv) | Definitions and preprocessing rules for the 29 input fields |
| **Saved Outputs** | [Selected-Model Results](results/locked_test_result.json) | Holdout test metrics and bootstrap confidence intervals for LightGBM |

### Directory Layout

| Path | What's Inside |
| :--- | :--- |
| **`code/`** | Three scripts: `run_study.py` (main pipeline), `make_figures.py`, and `extra_curves.py` |
| **`data/`** | `thyroid0387.data` and `data_dictionary.csv` |
| **`results/`** | `16` CSV tables and `3` JSON summaries (including `dataset_summary.json` and `age_qc_flagged_rows.csv`) |
| **`figures/`** | `33` PNG figures used in the reports and visual atlas |
| **`report/`** | Compiled PDFs, editable Word documents, and document build scripts in `report/source/` |
| **`environment/`** | Python dependencies for modeling, plotting, SHAP, and document generation |

---

## 2. Dataset & Experimental Setup

### Dataset & Quality Control

The dataset comes from the [UCI Thyroid Disease](https://archive.ics.uci.edu/dataset/102/thyroid+disease) repository (`thyroid0387.data`, DOI `10.24432/C5D010`, CC BY 4.0):

* **Records & Target:** `9,172` records and `29` predictor fields. `diagnosis == '-'` defines class `0` (`6,771` records), while any other diagnosis code defines class `1` (`2,401` records). Diagnosis strings, record IDs, target labels, and derived diagnosis groups are excluded from predictors.
* **Age Cleaning:** Four values outside the `1–110` year range (`455`, `65511`, `65512`, and `65526`) are recoded as missing while keeping all rows intact (logged with 1-based line numbers in `results/age_qc_flagged_rows.csv`).
* **Missing Values:** `?` denotes missing entries—notably **TBG** (`96.19%` missing), **T3** (`28.39%`), and **TSH** (`9.18%`). Six binary measurement flags track whether their corresponding lab assays were performed.

### Modeling & Evaluation Protocol

* **Leakage-Safe Preprocessing:** Fit strictly inside training folds—median imputation, missingness indicators, and scaling for numeric variables; mode imputation for binary flags; and mode imputation plus one-hot encoding for sex and referral source—yielding **42 encoded features**.
* **Grouped Splits & CV:** Stratified grouped splitting (`SEED=2026`; secondary split `2027`) keeps exact feature-pattern groups disjoint across splits: **5,504** train (`1,441` positive), **1,834** validation (`480` positive), and **1,834** test (`480` positive). Cross-validation uses **3-fold `StratifiedGroupKFold`**.
* **Model & Threshold Selection:** Compares **GaussianNB, Logistic Regression, Random Forest, RBF-SVM, MLP, XGBoost, LightGBM, and CatBoost**. The primary model is selected by maximum validation **AP**, and each model's decision threshold is tuned by maximizing validation **F1** over `0.05–0.95` (step `0.01`, breaking ties toward `0.50`).
* **Interpretation & Statistical Tests:** Uses validation permutation importance (`5` repeats), feature-group ablations, and optional SHAP (`220` validation records). Descriptive Mann–Whitney and Fisher tests use Benjamini–Hochberg (BH) correction, while seven test-set McNemar comparisons against GaussianNB use Holm correction. Bootstrap confidence intervals use `650` resamples for the selected model (`250` per model for all-model comparisons).

---

## 3. Study Coverage

| Component | Included Materials & Methods |
| :--- | :--- |
| **Data Inspection & QC** | Raw dataset, field dictionary, age anomaly flags (`age_qc_flagged_rows.csv`), missingness summaries, and exploratory figures |
| **Preprocessing & Splits** | Fold-local preprocessing pipeline, pattern-grouped split assignments (`row_split.csv`), and 3-fold `StratifiedGroupKFold` |
| **Eight-Model Benchmark** | Full training runner, 3-fold CV summaries, and validation/test metric tables across all eight classifiers |
| **Threshold Tuning & Evaluation** | Validation F1 threshold sweep, holdout test probabilities, confusion counts, and bootstrap confidence intervals |
| **Statistical & Feature Analysis** | BH-adjusted feature tests, Holm-adjusted McNemar comparisons, feature ablations, permutation importance, SHAP, and subgroup analyses |
| **Written Deliverables** | Six-page signed report (PDF & Word), two-column research paper (PDF), and 34-page figure atlas (PDF) |

---

## 4. Key Results

**LightGBM** achieved the highest validation **Average Precision (0.9621)** and was selected with a validation-tuned threshold of approximately **0.50**:

| Holdout Test Metric | LightGBM Result |
| :--- | :---: |
| **Average Precision (AP)** | **`0.9736`** (`95% CI: 0.9546–0.9881`) |
| **ROC-AUC** | **`0.9937`** (`95% CI: 0.9909–0.9962`) |
| **F1 Score** | **`0.9344`** |
| **Balanced Accuracy** | **`0.9689`** |
| **Sensitivity / Specificity** | `0.9792` / `0.9586` |
| **Brier Score / Log Loss** | `0.0261` / `0.0964` |
| **Confusion Matrix (`TP / FP / TN / FN`)** | `470` / `56` / `1,298` / `10` |

> [!NOTE]
> * **Laboratory Feature Impact:** Removing the six lab measurements drops validation AP from **0.9621** down to **0.3473**, with **TSH** ranking first in validation permutation importance.
> * **Interpretation Scope:** Because the target reflects historical diagnostic codes and clinical measurement patterns with substantial missingness, these metrics characterize internal retrospective prediction rather than calibrated external clinical risk.

---

## 5. Running the Analysis

### 5.1 Run Experiments & Generate Figures

From the repository root, navigate to the Week 07 folder, set up a virtual environment (originally run on **Python 3.13.5** with `scikit-learn >= 1.8`), and run the scripts:

```bash
cd assignments/week-07-topic
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt

python code/run_study.py --cv 3 --shap
python code/make_figures.py
python code/extra_curves.py
```

By default, the scripts read `data/thyroid0387.data` and write outputs to `results/` and `figures/`. To write `run_study.py` outputs to a separate temporary folder without overwriting existing files:

```bash
python code/run_study.py --output /tmp/week07-rerun/results
```

> [!TIP]
> * `make_figures.py` also refits LightGBM to generate the validation threshold sweep, and `extra_curves.py` refits all available models to plot comparative curves.
> * All plot labels use **DejaVu Sans** and render directly in English without extra font configuration.

### 5.2 Rebuild Report Documents

```bash
python report/source/build_paper.py
python report/source/build_atlas.py
```

> [!NOTE]
> * `build_paper.py` generates `report/thyroid_model_comparison.docx` from the Chinese manuscript source (requires **Noto CJK** fonts installed for proper rendering; convert `.docx` to `.pdf` separately if needed).
> * `build_atlas.py` rebuilds `report/thyroid_analysis_figure_atlas.pdf` directly from the `33` PNG plots in `figures/`.