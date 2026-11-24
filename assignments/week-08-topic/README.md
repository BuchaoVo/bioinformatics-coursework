<div align="center">

# Week 08 · Linear Models, LDA & Logistic Regression

**Comparing discriminative and generative linear classifiers on heart disease data—from probability calibration to cost-sensitive thresholding.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](#6-running-the-analysis)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Heart_(n%3D270)-0969DA?style=flat-square)](#3-dataset--experimental-setup)
[![Tasks](https://img.shields.io/badge/Worksheet-Tasks_1--9_%2B_Follow--ups-2EA043?style=flat-square)](#4-task-coverage)

[Overview & Key Files](#1-overview--key-files) •
[Directory Layout](#2-directory-layout) •
[Dataset & Setup](#3-dataset--experimental-setup) •
[Task Coverage](#4-task-coverage) •
[Key Results](#5-key-results) •
[Running the Analysis](#6-running-the-analysis)

</div>

---

## 1. Overview & Key Files

This folder brings together the code, notebook, figures, and written reports for **Week 08: Linear Models, LDA, and Logistic Regression**.

Using 270 heart disease records, the experiments compare **Linear Discriminant Analysis (LDA)** and **Logistic Regression** in terms of modeling objectives, probability formulation, distributional assumptions, and parameter counts, while evaluating misclassification costs, probability calibration, and coefficient stability.

All filenames use English. Original report and image contents are preserved. The signed PDF retains `original_lecture07` in its English filename because the supplied filename identified Lecture 7, although its contents match this week; verify the course numbering before submission.

### Quick Entry Points

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet & Design** | [Original Worksheet (Tasks 1–9 + 5 Questions)](resources/student-worksheet.md) | Prompt sheet, core tasks, and follow-up questions |
| **Worksheet & Design** | [Experiment Design Guide](resources/experiment-design.md) | Experimental protocol, preprocessing, and evaluation rules |
| **Reports & Solutions** | [Complete Worksheet Solutions (4 pages)](report/week08_linear_models_worksheet_solutions.pdf) | Step-by-step answers and derivations (editable `.docx` in `report/`) |
| **Reports & Solutions** | [Course Research Report (7 pages)](report/week08_heart_disease_linear_models_report.pdf) | Two-column study paper on the heart disease dataset (`.docx` in `report/`) |
| **Reports & Solutions** | [27-Figure Analysis Atlas](report/week08_analysis_figure_atlas.pdf) | Complete visual collection of statistical and model diagnostics |
| **Reports & Solutions** | [Course Report (8 pages)](report/week08_signed_report_original_lecture07.pdf) · [Word](report/week08_signed_report.docx) | Final formatted write-up covering this week's study |
| **Interactive Code** | [Reproducible Analysis Notebook](notebooks/week08_linear_models_lda_logistic.ipynb) | Step-by-step notebook organized into stages A–E |

---

## 2. Directory Layout

| Path | What's Inside |
| :--- | :--- |
| **`code/`** | `run_analysis.py` — end-to-end preprocessing, model benchmarking, statistical tests, and figure generation |
| **`notebooks/`** | Course notebook structured into stages A–D plus summary section E |
| **`data/`** | `heart.csv` (`270` rows × `18` columns) |
| **`results/`** | `13` CSV tables and `3` JSON files covering predictions, metrics, statistical analyses, the locked pre-test plan, and a full summary |
| **`figures/`** | `27` PNG figures used across the reports and atlas |
| **`report/`** | Course paper, worksheet solutions, figure atlas, and editable Word sources (`7` files total) |
| **`resources/`** | Student worksheet and experiment design notes |
| **`environment/`** | Python dependencies for the analysis script and notebook |

---

## 3. Dataset & Experimental Setup

### Dataset Overview

The dataset (`data/heart.csv`) is a course-curated multi-site **UCI Heart Disease** compilation spanning Cleveland, Hungary, Switzerland, and VA Long Beach:

* **Target (`heart_disease_present`):** `120` positive and `150` negative cases.
* **Input Features (`13`):** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`. Identifier and auxiliary fields (`sample_id`, `record_id`, `site`, `num`) are excluded from modeling.
* **Missing Data Handling:** There are `557` raw missing cells, plus `42` records with `chol = 0` recoded as missing, bringing the total to `599` missing cells.

### Modeling & Evaluation Protocol

* **Data Split & CV:** Stratified split (`random_state=2026`) into `216` development and `54` holdout test records. Model comparison on the development set uses **3×5-fold repeated cross-validation**, while decision thresholds are tuned via a single **5-fold out-of-fold (OOF)** run.
* **In-Fold Preprocessing:** Numeric imputation, missingness indicators, standardization, and one-hot encoding are fit strictly inside each training fold. On the full development set, encoding yields `24` features (`25` nominal parameters for Logistic Regression vs. `300` shared covariance parameters for LDA).
* **Candidate Models & Selection Rule:** Evaluates a prevalence baseline, Gaussian Naive Bayes, LDA, **Shrinkage LDA**, **Logistic Regression**, Random Forest, XGBoost, and RBF-SVM. Primary model selection compares Logistic Regression against Shrinkage LDA on development-set misclassification cost (`5 × FN + FP`), breaking ties with the Brier score.
* **Thresholds & Bootstrap:** Probability thresholds are swept over `0.05–0.95` (step `0.01`). Uncertainty intervals use `650` Bootstrap resamples for test metrics and `180` resamples for age-coefficient stability.
* **Saved Experiment Records:** `results/frozen_protocol_before_test.json` locks in the selected model, threshold, and scoring rules prior to test evaluation, while `results/analysis_summary.json` stores all numerical outputs and experimental settings.

---

## 4. Task Coverage

| Task | Topic | Supporting Outputs & Materials |
| :---: | :--- | :--- |
| **Tasks 1–4** | **Model Foundations** | Comparison table, decision boundary derivations, and parameter counts in the worksheet solutions |
| **Tasks 5–6** | **Data Audit & Benchmarking** | Notebook stages A–B, missingness summaries, OOF metrics, and `120` model × fold results |
| **Tasks 7–8** | **Cost, Thresholds & Stability** | Cost-threshold curves, odds ratios/coefficients, age resampling, and covariance diagnostics |
| **Task 9** | **Locked Test & CERB** | Pre-test locked config, `54` test predictions, Bootstrap confidence intervals, and 4-part CERB write-up |
| **Follow-ups** | **5 Review Questions & Exit Ticket** | Complete worked answers at the end of the worksheet solution PDF/Word |

---

## 5. Key Results

### Development-Set Selection & Holdout Test Performance

Following the pre-test decision rule on the development set, **Shrinkage LDA** achieves a lower misclassification cost than Logistic Regression (**82** vs. **83**) and is locked in with a probability threshold of **0.15**:

| Model | Dev Cost (`5×FN+FP`) | Locked Threshold | Test ROC-AUC (95% CI) | Test AP | Test Errors (`FN` / `FP`) | Test Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Shrinkage LDA (Selected)** | **`82`** | **`0.15`** | **`0.8181`** (`0.6973–0.9217`) | **`0.7734`** | `2` / `16` | **`26`** |
| **Logistic Regression** | `83` | — | — | — | — | `17` |

> [!NOTE]
> * **Pre-Test Commitment:** Although Logistic Regression achieves a lower raw cost on the 54-record test split (`17` vs. `26`), the model choice remains fixed to Shrinkage LDA per the pre-test selection protocol.
> * **Scope & Artifacts:** The test set is an internal workflow holdout within `heart.csv` rather than an external clinical cohort. All PDF and Word reports in `report/` are already compiled; `code/run_analysis.py` regenerates the numerical tables (`results/`) and `27` PNG plots (`figures/`).

---

## 6. Running the Analysis

### 6.1 Run the Full Script

From the repository root, navigate to the Week 08 folder, set up a virtual environment, and run the analysis pipeline:

```bash
cd assignments/week-08-topic
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python code/run_analysis.py
```

By default, `run_analysis.py` reads `data/heart.csv` and writes updated tables to `results/` and PNG plots to `figures/`. To test a run without overwriting existing outputs, pass `--out`:

```bash
python code/run_analysis.py --out /tmp/week08-rerun
```

### 6.2 Explore the Interactive Notebook

```bash
jupyter notebook notebooks/week08_linear_models_lda_logistic.ipynb
```

> [!TIP]
> * **Execution Order:** The notebook cells currently have no cached outputs. Run stages **A–C** (data inspection, development comparison, and pre-test locking) before executing stage **D**, which invokes the full pipeline and overwrites `results/` and `figures/`.
> * **Fonts:** New plots use Matplotlib's **DejaVu Sans** font and English labels. Existing images retain their original labels.
