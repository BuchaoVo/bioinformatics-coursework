<div align="center">

# Week 09 · Ensemble Learning & Bootstrap

**Comparing single trees, Bagging, Random Forests, and Boosting against linear and kernel baselines.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](#6-running-the-experiments)
[![Datasets](https://img.shields.io/badge/Datasets-WDBC_%C2%B7_CTG-0969DA?style=flat-square)](#3-datasets--experimental-setup)
[![Tasks](https://img.shields.io/badge/Worksheet-Parts_A--H-2EA043?style=flat-square)](#4-worksheet-coverage-ah)

[Overview](#1-overview--key-files) •
[Directory Layout](#2-directory-layout) •
[Datasets & Setup](#3-datasets--experimental-setup) •
[Worksheet (A–H)](#4-worksheet-coverage-ah) •
[Key Results](#5-key-results) •
[Running the Experiments](#6-running-the-experiments)

</div>

---

## 1. Overview & Key Files

This folder brings together the coursework, code, figures, and reports for **Week 09: Ensemble Learning and Bootstrap** (originally numbered Lecture 10 in the slide deck and Lecture 9 in the assignment notes).

The goal is to evaluate whether **Bagging, Random Forests, and Boosting** improve predictive accuracy and stability over a **single decision tree** under identical data splits and validation protocols, and to compare them with **Logistic Regression** and **RBF-SVM**. Along the way, the experiments separate training perturbation from metric sampling uncertainty and clarify where each modeling conclusion holds.

### Quick Entry Points

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet & Solutions** | [Original Worksheet (A–H)](resources/teaching/student-worksheet.md) | Prompt sheet and task requirements |
| **Worksheet & Solutions** | [Complete Solutions (A–H)](report/source/worksheet_final.md) | Full write-up and derivations for Parts A–H |
| **Reports & Guides** | [Readable Manuscript](report/source/manuscript_readable.md) | Clean Markdown source of the course paper |
| **Reports & Guides** | [Experiment & Figure Guide](report/source/experiment_design.md) | Experimental protocol, metrics, and plotting specs |
| **Compiled PDFs** | [Course Report (10 pages)](report/submission/week09_signed_report.pdf) | Concise 10-page submission report |
| **Compiled PDFs** | [Full Study Paper (13 pages)](report/generated/week09_ensemble_learning_course_paper.pdf) | Two-dataset course paper generated from source |
| **Compiled PDFs** | [A–H Solutions & Plot Guide (17 pages)](report/generated/week09_worksheet_solutions_and_plot_guide.pdf) | Combined solutions and figure documentation |

---

## 2. Directory Layout

| Path | What's Inside |
| :--- | :--- |
| **`code/`** | Three scripts: main experiments, AdaBoost weight tracking, and plotting |
| **`data/wdbc/student/`**, **`data/ctg/student/`** | Fixed train, validation, and locked-test feature CSVs (6 files total) |
| **`figures/`** | 8 main figures + 1 supplementary figure, each in PDF and PNG formats (18 files total) |
| **`results/wdbc/`**, **`results/ctg/`** | Metrics, sample-level predictions, CV folds, Bootstrap outputs, hyperparameter curves, and protocol configs |
| **`report/source/`** | Markdown sources for the paper and A–H solutions, figure captions, templates, and build scripts |
| **`report/generated/`** | Built PDF and Word versions of the full paper and supplementary guide |
| **`report/submission/`** | 10-page course report PDF |
| **`resources/teaching/`** | Lecture slides and the original worksheet |
| **`environment/`** | Dependencies for experiments and report generation |

---

## 3. Datasets & Experimental Setup

### Dataset Summary

| Dataset | Features | Train | Validation | Locked Test Features | Target & Split Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **WDBC** | `30` | `341` | `114` | `114` | `malignant=1` · Fixed stratified split |
| **CTG** | `21` | `1,279` | `424` | `423` | `pathologic=1` · Grouped by `recording_id` |

* **Data Sources:** WDBC is the [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), DOI `10.24432/C5DW2B`, distributed here through the course's sklearn-based split. CTG is [UCI Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography), DOI `10.24432/C51S4N`.
* **Data Conventions:** Analysis scripts read exclusively from `student/`. `sample_id` is an identifier; CTG's `recording_id` defines groups and is excluded from model features. WDBC uses malignant=1 (reversing sklearn's original target); CTG uses pathologic=1 (original NSP=3). Test feature files contain no labels. Use the supplied fixed CSVs directly.
* **What the Experiments Cover:**
  * **Models:** 7 primary models plus an unpruned single-tree reference baseline.
  * **Validation & Resampling:** Shared 5-fold cross-validation, 30 training Bootstrap resamples, 2,000 validation metric Bootstrap iterations, and a 1,000-draw teaching simulation.
  * **Diagnostics:** Sweeps over tree depth, ensemble size, and learning rate, alongside feature importance and AdaBoost sample-weight tracking across the first 10 boosting rounds.

---

## 4. Worksheet Coverage (A–H)

| Part | Topic | Key Outputs & Supporting Evidence |
| :---: | :--- | :--- |
| **A** | **Decision Trees** | Split paths, Gini impurity, and feature scaling (`tree-depth.csv`, `bootstrap-roots.csv`) |
| **B** | **Voting & Variance** | Worked hard/soft voting examples and variance reduction derivations (Figure 4) |
| **C** | **Bootstrap & OOB** | Resampling simulation and out-of-bag error analysis (`bootstrap-simulation*.csv`) |
| **D** | **Single Tree vs. Bagging & RF** | Model benchmarks, tree count sweeps, error correlation, and feature importance (Figures 3–5, 7) |
| **E** | **Boosting** | Learning rate vs. iteration curves, AdaBoost weight trajectories, and library comparisons (Figure 6, Figure S1) |
| **F** | **Fair Comparison** | In-fold scaling, shared CV splits, metric trade-offs, and model selection |
| **G** | **Stability & Uncertainty** | Prediction flip rates under resampling, metric confidence intervals, and CTG recording-cluster analysis |
| **H** | **Synthesis & Memo** | Bagging vs. Boosting comparison matrix (H1) and research memo (H2) |

---

## 5. Key Results

### WDBC Development Set (5-Fold CV AUC)

| Model | CV AUC | Takeaway |
| :--- | :---: | :--- |
| **Logistic Regression (LR)** | `0.9942` | Ensembles dramatically outperform a single tree (`0.9205`), though simple linear and kernel baselines remain slightly ahead on WDBC. |
| **RBF-SVM** | `0.9928` | Matches LR as a top-performing baseline on WDBC. |
| **Random Forest (RF)** | `0.9893` | Strongest tree-based model on WDBC with substantially lower variance than a single tree. |
| **Decision Tree (Depth 4)** | `0.9205` | Single-tree baseline; sensitive to training-set perturbations. |

### CTG Highlights

* **Random Forest** and **Gradient Boosting** emerge as the strongest tree-ensemble candidates on CTG.
* On the fixed validation split, the confidence interval for the **Average Precision (AP) difference (`RF − LR`)** spans `0`, indicating comparable overall precision-recall performance.

---

## 6. Running the Experiments

### 6.1 Reproduce All Experiments & Figures

From the repository root, navigate to the Week 09 folder and set up a **Python 3.12** environment (exact package versions are pinned in `environment/requirements.txt` and each dataset's `protocol.json`):

```bash
cd assignments/week-09-topic
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt

# Run experiments into separate output folders to keep existing results untouched
python code/run_experiments.py --dataset wdbc --output-root results/reproduced
python code/run_experiments.py --dataset ctg --output-root results/reproduced
python code/trace_adaboost.py --output-root results/reproduced
python code/make_figures.py --results results/reproduced --out figures/reproduced
```

> [!TIP]
> All scripts use local relative paths under `data/` out of the box—no path edits are needed.

### 6.2 Regenerate Figures Only

If you only want to re-render the plots from existing result files:

```bash
python code/make_figures.py --out figures/reproduced
```

### 6.3 Rebuild the Paper & Supplementary Documents

```bash
python report/source/build_documents.py
```

> [!NOTE]
> Building the PDF and Word files into `report/generated/` also requires **Pandoc**, **XeLaTeX**, `python-docx`, and the **Noto Serif CJK SC**, **Noto Sans CJK SC**, and **DejaVu Sans Mono** fonts.
