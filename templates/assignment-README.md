<div align="center">

# Week XX · Assignment Title

**One-line summary of the scientific question, dataset, and analytical workflow.**

[![Course](https://img.shields.io/badge/Course-Bioinformatics-0969DA?style=flat-square)](#1-objective)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#5-reproducibility)
[![Status](https://img.shields.io/badge/Status-In_Progress-D97706?style=flat-square)](#8-deliverables)

[Objective](#1-objective) •
[Background](#2-background) •
[Data](#3-data) •
[Methods](#4-methods) •
[Reproducibility](#5-reproducibility) •
[Results](#6-results) •
[Discussion](#7-discussion) •
[Deliverables](#8-deliverables)

</div>

---

## 1. Objective

State the core scientific, biological, or computational question this assignment addresses, along with the specific analytical goals or hypotheses being tested.

---

## 2. Background

* **Biological Context:** Summarize the biological system, clinical problem, or experimental assay motivating the analysis.
* **Methodological Context:** Explain the statistical or machine learning methods used and why they fit the data structure and research question.

---

## 3. Data

| Attribute | Details |
| :--- | :--- |
| **Source** | Dataset name and repository / database link |
| **Version / DOI** | Release version, accession number, or DOI |
| **License** | Data usage license (e.g., CC BY 4.0, Public Domain) |
| **Unit of Observation** | Sample, patient, gene, sequence, or recording unit ($N = \text{XX}$) |
| **Features / Labels** | Predictor variables ($p = \text{XX}$), target definition, and class balance |
| **Inclusion & Exclusion Criteria** | Filtering rules, subgroup definitions, and excluded identifier columns |

---

## 4. Methods

* **Data Cleaning & Quality Control:** Describe missing-value handling, outlier or anomaly checks, and duplicate filtering.
* **Preprocessing & Data Splitting:** Specify train/validation/test split ratios, stratification or grouping constraints, and fold-local transformations (imputation, scaling, encoding).
* **Algorithms & Parameter Settings:** List baseline and candidate models, key hyperparameters, and model/threshold selection rules.
* **Statistical & Evaluation Methods:** Define primary and secondary evaluation metrics, cross-validation scheme, uncertainty estimation (e.g., bootstrap intervals), and hypothesis tests with multiple-testing corrections.

---

## 5. Reproducibility

### Environment Specs

* **Python:** `3.10+`
* **Random Seed:** `SEED = XXXX`
* **Dependencies:** `environment/requirements.txt`

### Run Commands

```bash
# Commands to run from the repository root
cd assignments/week-XX-topic
python -m pip install -r environment/requirements.txt
python code/run_analysis.py
```

---

## 6. Results

Summarize the primary quantitative findings and key comparisons below.

| Model / Condition | Validation Metric | Holdout Test Metric (95% CI) | Notes |
| :--- | :---: | :---: | :--- |
| Baseline | `0.000` | `0.000 (0.000–0.000)` | Reference baseline |
| Method A | `0.000` | `0.000 (0.000–0.000)` | Key comparison |

* **Key Figures:** Link main plots in `figures/` (e.g., [`figures/fig01_overview.png`](figures/fig01_overview.png)).
* **Saved Tables & Metrics:** Link numerical outputs in `results/` (e.g., [`results/metrics_summary.csv`](results/metrics_summary.csv)).

---

## 7. Discussion

* **Interpretation:** Explain what the findings mean in light of the research question and domain expectations.
* **Scope & Generalizability:** Clarify the conditions under which these conclusions hold (internal vs. external validation).
* **Biases & Limitations:** Note data missingness, sample size constraints, potential confounders, or computational trade-offs.

---

## 8. Deliverables

- [ ] **Code** (`code/`)
- [ ] **Notebook** (`notebooks/`)
- [ ] **Markdown report** (`report/report.md`)
- [ ] **PDF report** (`report/report.pdf`)
- [ ] **Figures / tables** (`figures/`, `results/`)