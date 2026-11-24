<div align="center">

# Bioinformatics Coursework

**Weekly assignments, reproducible analysis workflows, computational experiments, and reports.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Shell](https://img.shields.io/badge/Shell-Bash%20%7C%20PowerShell-4EAA25?style=flat-square&logo=gnu-bash&logoColor=white)](#getting-started)
[![Course Status](https://img.shields.io/badge/Progress-Weeks%2001--09%20Planned-0969DA?style=flat-square)](#course-roadmap)

[Overview](#overview) •
[Repository Structure](#repository-structure) •
[Directory Guide](#directory-guide) •
[Course Roadmap](#course-roadmap) •
[Weekly Assignment Structure](#weekly-assignment-structure) •
[Getting Started](#getting-started) •
[Reproducibility](#reproducibility)

</div>

---

## Overview

This repository consolidates weekly coursework, exploratory analysis notebooks, reusable scripts, publication-ready figures, experimental outputs, and written reports for the **Bioinformatics** course.

All materials are organized chronologically by course week under a standardized directory convention, ensuring the codebase remains easy to navigate, maintain, reproduce, and review throughout the semester.

---

## Repository Structure

```text
bioinformatics-coursework/
├── README.md
├── course-info/                 # Syllabus, schedule, and assignment index
│   ├── syllabus.pdf
│   ├── schedule.md
│   └── assignment-list.md
├── notes/                       # Structured course and literature notes
│   ├── concepts/
│   ├── tools/
│   └── literature/
├── assignments/                 # Weekly coursework (standardized layout)
│   ├── week-01-topic/
│   │   ├── README.md
│   │   ├── code/
│   │   ├── notebooks/
│   │   ├── data/
│   │   ├── figures/
│   │   ├── results/
│   │   └── report/
│   └── ...
├── projects/                    # Multi-topic integrative course projects
├── resources/                   # Curated databases, docs, and references
├── templates/                   # Reusable report and notebook templates
├── environment/                 # Dependency specs and setup guides
│   └── requirements.txt
└── .github/
    └── workflows/               # Automated CI and repository checks
```

---

## Directory Guide

| Directory | Description |
| :--- | :--- |
| **`course-info/`** | Course syllabus, weekly schedule, and master assignment list. |
| **`notes/concepts/`** | Notes on core bioinformatics concepts, algorithms, and statistical methods. |
| **`notes/tools/`** | Practical guides for Python, Linux / Shell, and bioinformatics CLI tools. |
| **`notes/literature/`** | Research paper summaries, critical reviews, and reading notes. |
| **`assignments/`** | Self-contained weekly coursework, computational experiments, and reports. |
| **`projects/`** | Extended projects integrating multiple course topics and workflows. |
| **`resources/`** | External biological databases, software documentation, and reference links. |
| **`templates/`** | Standardized templates for assignment `README`s, reports, and notebooks. |
| **`environment/`** | Python dependency specifications (`requirements.txt`) and setup notes. |
| **`.github/workflows/`** | Automated repository validation and continuous integration checks. |

---

## Course Roadmap

The roadmap below outlines the planned weekly topics and primary deliverables. **Weeks 10–16** will be updated incrementally as the semester progresses.

### Module 1 · Foundations (`Weeks 01–04`)

| Week | Topic | Primary Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **01** | Introduction to Bioinformatics and Research Questions | Course notes, environment setup | `Active` |
| **02** | Biomedical Data Types and Data Quality | Data dictionary, quality assessment | `Active` |
| **03** | Linux / Shell and Reproducible Workflows | Reproducible Shell scripts | `Active` |
| **04** | Python for Data Processing and Visualization | Exploratory Jupyter Notebook | `Active` |

### Module 2 · Statistical and Computational Methods (`Weeks 05–08`)

| Week | Topic | Primary Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **05** | Statistical Inference and Hypothesis Testing | Statistical analysis report | `Active` |
| **06** | Multiple Testing and False Discovery Rate (FDR) | Multiple-testing analysis & results | `Active` |
| **07** | Thyroid Diagnostic Codes and Model Comparison | [Code, data, results, and reports](assignments/week-07-topic/README.md) | `Materials organized; full rerun and submission pending` |
| **08** | Linear Models and Classification | Model comparison report | `Active` |

### Module 3 · Advanced Topics and Course Assignments (`Weeks 09–16`)

| Week | Topic | Primary Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **09** | Ensemble Learning and Bootstrap | [Code, data, results, and reports](assignments/week-09-topic/README.md) | `Materials archived; report reconciliation pending` |
| **10–16** | *Topics to be announced* | Weekly assignments, experiments, and reports | `Upcoming` |

---

## Weekly Assignment Structure

Each weekly assignment resides in its own directory named `week-XX-topic/` and adheres to a uniform layout:

```text
week-XX-topic/
├── README.md          # Assignment overview, execution steps, and key findings
├── code/              # Reusable Python scripts and Shell pipelines
├── notebooks/         # Exploratory data analysis and visualization notebooks
├── data/              # Dataset documentation and (where appropriate) input data
├── figures/           # Generated plots used in analysis and reports
├── results/           # Summary tables, evaluation metrics, and model outputs
└── report/
    ├── report.md      # Source Markdown report
    └── report.pdf     # Compiled PDF deliverable
```

> [!IMPORTANT]
> **Assignment-Level `README.md` Specification**
> Every `week-XX-topic/README.md` must briefly document five core sections:
> 1. **Objective** — Research question, biological/statistical context, and assignment goals.
> 2. **Data** — Dataset description, preprocessing notes, and original source.
> 3. **Methods** — Analytical methods, algorithms, and workflow pipeline.
> 4. **Execution** — Exact commands required to reproduce the analysis.
> 5. **Findings** — Summary of key results, interpretations, and limitations.

---

## Getting Started

### 1. Clone the repository

```bash
git clone [https://github.com/BuchaoVo/bioinformatics-coursework.git](https://github.com/BuchaoVo/bioinformatics-coursework.git)
cd bioinformatics-coursework
```

### 2. Create and activate a virtual environment

* **Linux / macOS**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

* **Windows PowerShell**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Reproducibility

To ensure all coursework remains verifiable and easy to rerun across environments:

- **Relative Paths Only** — Always use repository-relative paths rather than machine-specific absolute paths.
- **Pinned Environments & Seeds** — Record relevant package versions and fix random seeds for stochastic experiments.
- **Separation of Logic & Exploration** — Place reusable functions and pipelines in `code/`; keep `notebooks/` focused on exploration and visualization.
- **Structured Outputs** — Export all generated plots to `figures/` and numerical tables/metrics to `results/`.
- **Synchronized Reports** — Maintain both the Markdown source (`report.md`) and compiled document (`report.pdf`) inside `report/`.

---

<div align="center">
<sub>Maintained by <a href="https://github.com/BuchaoVo">BuchaoVo</a> · <a href="#bioinformatics-coursework">Back to Top ↑</a></sub>
</div>
