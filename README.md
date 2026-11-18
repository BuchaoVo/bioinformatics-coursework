# Bioinformatics Course

```{=html}
<p align="center">
```
`<em>`{=html}Course Notes · Weekly Assignments · Computational
Experiments · Reports`</em>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## Contents

-   [Overview](#overview)
-   [Repository Structure](#repository-structure)
-   [Directory Guide](#directory-guide)
-   [Course Roadmap](#course-roadmap)
-   [Weekly Assignment Structure](#weekly-assignment-structure)
-   [Getting Started](#getting-started)
-   [Reproducibility](#reproducibility)

## Overview

This repository brings together weekly assignments, analysis notebooks, reusable scripts, figures, experimental results, and course reports.

Materials are organized by course week, with a consistent structure for each assignment. This makes the repository easier to navigate, maintain, reproduce, and review throughout the semester.

## Repository Structure

``` text
bioinformatics-coursework/
├── README.md
├── course-info/
│   ├── syllabus.pdf
│   ├── schedule.md
│   └── assignment-list.md
├── notes/
│   ├── concepts/
│   ├── tools/
│   └── literature/
├── assignments/
│   ├── week-01-topic/
│   │   ├── README.md
│   │   ├── code/
│   │   ├── notebooks/
│   │   ├── data/
│   │   ├── figures/
│   │   ├── results/
│   │   └── report/
│   └── ...
├── projects/
├── resources/
├── templates/
├── environment/
│   └── requirements.txt
└── .github/
    └── workflows/
```

## Directory Guide

  -----------------------------------------------------------------------
  Directory                           Description
  ----------------------------------- -----------------------------------
  `course-info/`                      Course syllabus, weekly schedule, and assignment list

  `notes/concepts/`                   Notes on core bioinformatics concepts and methods

  `notes/tools/`                      Notes on Python, Linux / Shell, and computational tools

  `notes/literature/`                 Research paper summaries and reading notes

  `assignments/`                      Weekly coursework, experiments, and reports

  `projects/`                         Larger projects spanning multiple course topics

  `resources/`                        Useful databases, software, documentation, and references

  `templates/`                        Reusable assignment, report, and notebook templates

  `environment/`                      Dependency specifications and environment setup notes

  -----------------------------------------------------------------------

## Course Roadmap

The table below provides an overview of the course topics. Weeks 10--16 will be updated as the course progresses.

  ------------------------------------------------------------------------
                Week               Topic               Main Deliverables
  -------------------------------- ------------------- -------------------
                 01                Introduction to     Course notes;
                                   Bioinformatics and  environment setup
                                   Research Questions  

                 02                Biomedical Data     Data dictionary;
                                   Types and Data      quality assessment
                                   Quality             

                 03                Linux / Shell and   Shell scripts
                                   Reproducible        
                                   Workflows           

                 04                Python for Data     Jupyter Notebook
                                   Processing and      
                                   Visualization       

                 05                Statistical         Analysis report
                                   Inference and       
                                   Hypothesis Testing  

                 06                Multiple Testing    Statistical
                                   and False Discovery analysis and
                                   Rate (FDR)          results

                 07                High-Dimensional    PCA and related
                                   Data Analysis and   analyses
                                   Dimensionality      
                                   Reduction           

                 08                Linear Models and   Model comparison
                                   Classification      report

                 09                Ensemble Learning   Experiments, code,
                                   and Bootstrap       and report

               10--16              To be updated       Weekly assignments
                                                       and reports
  ------------------------------------------------------------------------

## Weekly Assignment Structure

Each assignment is stored in its own directory, named `week-XX-topic`, and follows a consistent structure:

``` text
week-XX-topic/
├── README.md
├── code/          # Python scripts and Shell commands
├── notebooks/     # Exploratory analysis and visualizations
├── data/          # Dataset notes and, where appropriate, input data
├── figures/       # Figures used in the analysis and report
├── results/       # Tables, metrics, and experiment outputs
└── report/
    ├── report.md
    └── report.pdf
```

Each assignment-level `README.md` should briefly describe:

1.  **Objective** --- the research question and assignment goals.
2.  **Data** --- the dataset and its source.
3.  **Methods** --- the methods and analysis workflow.
4.  **Execution** --- how to run the code.
5.  **Findings** --- the main results and limitations.

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/BuchaoVo/bioinformatics-coursework.git
cd bioinformatics-coursework
```

### 2. Create a virtual environment

**Linux / macOS**

``` bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

### 4. Launch Jupyter Notebook

``` bash
jupyter notebook
```

## Reproducibility

To keep coursework reproducible and maintainable:

-   Prefer relative paths over machine-specific absolute paths.
-   Record relevant package versions and random seeds.
-   Keep reusable analysis logic in `code/`; use notebooks for exploration and visualization.
-   Save figures and tables in the corresponding `figures/` and `results/` directories.
-   Keep the Markdown report and its PDF version together in `report/`.
