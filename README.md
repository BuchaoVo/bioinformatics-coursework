# Bioinformatics Course

## Overview

This repository organizes the learning materials and practical work
completed throughout the Bioinformatics course. It brings together
weekly assignments, analysis notebooks, reusable scripts, figures,
results, and written reports.

The repository is organized by course week. Each assignment follows a
consistent structure so that the analysis process is easy to navigate,
reproduce, and review.

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

### Directory Guide

  -----------------------------------------------------------------------
  Directory                           Contents
  ----------------------------------- -----------------------------------
  `course-info/`                      Course syllabus, weekly schedule,
                                      and assignment list

  `notes/concepts/`                   Notes on core bioinformatics
                                      concepts and methods

  `notes/tools/`                      Notes on Python, Linux / Shell, and
                                      computational tools

  `notes/literature/`                 Research paper summaries and
                                      reading notes

  `assignments/`                      Weekly coursework, experiments, and
                                      reports

  `projects/`                         Larger projects that span multiple
                                      course topics

  `resources/`                        Useful databases, software,
                                      documentation, and references

  `templates/`                        Reusable assignment, report, and
                                      notebook templates

  `environment/`                      Dependency specifications and
                                      environment setup notes

  `.github/workflows/`                Automated repository checks
  -----------------------------------------------------------------------

## Course Roadmap

The following table provides an overview of the course topics. Weeks
10--16 will be updated as the course progresses.

  ------------------------------------------------------------------------
                          Week Topic                 Main Deliverables
  ---------------------------- --------------------- ---------------------
                            01 Introduction to       Course notes,
                               Bioinformatics and    environment setup
                               Research Questions    

                            02 Biomedical Data Types Data dictionary,
                               and Data Quality      quality assessment

                            03 Linux / Shell and     Shell scripts
                               Reproducible          
                               Workflows             

                            04 Python for Data       Jupyter Notebook
                               Processing and        
                               Visualization         

                            05 Statistical Inference Analysis report
                               and Hypothesis        
                               Testing               

                            06 Multiple Testing and  Statistical analysis
                               False Discovery Rate  and results
                               (FDR)                 

                            07 High-Dimensional Data PCA and related
                               Analysis and          analyses
                               Dimensionality        
                               Reduction             

                            08 Linear Models and     Model comparison
                               Classification        report

                            09 Ensemble Learning and Experiments, code,
                               Bootstrap             and report

                        10--16 To be updated         Weekly assignments
                                                     and reports
  ------------------------------------------------------------------------

## Weekly Assignment Structure

Each assignment is stored in its own directory, named `week-XX-topic`,
and follows the structure below:

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

Each assignment `README.md` should briefly describe:

-   The research question and assignment objectives
-   The dataset and its source
-   The methods and analysis workflow
-   How to run the code
-   The main findings and limitations

## Getting Started

Clone the repository:

``` bash
git clone https://github.com/BuchaoVo/bioinformatics-coursework.git
cd bioinformatics-coursework
```

Create and activate a Python virtual environment.

**Linux / macOS:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

``` bash
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

Launch Jupyter Notebook:

``` bash
jupyter notebook
```

## Reproducibility

To make coursework easier to reproduce and maintain:

-   Prefer relative paths over machine-specific absolute paths.
-   Record relevant package versions and random seeds.
-   Keep reusable analysis logic in `code/` and use notebooks for
    exploration and visualization.
-   Save figures, tables, and other analysis outputs in the
    corresponding `figures/` and `results/` directories.
-   Keep the Markdown report and its PDF version together in `report/`.

