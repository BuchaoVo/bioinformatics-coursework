# Weekly Assignment Maintenance Rules

> [!NOTE]
> **Core Baseline:** Subsequent weeks follow the streamlined standard established in **Week 09**, with all organization and maintenance geared toward completing the coursework experiments and the final course project.

---

## 1. Documentation & Directory Standards

* **Single Main `README.md` Policy**
  Keep only one `README.md` in each week's root directory—do not add subdirectory READMEs or README variants. Consolidate task requirements, data sources, directory descriptions, run commands, and progress status into the main `README.md`, using English and preserving the existing formatting style.
* **English by Default**
  Use English for new or maintained documentation, code comments, Notebook narrative, user-facing messages, model labels, and newly generated plot/table labels. Keep scientific values and required course terminology intact. Apply this rule to future weekly maintenance without automatically rewriting unrelated weeks.
* **English File Naming**
  Use English ASCII filenames for all files, including imported reports, slides, images, datasets, and existing artifacts; prefer descriptive snake_case names. When renaming, update every code path, Notebook reference, document link, and output-generation filename together. Preserve original document/image contents and scientific data; original source language inside those files does not exempt their filenames from this rule. Record any meaningful original labeling discrepancy in the main README.
* **Content-Driven Directories, No Placeholders**
  Retain only the **code, input data, dependencies, numerical results, required figures, reports and sources, lecture slides, and worksheets** needed for the coursework experiments. Create folders strictly based on actual contents—no empty directories, placeholder files, or unused notebooks.

---

## 2. Artifact Lean-Out & Redundancy Control

* **No Boilerplate Metadata or Persistent Logs**
  Do not add hash manifests, provenance mappings, archive manifests, verification reports, completion markers, persistent logs, or extra engineering frameworks. Run quick sanity checks directly in the terminal without saving check files, and ensure scripts do not regenerate these redundant artifacts.
* **Keep Only Report-Used Figure Formats**
  Retain figures only in the formats actually embedded in the report (e.g., `PDF` and `PNG` in Week 09). Do not generate extra `SVG` files, low-resolution previews, or duplicate overview grids by default.

---

## 3. Scientific Rigor & Data Isolation

* **Preserve All Reproducibility Essentials**
  Random seeds, model hyperparameters, required dependency versions, data splits, and outputs needed for scientific analysis must remain intact. **Never alter existing scientific numbers or experimental results** during directory cleanup.
* **Keep Critical Data Validation Checks**
  Data validity and group-isolation checks are essential safeguards for scientific integrity and must not be removed for the sake of brevity.

---

## 4. Cleanup Workflow & Archive Handling

* **Verify Dependencies Before Deleting**
  Remove duplicate files, unrelated reference assignments, instructor solution keys, and any demos, slide-generation scripts, or raw-data reconstruction materials not required by the current experiments. **Before deleting anything, check all code imports/calls, file reads, report build scripts, and document links** to avoid breaking existing workflows.
* **Archive Inspection & Deduplication**
  When handling compressed archives, inspect and categorize their contents first. Once required files are confirmed intact, delete the original archive and remove any nested duplicate archives. Retain a raw data archive only if experiment scripts still read directly from it.

---

## 5. Wrap-Up Checks & Repository Boundaries

* **Sync Paths & State Status Honestly**
  After cleanup, update runtime paths and the main `README.md`, then run proportional entry-point, link, or smoke checks. Do not retrain all models or rebuild final reports merely for directory tidying. **Explicitly note any unverified components—never label "materials complete" as "reproduced," "submitted," or "accepted."**
* **Strict Operational Scope**
  Do not batch-modify other weekly folders while maintaining a given week, and never automatically run `git commit` or `git push`.

---

## Quick Reference Matrix

| Dimension | Keep / Do | Remove / Avoid |
| :--- | :--- | :--- |
| **Documentation** | Single root `README.md`, slides, worksheet, reports & sources | Subdirectory READMEs, README variants, empty placeholder notes |
| **Folders & Files** | Active code, input data, dependencies, numerical outputs | Empty folders, placeholder files, unused notebooks |
| **Figure Formats** | Formats actually used in reports (e.g., `PDF`, `PNG`) | Unused `SVG`s, low-res previews, duplicate overview grids |
| **Metadata & Logs** | Direct CLI execution for quick checks | Hash manifests, provenance maps, verification reports, completion flags, persistent logs |
| **Archives** | Raw data archives read directly by scripts | Original archives after extraction, nested duplicate archives |
| **Reproducibility** | Seeds, hyperparameters, pinned versions, splits, group-isolation checks | Modifying existing scientific numbers, removing data validity checks |
| **Status & Git** | Synced paths/links, clear notes on unverified items | Overstating status (reproduced/accepted), cross-week batch edits, auto-commit/push |
