# Weekly Assignments

This directory houses weekly coursework assignments and computational experiments. Each week lives in its own standalone folder, initialized from a shared template and organized around actual deliverables.

---

## 1. Starting a New Week

1. **Create a dedicated folder** using the `week-XX-topic` naming convention (e.g., `week-01-topic`).
2. **Initialize the main README** by copying [`../templates/assignment-README.md`](../templates/assignment-README.md) into the new directory as its single `README.md`.
3. **Organize deliverables** based on actual outputs, consolidating task objectives, data sources, run commands, and completion status into the main `README.md`.

```bash
mkdir -p week-01-topic
cp ../templates/assignment-README.md week-01-topic/README.md
```

---

## 2. Recommended Directory Layout

Create subdirectories strictly based on **actual contents**—do not add empty folders, placeholder files, or unused notebooks:

```text
week-XX-topic/
├── README.md          # Single root doc (goals, data notes, run commands, and status)
├── code/              # Experiment runners, analysis scripts, and plotting code
├── data/              # Input datasets and field descriptions
├── figures/           # Figures embedded in the report (e.g., PDF, PNG)
├── results/           # Evaluation metrics, CV folds, and resampling outputs
├── report/            # Report sources and compiled deliverables (Markdown / PDF / Word)
├── resources/         # Worksheets, lecture slides, and course handouts
└── environment/       # Pinned dependency specs (e.g., requirements.txt)
```

---

## 3. Maintenance & Submission Standards

| Guideline | Standard |
| :--- | :--- |
| **Language** | Use English for maintained text and new artifacts; preserve original course materials and required reference names. |
| **Single Main `README.md`** | Keep only one `README.md` in the weekly root folder—no subdirectory READMEs or variants. |
| **Content-Driven Structure** | Retain only files actively used by experiments or reports; omit folders that have no contents. |
| **Lean Figures & Artifacts** | Keep only figure formats referenced in the report (`PDF`, `PNG`); skip extra previews or persistent logs. |
| **Reproducibility Essentials** | Preserve random seeds, hyperparameters, dependency versions, data splits, and group-isolation checks. |
| **Honest Status Reporting** | Clearly note completed vs. unverified items in `README.md`—never label "materials complete" as "reproduced" or "accepted." |
