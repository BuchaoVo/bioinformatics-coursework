# Lecture 8: Problem-Based Experiment Design

## Core Design

Treat LDA and logistic regression as two distinct probabilistic modeling approaches. Students first complete a modeling-target–parameters–objective table, then run both methods on the same dataset. Interpret every result through four questions: what does the model assume, how many parameters does it estimate, where do its probabilities come from, and when do these differences affect method selection?

| Core concept | Activity | Output |
|---|---|---|
| Modeling target | Compare `P(Y|X)` with `P(X|Y)P(Y)` | Two probability-generation diagrams |
| Optimization/estimation objective | Write logistic cross-entropy and the LDA class-conditional likelihood | Objective comparison |
| Parameter counts | Count logistic coefficients and LDA means/covariances after one-hot encoding | Parameter-count table |
| Linear boundaries | Explain the sigmoid and cancellation of LDA quadratic terms | Boundary derivation |
| Method selection | Use assumption checks, sample size, probability objectives, and error costs to justify selection | Conditional selection rule |
| Predictive evaluation | Compare AUC, calibration, FN/FP, and cost | Development evidence table |
| Interpretation stability | Use bootstrap and a near-duplicate variable | Coefficient versus prediction stability comparison |

## Schedule

First class, 45 minutes: driving question and tasks 1–4, focusing on modeling targets, parameters, and assumptions. Independent work, 60–90 minutes: Notebook A–C, including leakage prevention, development comparisons, thresholds, and stability. Second class, 45 minutes: tasks 7–9; open the locked test after submitting the frozen protocol, then use CERB to explain the conditions and limits of the method choice.

## Fixed Protocol

Use 270 records: 120 positive and 150 negative, with 13 mixed-type inputs. Use a stratified 80/20 split with `random_state=2026`. Fit continuous-variable scaling and categorical one-hot encoding within training folds. The core models are a prevalence baseline, Gaussian NB, LDA, and logistic regression. Error cost is `5×FN+FP`. Use bootstrap intervals for test metrics and coefficients.

Do not select methods using a single score alone. Examine modeling targets and assumptions, parameter counts relative to sample size, ROC-AUC/PR-AUC, Brier/log loss, fixed-threshold FN/FP, calibration, and coefficient stability. Open the test set only after freezing the protocol.

## Interpretation Limits

The experiment supports method-selection arguments under the stated data protocol and objectives. It does not establish causal effects, clinical deployment readiness, or external validity across hospitals. Coefficients represent conditional associations affected by units, reference levels, collinearity, and sample size.
