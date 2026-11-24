# Lecture 8: Linear Models, LDA, and Logistic Regression

## Driving Question

LDA and logistic regression can both produce linear classification boundaries and probabilities. How do their **modeling targets, probability sources, assumptions, optimization objectives, and parameter counts** differ? For a small heart-disease dataset with mixed variable types, how should research objectives and data conditions guide method selection?

Build a chain of reasoning: explain each model's assumptions and estimates, then examine how these differences affect probabilities, boundaries, stability, and error costs. Write an expectation before each task and revise it using Notebook evidence. Do not open the locked test early or interpret model coefficients directly as causal effects.

## Data and Protocol

Use `data/heart.csv`: 270 records, 120 positive and 150 negative. There are 13 mixed-type inputs; the outcome is `heart_disease_present`. Stratify by the target and split 80% development / 20% locked test with `random_state=2026`. Compare methods using stratified cross-validation within development data. Fit all scaling, one-hot encoding, and models within training folds. Error cost is `5×FN+FP`.

## Task 1: Modeling Targets, Parameters, and Objectives

Before running models, complete:

| Question | Logistic regression | LDA |
|---|---|---|
| What is modeled directly? | ____ | ____ |
| What are the main parameters? | ____ | ____ |
| Which data relationships determine those estimates? | ____ | ____ |
| What is the optimization or estimation objective? | ____ | ____ |
| Where do the output probabilities come from? | ____ | ____ |

Draw two flows: `x → z → sigmoid → P(Y=1|x)` and `x → class-conditional densities and priors → Bayes normalization → P(Y=k|x)`. Mark shared and distinct components. State your current selection rule in one sentence: under which data conditions would you favor each method?

## Task 2: Shared Linear Scores and Boundaries

Explain each symbol in `z=β0+xᵀβ`. Explain how `z=c` defines a linear decision boundary.

- The logistic linear score is an intermediate quantity for conditional probability; the LDA score discriminates between generative models. What follows from this distinction?
- Why must comparisons of coefficient magnitudes specify units, scaling, categorical reference levels, and other covariates included in the model?
- If only the probability threshold changes, how does the boundary move? Do model parameters change?

## Task 3: Logistic Regression Objectives and Parameter Counts

Explain the sigmoid and logit: `p=1/(1+e^−z)` and `log[p/(1−p)]=z`. Explain how binary cross-entropy estimates `β0,β`: specify the parameters sought and the total loss minimized, rather than simply saying “call fit.”

For the 13 inputs, estimate the logistic parameter count after one-hot encoding. Explain how categorical reference levels affect the count and interpretation.

Interpret `OR=e^βj` for a coefficient `βj`, specifying the modeled unit, reference level, and that other covariates are held fixed. Explain why an OR is neither a probability multiplier nor automatically a causal effect.

## Task 4: LDA Assumptions and Parameter Counts

Starting from `X|Y=k ~ N(μk,Σ)`, explain class means `μk`, shared covariance `Σ`, and priors `πk`. Explain how LDA obtains posteriors through estimated class-conditional distributions, priors, and Bayes' rule.

- For two classes, how many mean, shared covariance, and independent prior parameters must be estimated? How does the covariance count grow with the encoded dimension?
- Why does a shared covariance cancel `xᵀΣ⁻¹x` when comparing classes, leaving a linear boundary?
- How would class-specific covariances affect the boundary, parameter count, and variance?
- Can LDA output posterior probabilities? How does their source differ from logistic probabilities?

## Task 5: Data Audit and Leakage Prevention

Run Notebook A. Record variable types, missing values, class proportions, and the development/test split. Identify columns that must be excluded from predictors and explain why.

Draw the fold-local workflow: fit preprocessing on each training fold, transform its validation fold, estimate model parameters from training data, and evaluate predictions. Explain why scaling the full dataset before CV still uses validation information.

## Task 6: Compare the Consequences of Model Assumptions

Run Notebook B and compare the prevalence baseline, Gaussian NB, LDA, and logistic regression. Record:

| Dimension | What to examine | Evidence |
|---|---|---|
| Probability source | Generative densities/priors or a conditional-probability loss? | ____ |
| Parameter scale | Means, covariances, coefficients, and encoded dimension | ____ |
| Assumption checks | Within-class normality, shared covariance, linear log-odds | ____ |
| Optimization objective | Cross-entropy, likelihood, or another estimation objective | ____ |
| Predictive consequences | AUC, Brier, log loss, and fixed-threshold errors | ____ |

If LDA has slightly higher AUC, what further evidence would justify selecting it for this task? If logistic regression has slightly higher AUC, does that make LDA's generative assumptions irrelevant?

## Task 7: Threshold Selection Does Not Replace Model Selection

For prespecified candidate models, compare thresholds using development probabilities. Select thresholds by `5×FN+FP`, then report sensitivity, specificity, precision, and confusion matrices.

Distinguish how a model produces probabilities/rankings from how task-specific costs turn probabilities into actions. Explain why changing a threshold affects FN/FP but not model parameters, AUC, or raw probabilities. Explain why trying multiple thresholds on the locked test compromises its final evaluation role.

## Task 8: Coefficient Stability, Near-Duplicate Variables, and Assumption Checks

Record logistic coefficients and ORs; give one interpretation with all necessary conditions. Add a near-duplicate age variable and compare predictive metrics with bootstrap coefficient distributions.

Examine LDA's within-class distributions, covariance differences, and a skewness stress test. Does stable prediction imply stable individual coefficients? When assumptions are imperfect, which predictive and calibration evidence should guide judgment? How are parameter count, sample size, and covariance estimation related?

## Task 9: Freeze the Protocol, Open the Locked Test, and Write CERB

Submit the fixed model, preprocessing, threshold, error cost, and evaluation metrics before opening Notebook D. Record test metrics and bootstrap intervals without changing the protocol.

Write 4–6 Claim–Evidence–Reasoning–Boundary paragraphs addressing:

1. Which method has stronger justification given the objective, sample size, variable types, and assumption checks? Explain if evidence is insufficient to rule out the alternative.
2. How does the evidence support separate judgments about discrimination, calibration, and error cost?
3. Under which conditions is an OR, LDA posterior, or model difference interpretable?
4. Which clinical or causal conclusions cannot follow from these internal results? What independent evidence is needed next?

## Homework

Answer in your own words:

1. What are the modeling targets, probability sources, and optimization/estimation objectives of LDA and logistic regression?
2. Which parameters cause LDA's parameter count to grow rapidly after one-hot encoding, and what does this imply for small samples?
3. Why does OR=2 not mean double the disease probability?
4. Why are AUC, calibration, and fixed-threshold errors not interchangeable?
5. Give a method-selection rule based on assumptions and objectives, rather than the highest score alone.

Submit the Notebook, comparison table, parameter-count calculations, CERB, and exit card. Experiment entry points are `code/run_analysis.py` and `notebooks/`; numerical outputs are in `results/`, and plots are in `figures/`.
