# BACKD00R Algorithmic Framework Documentation

This file cross-references Chapter 3, Section 3.4, and the updated BACKD00R
model training specification against the current Python backend under
`Models/backd00r_ai`.

## Implementation Verification Summary

| Section 3.4 item | Match status | Notes |
|---|---:|---|
| Equation 1: Unified Feature Vector | Implemented | The backend uses the canonical 29-feature vector. |
| Equations 2-3: Ensemble Detection and Accuracy Matrix | Implemented as Random Forest-based MoE-inspired expert ensemble with contextual routing | Three 3-class experts, one 3-class gate, weighted-sum aggregation. |
| Equation 4: Refactoring Impact Simulation | Not implemented in Python | No Python module simulates refactoring deltas. |
| Equation 5: WMCA | Not implemented in Python | The model exports `max(P_final)` confidence for WMCA. |
| Equations 6-8: DCAS sequencing and cycle breaking | Not implemented in Python | No Python dependency graph/DCAS sequencer exists. |
| Min-Max normalization | Implemented | Split first, fit training bounds only, normalize all folds. |
| SMOTE | Implemented | Training fold only; target count is the negative/no-smell majority count. |

## Equation 1. Unified Feature Vector

**Algorithm / Concept:** Unified 29-feature vector mapping.

```math
v_n =
[
wmc, cbo, lcom, rfc, dit, loc, cc\_max, nom, nof,
cida, coa,
method\_loc\_max, method\_loc\_avg, method\_param\_max,
foreign\_method\_calls, foreign\_field\_accesses, local\_field\_accesses,
accessed\_foreign\_class\_count, envy\_method\_ratio,
jdeodorant\_signal, iplasma\_signal, designite\_signal, jspirit\_signal,
code\_churn, volatility, bug\_fix\_ratio, author\_count,
hotspot\_proxy, evolution\_intensity
]
```

**Implementation location:**

- `Models/backd00r_ai/configs/feature_schema.py`
  - `FEATURE_NAMES_29`
  - `FeatureSchema.validate_vector`
- `Models/backd00r_ai/extraction/feature_vector_builder.py`
  - `FeatureVector`
  - `FeatureVectorBuilder.build`
- `Models/backd00r_ai/dataset/extract_training_features.py`
  - `OUTPUT_COLUMNS`
  - `migrate_legacy_training_features`
  - `run_batched_extraction`
  - `process_commit_batch`
  - `process_file_rows`
  - `append_training_rows`

## Raw Metric Extraction Logic

**Class-level and method-level metrics:**

```text
wmc = sum(method cyclomatic complexity)
cbo = count(distinct project-local external classes referenced)
rfc = own methods + project-local external method calls
method_loc_avg = sum(method LOC) / max(1, nom)
envy_method_ratio = methods where foreign accesses > local accesses / max(1, nom)
```

**Access metrics:**

```math
cida =
\frac{direct\ internal\ field\ accesses}
{max(1,\ direct\ internal\ field\ accesses + internal\ accessor\ calls)}
```

```math
coa =
\frac{project\text{-}local\ external\ member\ accesses}
{max(1,\ project\text{-}local\ external\ member\ accesses + local\ member\ accesses)}
```

Foreign access counts exclude JDK, IntelliJ, Kotlin, and external dependency
types where they can be identified from the source-level heuristic parser.

**Implementation location:**

- `Models/backd00r_ai/extraction/java_parser.py`
- `Models/backd00r_ai/extraction/structural_metrics.py`
- `Models/backd00r_ai/extraction/access_metrics.py`
- `Models/backd00r_ai/extraction/git_history_miner.py`

## Detector-Inspired Signals

Detector-inspired signals are computed after normalization. Safe helpers:

```math
safe\_div(a,b) = \frac{a}{max(1,b)}
```

```math
clamp(x) = min(1, max(0, x))
```

Base scores:

```math
god\_class\_base =
clamp(
0.25wmc_n + 0.20cbo_n + 0.20lcom_n +
0.15loc_n + 0.10nom_n + 0.10nof_n
)
```

```math
long\_method\_base =
clamp(
0.40method\_loc\_max_n +
0.30cc\_max_n +
0.20method\_param\_max_n +
0.10method\_loc\_avg_n
)
```

```math
foreign\_access\_ratio =
\frac{foreign\_method\_calls + foreign\_field\_accesses}
{max(1,\ foreign\_method\_calls + foreign\_field\_accesses + local\_field\_accesses)}
```

```math
feature\_envy\_base =
clamp(
0.35foreign\_access\_ratio +
0.20accessed\_foreign\_class\_count_n +
0.20envy\_method\_ratio +
0.15coa +
0.10cbo_n
)
```

Each detector feature is:

```math
detector\_signal =
max(god\_class\_score, feature\_envy\_score, long\_method\_score)
```

**Implementation location:**

- `Models/backd00r_ai/extraction/expert_signals.py`
  - `ExpertSignalExtractor.base_scores`
  - `ExpertSignalExtractor.jdeodorant`
  - `ExpertSignalExtractor.iplasma`
  - `ExpertSignalExtractor.designite`
  - `ExpertSignalExtractor.jspirit`
- `Models/backd00r_ai/training/preprocess.py`
  - `add_post_normalization_features`

## Equations 2 and 3. Ensemble Detection and Accuracy Matrix

**Algorithm / Concept:** Random Forest-based MoE-inspired expert ensemble with
contextual routing.

Three experts output 3-class distributions:

```math
E_{god}(x), E_{envy}(x), E_{long}(x) \in \mathbb{R}^{3}
```

```text
[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]
```

Specialization weights:

```math
w_i =
\begin{cases}
3.0, & label_i = target\_label \\
1.0, & label_i \ne target\_label
\end{cases}
```

The gate produces:

```math
G(x) = [w_{god}, w_{envy}, w_{long}]
```

Final aggregation:

```math
P_{final} =
normalize(
w_{god}E_{god}(x)
+ w_{envy}E_{envy}(x)
+ w_{long}E_{long}(x)
)
```

```math
confidence = max(P_{final})
```

**Implementation location:**

- `Models/backd00r_ai/models/smell_experts.py`
  - `SmellExpert.fit`
  - `SmellExpert.predict_distribution`
- `Models/backd00r_ai/training/train_experts.py`
  - `_specialization_weights`
  - `train`
- `Models/backd00r_ai/models/moe_contextual_gate.py`
  - `MoE_ContextualGate.fit`
  - `MoE_ContextualGate.predict_proba`
- `Models/backd00r_ai/models/aggregation.py`
  - `ProbabilityAwareAggregator.aggregate`

## Equation 4. Refactoring Impact Simulation

```math
\Delta_n =
\sum_{i=1}^{m}
(M_{current,i} - M_{simulated,i})
```

**Implementation location:** Not implemented in the current Python backend.

## Equation 5. Weighted Multi-Criteria Analysis (WMCA)

```math
P_{total} =
\left[
\left(
\beta
\left(
\alpha \frac{E_n}{E_{max}}
+ (1-\alpha)\frac{H_n}{H_{max}}
\right)
+ (1-\beta)\frac{CM_n}{CM_{max}}
\right)
\times
\frac{CL_n}{CL_{max}}
\times
CS
\right]
+ w_1\frac{SR_n}{SR_{max}}
+ w_2\frac{\Delta_n}{\Delta_{max}}
```

The model layer provides:

```math
CS = max(P_{final})
```

**Implementation location:** WMCA itself is not implemented in the current
Python backend.

## Equation 6. Weighted Candidate Priority

```math
P_{total} =
\sum_{i=1}^{n}
w_i x_i
```

**Implementation location:** Not implemented in the current Python backend.

## Equations 7 and 8. DCAS Sequencing and Cycle Breaking

Same-level dependency logic:

```math
L(v_i) = L(v_j)
```

Cycle-breaking priority difference:

```math
\Delta P_{total}
=
\left|
P_{total}(v_i) - P_{total}(v_j)
\right|
```

**Implementation location:** Not implemented in the current Python backend.

## Min-Max Normalization

```math
x' =
\frac{x - x_{min}}{x_{max} - x_{min}}
```

Bounds are fitted strictly on the training fold:

```math
x_{min}, x_{max} = fit(X_{train})
```

Implementation details:

```text
1. Split by project.
2. Fit Min-Max bounds on training set only.
3. Normalize train/validation/test using training bounds.
4. Compute detector-inspired signals using normalized metric values.
5. Apply SMOTE only to training set.
6. Save feature_bounds.json.
```

**Implementation location:**

- `Models/backd00r_ai/training/preprocess.py`
  - `split_70_15_15`
  - `fit_minmax_bounds`
  - `apply_minmax`
  - `add_post_normalization_features`
  - `export_bounds`

## SMOTE Balancing Engine

```math
x_{new} = x_i + \lambda(x_{nn} - x_i)
```

```math
\lambda \sim U(0,1)
```

BACKD00R applies SMOTE only after splitting and training-fold normalization.
Each supported smell class is oversampled up to the absolute negative/no-smell
majority class size. Negative rows are preserved unchanged; validation and test
folds are never oversampled.

**Implementation location:**

- `Models/backd00r_ai/training/preprocess.py`
  - `smote_sampling_strategy`
  - `apply_smote_to_training_fold`

## Current Python-Only Coverage

Implemented:

```text
MLCQ parsing
label normalization
repository cloning/caching
raw Java/Git metric extraction
raw metrics cache
project-grouped 70/15/15 split with fallback
training-fold-only Min-Max normalization
detector-inspired signals from normalized metrics
training-fold-only SMOTE
three weighted 3-class Random Forest experts
single 3-class MoE_ContextualGate
weighted-sum MoE aggregation
formal evaluation artifacts
ONNX export
```

Not implemented in Python:

```text
refactoring impact simulation
WMCA final priority scoring
dependency graph construction
DCAS sequencing
DCAS deterministic cycle breaking
```
