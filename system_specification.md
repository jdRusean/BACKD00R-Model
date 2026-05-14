# BACKD00R System Model Technical Specification

This specification documents the current BACKD00R Python backend under
`Models/backd00r_ai` and aligns it with the updated model training
specification.

## 1. System Architecture Overview

BACKD00R uses a dual-pipeline design:

| Pipeline | Role |
|---|---|
| Model Training Pipeline | Converts MLCQ rows and repository snapshots into normalized 29-feature rows, trains Random Forest experts and the contextual gate, and exports Python/ONNX artifacts. |
| Plugin Execution Pipeline | Extracts the same 29 features inside the IDE, applies exported bounds, runs ONNX experts/gate, aggregates probabilities, and passes confidence to WMCA/DCAS/UI layers. |

BACKD00R uses a **29-feature unified vector composed of class-level structural
metrics, method-level summary metrics, detector-inspired smell signals, and
evolutionary metrics**.

### 1.1 Model Training Pipeline

| Stage | Description | Implementation |
|---|---|---|
| Dataset parsing | Reads MLCQ and keeps `GOD_CLASS`, `FEATURE_ENVY`, `LONG_METHOD`. | `Models/backd00r_ai/dataset/dataset_builder.py` |
| Repository mining | Clones/caches GitHub repositories and checks out exact commits. | `Models/backd00r_ai/dataset/extract_training_features.py` |
| Raw metric extraction | Groups rows by repository and commit, checks out each commit once, parses each target Java file once per batch, and extracts class, method/access, and Git metrics. | `Models/backd00r_ai/dataset/extract_training_features.py`, `Models/backd00r_ai/extraction/*.py` |
| Raw cache | Caches raw metrics only in `extracted_raw_metrics_cache.csv`, keyed by project/repository/commit/file/class/source/schema/extractor identity. | `extract_training_features.py` |
| Preprocessing | Splits by project, fits train-only Min-Max bounds, normalizes folds, computes detector signals from normalized metrics, then applies training-only SMOTE. | `Models/backd00r_ai/training/preprocess.py` |
| Expert training | Trains three 3-class Random Forest experts with target-smell sample weight `3.0` and other-smell weight `1.0`. | `Models/backd00r_ai/training/train_experts.py` |
| Gate training | Trains one 3-class `MoE_ContextualGate` over the same 29-feature vector. | `Models/backd00r_ai/training/train_gate.py` |
| Evaluation/export | Produces formal metrics, report CSV/JSON files, and four ONNX models. | `evaluate.py`, `export_onnx.py` |

### 1.2 Plugin Execution Pipeline

| Runtime layer | Responsibility |
|---|---|
| PSI/AST extraction | Extract class-level candidates enriched with method-level summary metrics. |
| JGit mining | Mine churn, volatility, bug-fix ratio, and author count. |
| Detector-inspired signals | Compute JDeodorant/iPlasma/Designite/JSpIRIT-inspired signal features after normalization. |
| ONNX inference | Run three experts and one gate locally. |
| MoE aggregation | Compute `P_final` and `max(P_final)` confidence. |
| IDE decision UI | Show dominant smell, confidence, explanations, and downstream priority/sequencing results. |

## 2. Input-Process-Output (IPO) Flow

### 2.1 Inputs

| Input | Description |
|---|---|
| `Models/Dataset/MLCQCodeSmellSamples.csv` | Native MLCQ labels and source metadata. |
| GitHub repositories | Exact Java snapshots and Git history referenced by MLCQ. |
| Java source files | Source used for raw class/method/access metric extraction. |
| Exported artifacts | Joblib/ONNX experts, gate, feature bounds, schemas, and reports. |

### 2.2 Process

```text
MLCQ row
  -> repository checkout
  -> raw Java/Git metrics
  -> project-grouped split
  -> train-only Min-Max normalization
  -> detector-inspired signals from normalized values
  -> training-only SMOTE
  -> Random Forest experts + MoE_ContextualGate
  -> weighted-sum aggregation
```

### 2.3 MoE Provider Routing and Inference

Each expert outputs:

```text
[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]
```

The gate outputs:

```text
[w_god, w_envy, w_long]
```

Aggregation:

```math
P_{final} =
normalize(
w_{god}E_{god}(x)
+ w_{envy}E_{envy}(x)
+ w_{long}E_{long}(x)
)
```

```text
dominant_smell = argmax(P_final)
confidence_score = max(P_final)
```

## 3. Unified Feature Vector

The canonical vector contains exactly 29 features.

| # | Feature | Category | Description |
|---:|---|---|---|
| 1 | `wmc` | Structural CK | Sum of method cyclomatic complexity. |
| 2 | `cbo` | Structural CK | Distinct project-local external classes referenced. |
| 3 | `lcom` | Structural CK | Lack of cohesion from shared field usage. |
| 4 | `rfc` | Structural CK | Own methods plus distinct project-local external method calls. |
| 5 | `dit` | Structural CK | Inheritance depth proxy. |
| 6 | `loc` | Structural | Class LOC excluding blank/comment lines. |
| 7 | `cc_max` | Method summary | Highest method cyclomatic complexity. |
| 8 | `nom` | Structural | Number of non-constructor methods. |
| 9 | `nof` | Structural | Number of fields. |
| 10 | `cida` | Access | Internal field accesses / internal field accesses plus internal accessor calls. |
| 11 | `coa` | Access | Project-local external member accesses / external plus local member accesses. |
| 12 | `method_loc_max` | Method summary | Maximum method LOC. |
| 13 | `method_loc_avg` | Method summary | Average method LOC. |
| 14 | `method_param_max` | Method summary | Maximum method parameter count. |
| 15 | `foreign_method_calls` | Access | Calls to methods owned by other project classes. |
| 16 | `foreign_field_accesses` | Access | Accesses to fields owned by other project classes. |
| 17 | `local_field_accesses` | Access | Accesses to fields owned by the current class. |
| 18 | `accessed_foreign_class_count` | Access | Distinct project-local foreign classes accessed. |
| 19 | `envy_method_ratio` | Access | Methods where foreign accesses exceed local accesses divided by `nom`. |
| 20 | `jdeodorant_signal` | Detector-inspired | Max JDeodorant-inspired smell score. |
| 21 | `iplasma_signal` | Detector-inspired | Max iPlasma-inspired smell score. |
| 22 | `designite_signal` | Detector-inspired | Max Designite-inspired smell score. |
| 23 | `jspirit_signal` | Detector-inspired | Max JSpIRIT-inspired smell score. |
| 24 | `code_churn` | Evolutionary | Changed lines from Git history. |
| 25 | `volatility` | Evolutionary | Change frequency/dispersion. |
| 26 | `bug_fix_ratio` | Evolutionary | Bug-fix commits divided by total commits. |
| 27 | `author_count` | Evolutionary | Distinct Git authors. |
| 28 | `hotspot_proxy` | Evolutionary composite | `0.5 * normalized_code_churn + 0.5 * bug_fix_ratio`. |
| 29 | `evolution_intensity` | Evolutionary composite | `0.6 * normalized_code_churn + 0.4 * normalized_volatility`. |

Schema source of truth:

```text
Models/backd00r_ai/configs/feature_schema.py
```

## 4. Hardware and Software Specifications

### 4.1 Development Environment

| Component | Requirement / Tool |
|---|---|
| Python | Python 3.14 in the current workspace environment |
| Git CLI | Git for Windows or equivalent |
| Dataset | `Models/Dataset/MLCQCodeSmellSamples.csv` |
| Core packages | pandas, scikit-learn, imbalanced-learn, joblib, tqdm |
| Export/evaluation | skl2onnx, onnx, onnxruntime, pytest |

### 4.2 Deployment Environment

| Component | Requirement / Tool |
|---|---|
| IntelliJ IDEA | IntelliJ Platform SDK plugin host |
| Java project support | Maven/Gradle and Java PSI |
| Git history | JGit |
| Inference | ONNX Runtime Java |
| Artifacts | four ONNX models plus `feature_bounds.json`, feature/label schemas, and metadata |

## 5. Data Sources and Preprocessing

### 5.1 Primary Dataset

MLCQ provides source metadata and labels, not model-ready metrics. The Python
pipeline reconstructs missing metrics from source snapshots and Git history.

### 5.2 Label Normalization and Data Reduction

| MLCQ smell | BACKD00R label | Status |
|---|---|---|
| `blob` | `GOD_CLASS` | Included |
| `feature envy` | `FEATURE_ENVY` | Included |
| `long method` | `LONG_METHOD` | Included |
| `data class` | Unsupported | Logged and excluded |

`binary_target = 0` when severity is `none`; otherwise `binary_target = 1`.

### 5.3 Repository and Raw Feature Reconstruction

`extract_training_features.py` writes:

```text
Models/artifacts/training_features.csv
Models/artifacts/training_features.csv.legacy23.backup
Models/artifacts/extracted_raw_metrics_cache.csv
Models/artifacts/extraction_errors.jsonl
```

The cache stores raw class metrics, method/access metrics, Git metrics, and
label mapping keyed by project id, repository id, commit hash, file path,
class/method identity, source hash, feature schema version, and extractor
version.

If an existing `training_features.csv` uses the legacy 23-feature header, the
extractor enters upgrade mode instead of halting. It backs up the legacy file,
reuses cached repositories from `Models/artifacts/repo_cache`, maps safe legacy
structural/history fields, force-reextracts corrected `cida`, `coa`, method
parameter/access metrics, rewrites `training_features.csv` with the canonical
29-feature header, and then resumes normal extraction for rows that were not
upgraded.

Performance safeguards:

- Pending rows are grouped by `repository` and `commit_hash`.
- `git checkout` runs once per repo+commit batch.
- Java file parsing is cached per batch and shared by all rows that reference the same file.
- Git history mining is performed once per target file in the checked-out batch.
- File/class extraction inside a checked-out commit uses a bounded worker pool with an effective maximum of 8 workers.
- Multiple checkouts in the same repository working tree are not run concurrently.

### 5.4 Min-Max Normalization

Implemented in `Models/backd00r_ai/training/preprocess.py`.

```math
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
```

Implementation behavior:

- Split by project before fitting bounds.
- Fit `x_min` and `x_max` on the training fold only.
- Apply the same bounds to train/validation/test.
- Clip normalized values to `[0, 1]`.
- Export `Models/artifacts/models/feature_bounds.json`.

### 5.5 SMOTE Balancing Engine

Implemented in `Models/backd00r_ai/training/preprocess.py`.

SMOTE is applied only to the normalized training fold. Each supported smell
class is oversampled up to the negative/no-smell majority count; validation and
test folds are never oversampled.

### 5.6 Training and Evaluation Outputs

Training consumes:

```text
Models/artifacts/train_balanced.csv
Models/artifacts/val_normalized.csv
```

Model outputs:

```text
Models/artifacts/models/god_class_expert.joblib
Models/artifacts/models/feature_envy_expert.joblib
Models/artifacts/models/long_method_expert.joblib
Models/artifacts/models/moe_contextual_gate.joblib
Models/artifacts/models/feature_bounds.json
Models/artifacts/models/feature_importance.csv
```

Evaluation/report outputs:

```text
Models/artifacts/evaluation_report.json
Models/artifacts/test_predictions.csv
Models/artifacts/classification_report.json
Models/artifacts/confusion_matrix.csv
Models/artifacts/precision_recall_f1.csv
Models/artifacts/gate_weights_summary.csv
```

ONNX outputs:

```text
Models/artifacts/onnx/god_class_expert.onnx
Models/artifacts/onnx/feature_envy_expert.onnx
Models/artifacts/onnx/long_method_expert.onnx
Models/artifacts/onnx/moe_contextual_gate.onnx
Models/artifacts/onnx/feature_schema.json
Models/artifacts/onnx/label_schema.json
Models/artifacts/onnx/model_card.json
```

Each ONNX model uses input node `float_input`, input shape `[1, 29]`, output
node `backd00r_probabilities_ordered`, and output order
`[GOD_CLASS, FEATURE_ENVY, LONG_METHOD]`.

## 6. Current Alignment Notes

The Python backend implements the feature/model side of Chapter 3. Refactoring
impact simulation, WMCA final priority scoring, dependency graph construction,
and DCAS sequencing remain plugin/runtime decision-layer concerns documented in
`algorithms.md`.
