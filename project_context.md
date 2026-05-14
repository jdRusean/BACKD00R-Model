# BACKD00R Project Context

## Project Summary

BACKD00R is a thesis-scoped code-smell detection and prioritization system for
Java projects. The Python backend builds the Random Forest-based
MoE-inspired model layer from MLCQ, reconstructs missing source/history
metrics, and exports artifacts for local IDE plugin inference.

Use this wording:

> BACKD00R uses a 29-feature unified vector composed of class-level structural metrics, method-level summary metrics, detector-inspired smell signals, and evolutionary metrics.

> Random Forest-based MoE-inspired expert ensemble with contextual routing.

> Detector-inspired signals approximate smell evidence associated with JDeodorant, iPlasma, Designite, and JSpIRIT.

Do not describe the system as executing exact external detector tools or as a
classical neural Mixture-of-Experts architecture.

## Scope

| MLCQ label | BACKD00R label | Status |
|---|---|---|
| `blob` | `GOD_CLASS` | Supported |
| `feature envy` | `FEATURE_ENVY` | Supported |
| `long method` | `LONG_METHOD` | Supported |
| `data class` | Unsupported | Logged and excluded |

Primary dataset:

```text
Models/Dataset/MLCQCodeSmellSamples.csv
```

Main Python package:

```text
Models/backd00r_ai
```

## Pipeline Summary

```text
MLCQ CSV
  -> supported sample filtering
  -> repository clone/cache
  -> exact commit checkout
  -> repo+commit batched raw Java/Git metric extraction
  -> raw metrics cache
  -> project-grouped train/validation/test split
  -> train-fold Min-Max normalization
  -> detector-inspired signals from normalized metrics
  -> training-fold-only SMOTE
  -> three weighted Random Forest experts
  -> one MoE_ContextualGate
  -> weighted-sum aggregation
  -> ONNX export
```

## Unified 29-Feature Contract

| # | Feature | Category |
|---:|---|---|
| 1 | `wmc` | Structural |
| 2 | `cbo` | Structural |
| 3 | `lcom` | Structural |
| 4 | `rfc` | Structural |
| 5 | `dit` | Structural |
| 6 | `loc` | Structural |
| 7 | `cc_max` | Method summary |
| 8 | `nom` | Structural |
| 9 | `nof` | Structural |
| 10 | `cida` | Access |
| 11 | `coa` | Access |
| 12 | `method_loc_max` | Method summary |
| 13 | `method_loc_avg` | Method summary |
| 14 | `method_param_max` | Method summary |
| 15 | `foreign_method_calls` | Access |
| 16 | `foreign_field_accesses` | Access |
| 17 | `local_field_accesses` | Access |
| 18 | `accessed_foreign_class_count` | Access |
| 19 | `envy_method_ratio` | Access |
| 20 | `jdeodorant_signal` | Detector-inspired |
| 21 | `iplasma_signal` | Detector-inspired |
| 22 | `designite_signal` | Detector-inspired |
| 23 | `jspirit_signal` | Detector-inspired |
| 24 | `code_churn` | Evolutionary |
| 25 | `volatility` | Evolutionary |
| 26 | `bug_fix_ratio` | Evolutionary |
| 27 | `author_count` | Evolutionary |
| 28 | `hotspot_proxy` | Evolutionary composite |
| 29 | `evolution_intensity` | Evolutionary composite |

Schema source of truth:

```text
Models/backd00r_ai/configs/feature_schema.py
```

## Model Architecture

| Model | Input | Output |
|---|---|---|
| `god_class_expert` | 29-feature vector | `[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]` |
| `feature_envy_expert` | 29-feature vector | `[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]` |
| `long_method_expert` | 29-feature vector | `[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]` |
| `moe_contextual_gate` | 29-feature vector | `[w_god, w_envy, w_long]` |

Expert specialization weights:

```text
target smell rows = 3.0
other smell rows  = 1.0
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

## Key Artifacts

| Artifact | Purpose |
|---|---|
| `Models/artifacts/supported_samples.csv` | Supported MLCQ rows after label normalization |
| `Models/artifacts/training_features.csv` | Raw 29-feature training table |
| `Models/artifacts/training_features.csv.legacy23.backup` | Automatic backup when a legacy 23-feature table is upgraded |
| `Models/artifacts/extracted_raw_metrics_cache.csv` | Raw metric cache keyed by project, repository, commit, source, schema, and extractor identity |
| `Models/artifacts/train_balanced.csv` | Normalized and SMOTE-balanced training fold |
| `Models/artifacts/val_normalized.csv` | Normalized validation fold |
| `Models/artifacts/test_normalized.csv` | Normalized test fold |
| `Models/artifacts/models/feature_bounds.json` | Min-Max bounds for inference |
| `Models/artifacts/models/*_expert.joblib` | Python expert artifacts |
| `Models/artifacts/models/moe_contextual_gate.joblib` | Python gate artifact |
| `Models/artifacts/onnx/*.onnx` | Java/plugin inference artifacts |

## Non-Negotiable Boundaries

- Do not change the 29-feature order without updating Python, ONNX metadata, and plugin-side inference together.
- Do not parallelize multiple checkouts in the same repository working tree.
- Do not fit Min-Max bounds on validation, test, or runtime data.
- Do not apply SMOTE outside the training fold.
- Do not run external detector tools as part of this Python pipeline.
- Do not add neural MoE components.
- Do not add automated refactoring execution to the Python model pipeline.
