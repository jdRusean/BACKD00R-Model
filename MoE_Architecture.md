# BACKD00R MoE Architecture Spec

## Unified Input Contract

BACKD00R uses a **29-feature unified vector** composed of class-level structural
metrics, method-level summary metrics, detector-inspired smell signals, and
evolutionary metrics.

All experts and the gate consume this same ordered vector from
`Models/backd00r_ai/configs/feature_schema.py`.

## Expert Models (3 total)

Each expert is a **3-class probabilistic Random Forest classifier**.

| Model | Input | Output |
|---|---|---|
| `god_class_expert` | 29-feature vector | `[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]` |
| `feature_envy_expert` | 29-feature vector | `[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]` |
| `long_method_expert` | 29-feature vector | `[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]` |

Each expert is trained on the same 3-class training fold and specialized with
per-row `sample_weight`:

```text
target smell rows: 3.0
other smell rows:  1.0
```

## Gate Model (1 total)

A single **3-class Random Forest classifier** produces contextual routing
weights.

| Model | Input | Output |
|---|---|---|
| `moe_contextual_gate` | 29-feature vector | `[w_god, w_envy, w_long]` |

The gate output is a normalized probability vector interpreted as expert
routing weights.

## Aggregation

```text
P_final = normalize( w_god  * expert_god(x)
                   + w_envy * expert_envy(x)
                   + w_long * expert_long(x) )
```

`P_final` is a 3-class distribution:

```text
[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]
```

```text
dominant_smell = argmax(P_final)
confidence = max(P_final)
confidence_margin = top - second
```

## Normalization and Detector Signals

Training preprocessing follows this order:

```text
1. Split by project.
2. Fit Min-Max bounds on the training split only.
3. Normalize train/validation/test using training bounds.
4. Compute detector-inspired signals using normalized metric values.
5. Apply SMOTE only to the training set.
6. Save feature_bounds.json.
```

Detector-inspired signals approximate smell evidence associated with
JDeodorant, iPlasma, Designite, and JSpIRIT. External detector tools are not
executed.

## ONNX Export

```text
god_class_expert.onnx          -> output: float[1][3]
feature_envy_expert.onnx       -> output: float[1][3]
long_method_expert.onnx        -> output: float[1][3]
moe_contextual_gate.onnx       -> output: float[1][3]
feature_bounds.json            -> min/max per raw feature for inference-time normalization
```

All models share:

```text
input node: float_input
input shape: [1, 29]
output node: backd00r_probabilities_ordered
output order: [GOD_CLASS, FEATURE_ENVY, LONG_METHOD]
```

Runtime must extract the same 29 features, use the same order, apply
`feature_bounds.json`, run the three ONNX experts and ONNX gate, and aggregate
probabilities with the weighted sum formula above.

Legacy 23-feature training CSVs are not valid model inputs. The Python
extractor can upgrade an existing legacy `training_features.csv` by backing it
up, reusing cached repositories, mapping safe old structural/history metrics,
force-reextracting corrected access and method-level metrics, and then resuming
normal repo+commit batched extraction.

## Training Notes

- Experts are trained with target-smell sample weight `3.0` and other-smell sample weight `1.0`.
- Gate is trained on the full training fold with multi-class labels.
- SMOTE is applied to the training fold only, after normalization.
- `feature_bounds.json` is fitted on the training fold only; the same bounds are used at plugin inference time.
