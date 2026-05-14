# BACKD00R Model Training Spec

## Goal

Train a Random Forest-based MoE-inspired model for three smell labels:

```text
GOD_CLASS
FEATURE_ENVY
LONG_METHOD
```

BACKD00R uses **class-level candidates enriched with method-level summary metrics**.

## Final 29-Feature Order

```text
wmc,
cbo,
lcom,
rfc,
dit,
loc,
cc_max,
nom,
nof,
cida,
coa,
method_loc_max,
method_loc_avg,
method_param_max,
foreign_method_calls,
foreign_field_accesses,
local_field_accesses,
accessed_foreign_class_count,
envy_method_ratio,
jdeodorant_signal,
iplasma_signal,
designite_signal,
jspirit_signal,
code_churn,
volatility,
bug_fix_ratio,
author_count,
hotspot_proxy,
evolution_intensity
```

## Extraction Rules

Extract raw metrics from Java source using AST parsing.

Important rule:

```text
For foreign access metrics, count only accesses to project-local classes.
Exclude JDK and external library types.
```

Exclude:

```text
java.*
javax.*
jakarta.*
kotlin.*
org.jetbrains.*
com.intellij.*
third-party dependency packages
```

Best rule:

```text
foreign access = access to another class inside the analyzed project source set
```

## Feature Definitions

Class-level metrics:

```text
wmc                         sum of method cyclomatic complexity
cbo                         distinct project-local external classes referenced
lcom                        lack of cohesion from shared field usage
rfc                         own methods + distinct project-local external methods called
dit                         inheritance depth
loc                         class LOC excluding blank/comment lines
cc_max                      highest cyclomatic complexity among methods
nom                         number of non-constructor methods
nof                         number of fields
```

Access/method metrics:

```text
cida                        direct internal field accesses / max(1, direct internal field accesses + internal accessor calls)
coa                         project-local external member accesses / max(1, project-local external member accesses + local member accesses)
method_loc_max              maximum method LOC
method_loc_avg              average method LOC
method_param_max            maximum method parameter count
foreign_method_calls        calls to methods owned by other project classes
foreign_field_accesses      accesses to fields owned by other project classes
local_field_accesses        accesses to fields owned by the current class
accessed_foreign_class_count distinct project-local foreign classes accessed
envy_method_ratio           methods where project-local foreign accesses > local accesses / max(1, nom)
```

Evolutionary metrics:

```text
code_churn                  changed lines from Git history
volatility                  change frequency/dispersion
bug_fix_ratio               bug-fix commits / max(1, total commits)
author_count                distinct Git authors
hotspot_proxy               0.5 * normalized_code_churn + 0.5 * bug_fix_ratio
evolution_intensity         0.6 * normalized_code_churn + 0.4 * normalized_volatility
```

## Tool-Inspired Signals

Do not run external tools. Compute detector-inspired heuristic signals.

Each signal is numeric:

```text
0.0 to 1.0
```

Each detector computes internal smell scores:

```text
god_class_score
feature_envy_score
long_method_score
```

Then outputs one feature:

```text
detector_signal = max(god_class_score, feature_envy_score, long_method_score)
```

Final detector features:

```text
jdeodorant_signal
iplasma_signal
designite_signal
jspirit_signal
```

## Normalization

Split before fitting normalization.

```text
1. Split by project.
2. Fit Min-Max bounds on training set only.
3. Normalize train/validation/test using training bounds.
4. Compute detector-inspired signals using normalized metric values.
5. Apply SMOTE only to training set.
6. Save feature_bounds.json.
```

Safe helpers:

```text
safe_div(a, b) = a / max(1, b)
clamp(x) = min(1, max(0, x))
```

## Base Smell Scores

God Class:

```text
god_class_base =
clamp(
  0.25 * wmc_n +
  0.20 * cbo_n +
  0.20 * lcom_n +
  0.15 * loc_n +
  0.10 * nom_n +
  0.10 * nof_n
)
```

Long Method:

```text
long_method_base =
clamp(
  0.40 * method_loc_max_n +
  0.30 * cc_max_n +
  0.20 * method_param_max_n +
  0.10 * method_loc_avg_n
)
```

Feature Envy:

```text
foreign_access_total =
foreign_method_calls + foreign_field_accesses

local_access_total =
local_field_accesses

foreign_access_ratio =
foreign_access_total / max(1, foreign_access_total + local_access_total)

feature_envy_base =
clamp(
  0.35 * foreign_access_ratio +
  0.20 * accessed_foreign_class_count_n +
  0.20 * envy_method_ratio +
  0.15 * coa +
  0.10 * cbo_n
)
```

## Detector Formulas

JDeodorant-inspired:

```text
jdeodorant_signal =
max(
  clamp(0.45 * god_class_base + 0.30 * lcom_n + 0.25 * wmc_n),
  clamp(0.55 * feature_envy_base + 0.25 * foreign_access_ratio + 0.20 * envy_method_ratio),
  clamp(0.55 * long_method_base + 0.25 * method_loc_max_n + 0.20 * cc_max_n)
)
```

iPlasma-inspired:

```text
iplasma_signal =
max(
  clamp(0.35 * god_class_base + 0.25 * wmc_n + 0.20 * cbo_n + 0.20 * loc_n),
  clamp(0.45 * feature_envy_base + 0.25 * accessed_foreign_class_count_n + 0.20 * foreign_access_ratio + 0.10 * coa),
  clamp(0.45 * long_method_base + 0.30 * cc_max_n + 0.25 * method_loc_max_n)
)
```

Designite-inspired:

```text
designite_signal =
max(
  clamp(0.35 * loc_n + 0.25 * wmc_n + 0.20 * cbo_n + 0.20 * lcom_n),
  clamp(0.50 * feature_envy_base + 0.25 * foreign_method_calls_n + 0.15 * foreign_field_accesses_n + 0.10 * coa),
  clamp(0.40 * method_loc_max_n + 0.30 * cc_max_n + 0.20 * method_param_max_n + 0.10 * method_loc_avg_n)
)
```

JSpIRIT-inspired:

```text
jspirit_signal =
max(
  clamp(0.30 * god_class_base + 0.25 * rfc_n + 0.20 * cbo_n + 0.15 * wmc_n + 0.10 * lcom_n),
  clamp(0.40 * feature_envy_base + 0.25 * foreign_access_ratio + 0.20 * accessed_foreign_class_count_n + 0.15 * envy_method_ratio),
  clamp(0.40 * long_method_base + 0.25 * cc_max_n + 0.20 * method_loc_max_n + 0.15 * method_param_max_n)
)
```

## MLCQ Dataset Pipeline

```text
1. Load MLCQ labels.
2. Load corresponding Java source projects/files.
3. Parse Java source with AST parser.
4. Extract raw class-level, method-level, access, and Git metrics.
5. Map labels to GOD_CLASS, FEATURE_ENVY, LONG_METHOD.
6. Split by project.
7. Fit normalization on training split.
8. Normalize metrics.
9. Compute detector-inspired signals.
10. Build final 29-feature rows.
11. Apply SMOTE to training split only.
12. Train models.
```

For method-level labels:

```text
Map the method label to its containing class.
Represent method-level evidence through method/access summary metrics.
```

## Extraction Cache

Cache raw metrics only.

Do not cache normalized values, detector signals, or final feature rows unless the split and bounds are also part of the cache key.

Cache file:

```text
extracted_raw_metrics_cache.csv
```

Cache key:

```text
project_id,
file_path,
class_name,
method_signature,
source_hash,
feature_schema_version,
extractor_version
```

Cache stores:

```text
raw class metrics
raw method/access metrics
raw Git metrics
label mapping
```

Reuse cache only when:

```text
source_hash unchanged
feature_schema_version unchanged
extractor_version unchanged
```

## Random Forest MoE

Train three specialized Random Forest experts:

```text
god_class_expert
feature_envy_expert
long_method_expert
```

Each expert receives the same 29-feature vector and outputs:

```text
[P(GOD_CLASS), P(FEATURE_ENVY), P(LONG_METHOD)]
```

Specialize experts using sample weights:

```text
god_class_expert:     GOD_CLASS rows weight 3.0, others 1.0
feature_envy_expert:  FEATURE_ENVY rows weight 3.0, others 1.0
long_method_expert:   LONG_METHOD rows weight 3.0, others 1.0
```

Train contextual gate:

```text
moe_contextual_gate
```

Gate input:

```text
same 29-feature vector
```

Gate output:

```text
[w_god, w_envy, w_long]
```

Final aggregation:

```text
P_final =
normalize(
  w_god  * expert_god(x)
+ w_envy * expert_envy(x)
+ w_long * expert_long(x)
)
```

Dominant smell:

```text
argmax(P_final)
```

MoE confidence:

```text
max(P_final)
```

## Export Artifacts

```text
god_class_expert.onnx
feature_envy_expert.onnx
long_method_expert.onnx
moe_contextual_gate.onnx
feature_bounds.json
feature_schema.json
label_mapping.json
model_metadata.json
classification_report.json
confusion_matrix.csv
precision_recall_f1.csv
feature_importance.csv
gate_weights_summary.csv
```

## Runtime Requirement

The plugin must:

```text
1. Extract the same 29 features.
2. Use the same feature order.
3. Use exported feature_bounds.json.
4. Run the three ONNX experts.
5. Run the ONNX gate.
6. Aggregate probabilities.
7. Use argmax(P_final) as dominant smell.
8. Use max(P_final) as MoE confidence for WMCA.
```

## Thesis Wording

Use:

> BACKD00R uses a 29-feature unified vector composed of class-level structural metrics, method-level summary metrics, detector-inspired smell signals, and evolutionary metrics.

Use:

> Random Forest-based MoE-inspired expert ensemble with contextual routing.

Use:

> Detector-inspired signals approximate smell evidence associated with JDeodorant, iPlasma, Designite, and JSpIRIT.

Avoid:

> exact external tool execution

Avoid:

> classical neural Mixture-of-Experts architecture