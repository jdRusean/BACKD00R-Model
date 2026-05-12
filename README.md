# BACKD00R Python Backend AI Pipeline

This package implements the thesis-scoped Python backend AI model pipeline for
BACKD00R:

- MLCQ CSV parsing and label normalization.
- Java static metric reconstruction.
- CIDA and COA reconstruction from source access patterns.
- Git-history feature mining up to the target commit.
- Detector-philosophy signals inspired by JDeodorant, JSpIRIT, iPlasma, and DesigniteJava.
- Smell-specialist calibrated classifiers.
- `MoE_ContextualGate`, a calibrated Random Forest probability gate.
- Probability-aware final smell aggregation.
- ONNX export metadata and optional sklearn-to-ONNX model export.

The canonical deployment feature vector is defined in
`backd00r_ai/configs/feature_schema.py` and contains exactly 23 features.

## How to Run the Pipeline

Run all commands from the workspace root:

```powershell
cd "C:\Users\Owner\Documents\Sean\School\THESIS\System\SYSTEM MODEL"
```

The workspace root `pyproject.toml` adds `Models` to the pytest import path.
The executable scripts also add the `Models` folder to `sys.path` automatically
when you run them directly with VS Code's **Run Python File** button.

### Step 1. Create and Activate a Python Environment

Method A: VS Code Quick Setup

1. Open VS Code in this folder:

   ```text
   C:\Users\Owner\Documents\Sean\School\THESIS\System\SYSTEM MODEL
   ```

2. Open the VS Code terminal with `Terminal > New Terminal`.
3. Run:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r Models\requirements.txt
   ```

4. Select the interpreter at `.venv\Scripts\python.exe` in VS Code.

Method B: Terminal (Hybrid fallback)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r Models\requirements.txt
```

Expected output/artifacts:

- A new `.venv\` folder in the workspace root.
- Console output from `pip` showing installed packages such as `pandas`, `scikit-learn`, `pytest`, `skl2onnx`, and `onnxruntime`.

### Step 2. Verify the Codebase

Method A: VS Code Quick Run

- Open the VS Code Testing panel.
- Let VS Code discover the tests in `Models\tests`.
- Click run for all tests.

Method B: Terminal (Hybrid fallback)

```powershell
python -m pytest
```

Expected output/artifacts:

- Console output showing the tests in `Models\tests\` passing.
- No model files are created in this step.

### Step 3. Build the Supported MLCQ Sample Index

Method A: VS Code Quick Run

- Open `Models\backd00r_ai\dataset\dataset_builder.py`.
- Click **Run Python File**.
- The script uses these default paths automatically:
  - input: `Models\Dataset\MLCQCodeSmellSamples.csv`
  - output: `Models\artifacts\supported_samples.csv`
  - report: `Models\artifacts\data_quality.jsonl`

Method B: Terminal (Hybrid fallback)

```powershell
python -m backd00r_ai.dataset.dataset_builder --csv Models/Dataset/MLCQCodeSmellSamples.csv --out Models/artifacts/supported_samples.csv --report Models/artifacts/data_quality.jsonl
```

Expected output/artifacts:

- `Models\artifacts\supported_samples.csv`
  - Contains only v1-supported labels: `GOD_CLASS`, `FEATURE_ENVY`, and `LONG_METHOD`.
  - Maps MLCQ `blob` rows to `GOD_CLASS`.
  - Includes `binary_target`, where `severity != none` becomes `1`.
- `Models\artifacts\data_quality.jsonl`
  - Logs unsupported rows such as `data class`.
  - Logs duplicate rows when detected.

### Step 4. Fill the 23-Feature Training Table

This is the mass repository downloader and feature extractor. It reads
`Models\artifacts\supported_samples.csv`, clones/caches each GitHub repository,
checks out the exact sample commit, reconstructs all static/history/expert
features, and appends rows to `Models\artifacts\training_features.csv`.

The output CSV contains:

- `label`
- `binary_target`
- all 23 canonical feature columns from `backd00r_ai\configs\feature_schema.py`

Required columns:

```text
label, binary_target,
wmc, cbo, lcom, rfc, dit, loc, cc_max, nom, nof,
cida, coa,
jdeodorant_top, iplasma_top, designite_top, jspirit_top,
code_churn, volatility, bug_fix_ratio, author_count,
expert_score, hotspot_score, cm_score, complexity_level
```

File to create:

```text
Models\artifacts\training_features.csv
```

Method A: VS Code Quick Run

- Open `Models\backd00r_ai\dataset\extract_training_features.py`.
- Click **Run Python File**.
- The script uses these default paths automatically:
  - input: `Models\artifacts\supported_samples.csv`
  - output: `Models\artifacts\training_features.csv`
  - error log: `Models\artifacts\extraction_errors.jsonl`
  - repository cache: `Models\artifacts\repo_cache`
- It appends one completed training row at a time, so progress is saved even if you stop the run.
- It skips rows that already exist in `training_features.csv` when rerun.

Method B: Terminal (Hybrid fallback)

```powershell
python -m backd00r_ai.dataset.extract_training_features --supported Models/artifacts/supported_samples.csv --out Models/artifacts/training_features.csv --errors Models/artifacts/extraction_errors.jsonl --repo-cache Models/artifacts/repo_cache
```

For a small smoke run before downloading the full dataset:

```powershell
python -m backd00r_ai.dataset.extract_training_features --limit 5
```

Expected output/artifacts:

- `Models\artifacts\training_features.csv`
  - Fully populated rows containing metadata plus all 23 feature columns.
- `Models\artifacts\repo_cache\`
  - Cached cloned repositories. Existing repo folders are reused.
- `Models\artifacts\extraction_errors.jsonl`
  - Logs dead repositories, missing commits, missing files, parse/extraction failures, and other per-row errors.

The batch job continues after failures. A single dead repository, missing commit, or parse problem will not stop the whole extraction run.

### Step 5. Preprocess Training Features

Method A: VS Code Quick Run

- Open `Models\backd00r_ai\training\preprocess.py`.
- Confirm `Models\artifacts\training_features.csv` exists and contains actual rows.
- Click **Run Python File**.
- The script uses these default paths automatically:
  - input: `Models\artifacts\training_features.csv`
  - output directory: `Models\artifacts`
  - feature bounds: `Models\artifacts\models\feature_bounds.json`
  - log: `Models\artifacts\preprocess_log.jsonl`

Method B: Terminal (Hybrid fallback)

```powershell
python -m backd00r_ai.training.preprocess --input Models/artifacts/training_features.csv --out-dir Models/artifacts --bounds Models/artifacts/models/feature_bounds.json --log Models/artifacts/preprocess_log.jsonl
```

Expected output/artifacts:

- `Models\artifacts\train_balanced.csv`
- `Models\artifacts\val_normalized.csv`
- `Models\artifacts\test_normalized.csv`
- `Models\artifacts\models\feature_bounds.json`
- `Models\artifacts\preprocess_log.jsonl`

### Step 6. Train the Three Smell-Specialist Experts

Method A: VS Code Quick Run

- Open `Models\backd00r_ai\training\train_experts.py`.
- Confirm `Models\artifacts\train_balanced.csv` exists and contains actual rows.
- Click **Run Python File**.
- The script uses these default paths automatically:
  - input: `artifacts\train_balanced.csv` when a workspace-level `artifacts` folder exists; otherwise `Models\artifacts\train_balanced.csv`
  - validation fold: `artifacts\val_normalized.csv` when a workspace-level `artifacts` folder exists; otherwise `Models\artifacts\val_normalized.csv`
  - output directory: `artifacts\models` when a workspace-level `artifacts` folder exists; otherwise `Models\artifacts\models`

Method B: Terminal (Hybrid fallback)

```powershell
python -m backd00r_ai.training.train_experts --input Models/artifacts/train_balanced.csv --val Models/artifacts/val_normalized.csv --out-dir Models/artifacts/models
```

Expected output/artifacts:

- `Models\artifacts\models\god_class_expert.joblib`
- `Models\artifacts\models\feature_envy_expert.joblib`
- `Models\artifacts\models\long_method_expert.joblib`
- `Models\artifacts\models\experts_manifest.json`
- Console validation summary:
  - training positive/negative support for each one-vs-rest expert
  - validation positive support for each smell label
  - per-label threshold@0.50 Precision/Recall/F1 for `GOD_CLASS`, `FEATURE_ENVY`, and `LONG_METHOD`
  - per-label ranked@positives Precision/Recall/F1 for the same labels, useful when calibrated probabilities are below `0.50`

These are calibrated binary smell-specialist models.

### Step 7. Train the `MoE_ContextualGate`

Method A: VS Code Quick Run

- Open `Models\backd00r_ai\training\train_gate.py`.
- Confirm Step 6 produced the three expert `.joblib` files.
- Click **Run Python File**.
- The script uses these default paths automatically:
  - input: `artifacts\train_balanced.csv` when a workspace-level `artifacts` folder exists; otherwise `Models\artifacts\train_balanced.csv`
  - validation fold: `artifacts\val_normalized.csv` when a workspace-level `artifacts` folder exists; otherwise `Models\artifacts\val_normalized.csv`
  - experts directory: `artifacts\models` when a workspace-level `artifacts` folder exists; otherwise `Models\artifacts\models`
  - output: `artifacts\models\moe_contextual_gate.joblib` when a workspace-level `artifacts` folder exists; otherwise `Models\artifacts\models\moe_contextual_gate.joblib`

Method B: Terminal (Hybrid fallback)

```powershell
python -m backd00r_ai.training.train_gate --input Models/artifacts/train_balanced.csv --val Models/artifacts/val_normalized.csv --experts-dir Models/artifacts/models --out Models/artifacts/models/moe_contextual_gate.joblib
```

Expected output/artifacts:

- `Models\artifacts\models\moe_contextual_gate.joblib`
- The script internally verifies that these gate probability distributions normalize to `1.0`:
  - `smell_confidence_probability`
  - `expert_reliability_probability`
  - `contextual_routing_probability`
- Console validation summary:
  - macro Precision/Recall/F1 for `MoE_ContextualGate` on `val_normalized.csv`

### Step 8. Run Inference on 23-Feature Vectors

Create an input JSON file containing one or more 23-number vectors.

Example file:

```text
Models\artifacts\vectors.json
```

Example contents:

```json
[
  [
    30, 12, 0.75, 40, 1,
    300, 14, 20, 8,
    0.6, 10,
    0.7, 0.8, 0.5, 0.6,
    120, 0.4, 0.2, 3,
    0.65, 0.45, 0.55, 0.7
  ]
]
```

Method A: VS Code Quick Run

- Open `Models\backd00r_ai\inference\predict_project.py`.
- Confirm Step 6 and Step 7 produced all required `.joblib` files.
- Click **Run Python File**.
- If `Models\artifacts\vectors.json` does not exist, the script creates a sample vector file and then uses it.
- The script uses these default paths automatically:
  - artifacts directory: `Models\artifacts\models`
  - vector input: `Models\artifacts\vectors.json`
  - output: `Models\artifacts\predictions.json`

Method B: Terminal (Hybrid fallback)

```powershell
python -m backd00r_ai.inference.predict_project --artifacts-dir Models/artifacts/models --vectors-json Models/artifacts/vectors.json --out Models/artifacts/predictions.json
```

Expected output/artifacts:

- `Models\artifacts\predictions.json`
- Each prediction contains:
  - `predictions`: normalized final smell probability distribution.
  - `confidence_score`: highest final smell probability.
  - `dominant_smell`: predicted smell label.
  - `gate`: the three interpretable gate probability distributions.

### Step 9. Evaluate Predictions

This step requires a prediction CSV, not the JSON output from Step 8. The CSV must contain:

```text
label,predicted_label,final_confidence
```

Optional column:

```text
binary_target
```

Expected input file:

```text
Models\artifacts\prediction_results.csv
```

Method A: VS Code Quick Run

- Open `Models\backd00r_ai\training\evaluate.py`.
- Click **Run Python File**.
- If `Models\artifacts\prediction_results.csv` does not exist yet, the script creates a CSV template with the required header and stops safely.
- Fill `prediction_results.csv`, then click **Run Python File** again.
- The script uses these default paths automatically:
  - input: `Models\artifacts\prediction_results.csv`
  - output: `Models\artifacts\evaluation_report.json`

Method B: Terminal (Hybrid fallback)

```powershell
python -m backd00r_ai.training.evaluate --predictions Models/artifacts/prediction_results.csv --out Models/artifacts/evaluation_report.json
```

Expected output/artifacts:

- `Models\artifacts\evaluation_report.json`
- Contains:
  - classification report
  - confusion matrix
  - Brier score when `binary_target` is present

### Step 10. Export Models and Metadata to ONNX

Method A: VS Code Quick Run

- Open `Models\backd00r_ai\training\export_onnx.py`.
- Confirm Step 6 and Step 7 produced the `.joblib` model artifacts.
- Click **Run Python File**.
- The script uses these default paths automatically:
  - artifacts directory: `Models\artifacts\models`
  - output directory: `Models\artifacts\onnx`

Method B: Terminal (Hybrid fallback)

```powershell
python -m backd00r_ai.training.export_onnx --artifacts-dir Models/artifacts/models --out-dir Models/artifacts/onnx
```

Expected output/artifacts:

- `Models\artifacts\onnx\feature_schema.json`
- `Models\artifacts\onnx\label_schema.json`
- `Models\artifacts\onnx\model_card.json`
- `Models\artifacts\onnx\god_class_expert.onnx`
- `Models\artifacts\onnx\feature_envy_expert.onnx`
- `Models\artifacts\onnx\long_method_expert.onnx`
- `Models\artifacts\onnx\moe_contextual_gate_smell_confidence.onnx`
- `Models\artifacts\onnx\moe_contextual_gate_expert_reliability.onnx`
- `Models\artifacts\onnx\moe_contextual_gate_contextual_routing.onnx`

The gate is exported as three ONNX heads because `MoE_ContextualGate` is a custom Python wrapper around three calibrated Random Forest probability heads.

## Strict Execution Order

Use this order for a full local run:

1. Step 1: create environment and install dependencies.
2. Step 2: run tests.
3. Step 3: build `supported_samples.csv`.
4. Step 4: fill `training_features.csv` with `extract_training_features.py`.
5. Step 5: preprocess into `train_balanced.csv`, `val_normalized.csv`, and `test_normalized.csv`.
6. Step 6: train smell experts.
7. Step 7: train `MoE_ContextualGate`.
8. Step 8: run inference.
9. Step 9: evaluate predictions.
10. Step 10: export ONNX artifacts.

## Common Failure Points

- `No module named pytest`: dependencies are not installed; rerun Step 1.
- `Training CSV is missing feature columns`: `training_features.csv` does not contain all 23 canonical features.
- `train_balanced.csv` is missing: run Step 5 preprocessing before Step 6 training.
- `val_normalized.csv` is missing: run Step 5 preprocessing before training, or pass a different `--val` file.
- `Prediction requires joblib`: dependencies are not installed; rerun Step 1.
- `MoE_ContextualGate has not been fitted`: run Step 7 before Step 8.
- Missing ONNX files: confirm Step 6 and Step 7 produced `.joblib` artifacts before Step 10.
- VS Code says `ModuleNotFoundError: backd00r_ai`: make sure you are running one of the updated executable files listed above. Each one now bootstraps `Models` into `sys.path`.

Training requires a feature table that already includes the 23 reconstructed
feature columns plus `label` and `binary_target`.
