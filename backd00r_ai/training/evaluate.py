"""Evaluation helpers for BACKD00R model artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MODELS_ROOT = next(
    path for path in Path(__file__).resolve().parents if path.name == "Models"
)
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))


def evaluate_predictions(input_csv: Path, output_json: Path) -> None:
    if not input_csv.exists():
        write_prediction_results_template(input_csv)
        print(f"Prediction results CSV not found. Created template: {input_csv}")
        print("Fill this CSV with label,predicted_label,final_confidence rows, then run evaluate.py again.")
        return
    try:
        import pandas as pd
        from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix
    except ImportError as exc:
        raise RuntimeError("evaluate.py requires pandas and scikit-learn.") from exc

    frame = pd.read_csv(input_csv)
    required = {"label", "predicted_label", "final_confidence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")
    if frame.empty:
        print(f"Prediction results CSV has no rows yet: {input_csv}")
        print("Fill it with label,predicted_label,final_confidence rows, then run evaluate.py again.")
        return
    report = {
        "classification_report": classification_report(
            frame["label"], frame["predicted_label"], output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(frame["label"], frame["predicted_label"]).tolist(),
    }
    if "binary_target" in frame.columns:
        report["brier_score"] = brier_score_loss(
            frame["binary_target"].astype(int),
            frame["final_confidence"].astype(float),
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report: {output_json}")


def write_prediction_results_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "predicted_label", "final_confidence", "binary_target"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=MODELS_ROOT / "artifacts" / "prediction_results.csv",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=MODELS_ROOT / "artifacts" / "evaluation_report.json",
        type=Path,
    )
    args = parser.parse_args()
    evaluate_predictions(args.predictions, args.out)


if __name__ == "__main__":
    main()
