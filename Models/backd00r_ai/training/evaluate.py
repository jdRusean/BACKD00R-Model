"""Formal evaluation for BACKD00R MoE model artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

MODELS_ROOT = next(
    path for path in Path(__file__).resolve().parents if path.name == "Models"
)
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from backd00r_ai.configs.feature_schema import FEATURE_NAMES_29
from backd00r_ai.configs.label_schema import SUPPORTED_LABELS
from backd00r_ai.inference.predict_project import PredictionPipeline

DEFAULT_ARTIFACTS_DIR = (
    Path("artifacts") if (Path.cwd() / "artifacts").exists() else MODELS_ROOT / "artifacts"
)


def _require_dependencies():
    try:
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError("evaluate.py requires pandas, scikit-learn, and joblib.") from exc
    return {
        "pd": pd,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
    }


def _load_test_frame(test_csv: Path):
    deps = _require_dependencies()
    pd = deps["pd"]
    if not test_csv.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_csv}")
    frame = pd.read_csv(test_csv)
    missing = [name for name in FEATURE_NAMES_29 if name not in frame.columns]
    if missing:
        raise ValueError(f"{test_csv} is missing feature columns: {missing}")
    if "label" not in frame.columns:
        raise ValueError(f"{test_csv} must contain a 'label' column.")
    if frame.empty:
        raise ValueError(f"Test dataset has no rows: {test_csv}")
    frame["label"] = frame["label"].astype(str).str.strip().str.upper()
    return frame


def evaluate_moe_pipeline(
    test_csv: Path,
    artifacts_dir: Path,
    output_json: Path,
    predictions_csv: Path,
) -> dict[str, Any]:
    deps = _require_dependencies()
    frame = _load_test_frame(test_csv)
    vectors = frame.loc[:, FEATURE_NAMES_29].astype(float).values.tolist()
    pipeline = PredictionPipeline(artifacts_dir)
    outputs = pipeline.predict_vectors(vectors, already_normalized=True)

    y_true_all = frame["label"].astype(str).tolist()
    y_pred_all = [str(row["dominant_smell"]) for row in outputs]
    y_score_all = [
        [float(row["predictions"][label]) for label in SUPPORTED_LABELS]
        for row in outputs
    ]
    if "binary_target" in frame.columns:
        primary_indices = [
            idx for idx, value in enumerate(frame["binary_target"].fillna(0).astype(int))
            if value == 1
        ]
    else:
        primary_indices = list(range(len(frame)))
    if not primary_indices:
        raise ValueError("No positive smell-present rows found for primary classification evaluation.")

    y_true = [y_true_all[idx] for idx in primary_indices]
    y_pred = [y_pred_all[idx] for idx in primary_indices]
    y_score = [y_score_all[idx] for idx in primary_indices]

    primary_metrics = _classification_metrics(deps, y_true, y_pred, y_score)
    full_fold_metrics = _classification_metrics(deps, y_true_all, y_pred_all, y_score_all)
    ranking_metrics = _ranking_metrics(frame, outputs)

    report: dict[str, Any] = {
        "evaluation_dataset": str(test_csv),
        "artifacts_dir": str(artifacts_dir),
        "sample_count": len(frame),
        "primary_evaluation": "positive smell-present rows where binary_target == 1",
        "primary_sample_count": len(y_true),
        "full_fold_sample_count": len(y_true_all),
        "label_order": list(SUPPORTED_LABELS),
        "positive_smell_classification": primary_metrics,
        "full_fold_diagnostic": full_fold_metrics,
        "confidence_ranking": ranking_metrics,
    }

    _write_prediction_rows(predictions_csv, frame, outputs)
    _write_export_reports(output_json.parent, report, outputs)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report, output_json, predictions_csv)
    return report


def _write_export_reports(output_dir: Path, report: dict[str, Any], outputs: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = report["positive_smell_classification"]
    (output_dir / "classification_report.json").write_text(
        json.dumps(primary["classification_report"], indent=2),
        encoding="utf-8",
    )
    matrix = primary["confusion_matrix"]
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *matrix["labels"]])
        for label, row in zip(matrix["labels"], matrix["matrix"]):
            writer.writerow([label, *row])
    with (output_dir / "precision_recall_f1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        for label, values in primary["per_class"].items():
            writer.writerow({"label": label, **values})
    with (output_dir / "gate_weights_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["expert", "mean_weight"])
        writer.writeheader()
        if outputs:
            experts = outputs[0]["gate_weights"].keys()
            for expert in experts:
                writer.writerow(
                    {
                        "expert": expert,
                        "mean_weight": sum(float(row["gate_weights"][expert]) for row in outputs)
                        / len(outputs),
                    }
                )


def _classification_metrics(
    deps: dict[str, Any],
    y_true: list[str],
    y_pred: list[str],
    y_score: list[list[float]],
) -> dict[str, Any]:
    precision_recall_fscore_support = deps["precision_recall_fscore_support"]
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(SUPPORTED_LABELS),
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(SUPPORTED_LABELS),
        average="weighted",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(SUPPORTED_LABELS),
        average=None,
        zero_division=0,
    )
    return {
        "sample_count": len(y_true),
        "accuracy": float(deps["accuracy_score"](y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1_score": float(weighted_f1),
        "macro_roc_auc_ovr": _safe_macro_roc_auc(deps["roc_auc_score"], y_true, y_score),
        "macro_brier_score_ovr": _macro_brier_score_ovr(y_true, y_score),
        "multiclass_brier_score": _multiclass_brier_score(y_true, y_score),
        "per_class": {
            label: {
                "precision": float(per_precision[idx]),
                "recall": float(per_recall[idx]),
                "f1_score": float(per_f1[idx]),
                "support": int(per_support[idx]),
            }
            for idx, label in enumerate(SUPPORTED_LABELS)
        },
        "confusion_matrix": {
            "labels": list(SUPPORTED_LABELS),
            "matrix": deps["confusion_matrix"](
                y_true,
                y_pred,
                labels=list(SUPPORTED_LABELS),
            ).tolist(),
        },
        "classification_report": deps["classification_report"](
            y_true,
            y_pred,
            labels=list(SUPPORTED_LABELS),
            output_dict=True,
            zero_division=0,
        ),
    }


def _safe_macro_roc_auc(roc_auc_score, y_true: list[str], y_score: list[list[float]]) -> float | None:
    aucs: list[float] = []
    for label_index, label in enumerate(SUPPORTED_LABELS):
        binary_true = [1 if value == label else 0 for value in y_true]
        if len(set(binary_true)) < 2:
            continue
        label_score = [row[label_index] for row in y_score]
        aucs.append(float(roc_auc_score(binary_true, label_score)))
    if not aucs:
        return None
    return sum(aucs) / len(aucs)


def _macro_brier_score_ovr(y_true: list[str], y_score: list[list[float]]) -> float:
    class_scores: list[float] = []
    for label_index, label in enumerate(SUPPORTED_LABELS):
        total = 0.0
        for row_label, score_row in zip(y_true, y_score):
            observed = 1.0 if row_label == label else 0.0
            total += (score_row[label_index] - observed) ** 2
        class_scores.append(total / len(y_true))
    return sum(class_scores) / len(class_scores)


def _multiclass_brier_score(y_true: list[str], y_score: list[list[float]]) -> float:
    total = 0.0
    for row_label, score_row in zip(y_true, y_score):
        total += sum(
            (score - (1.0 if row_label == label else 0.0)) ** 2
            for label, score in zip(SUPPORTED_LABELS, score_row)
        )
    return total / len(y_true)


def _ranking_metrics(frame, outputs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if "binary_target" not in frame.columns:
        return None
    rows = [
        {
            "relevance": int(frame.iloc[idx]["binary_target"]),
            "confidence": float(output["confidence_score"]),
        }
        for idx, output in enumerate(outputs)
    ]
    rows.sort(key=lambda row: row["confidence"], reverse=True)
    total_relevant = sum(row["relevance"] for row in rows)
    metrics: dict[str, Any] = {"total_relevant": int(total_relevant)}
    for k in (3, 5, 10):
        top = rows[: min(k, len(rows))]
        metrics[f"precision_at_{k}"] = (
            sum(row["relevance"] for row in top) / len(top) if top else 0.0
        )
        metrics[f"ndcg_at_{k}"] = _ndcg_at_k([row["relevance"] for row in rows], k)
    return metrics


def _ndcg_at_k(relevance: list[int], k: int) -> float:
    import math

    top = relevance[: min(k, len(relevance))]
    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(top))
    ideal = sorted(relevance, reverse=True)[: min(k, len(relevance))]
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal))
    return 0.0 if idcg == 0.0 else dcg / idcg


def _write_prediction_rows(predictions_csv: Path, frame, outputs: list[dict[str, Any]]) -> None:
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "label",
        "predicted_label",
        "final_confidence",
        "confidence_margin",
        "prob_god_class",
        "prob_feature_envy",
        "prob_long_method",
    ]
    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, output in enumerate(outputs):
            predictions = output["predictions"]
            writer.writerow(
                {
                    "sample_id": frame.iloc[idx].get("sample_id", ""),
                    "label": frame.iloc[idx]["label"],
                    "predicted_label": output["dominant_smell"],
                    "final_confidence": output["confidence_score"],
                    "confidence_margin": output["confidence_margin"],
                    "prob_god_class": predictions["GOD_CLASS"],
                    "prob_feature_envy": predictions["FEATURE_ENVY"],
                    "prob_long_method": predictions["LONG_METHOD"],
                }
            )


def _print_report(report: dict[str, Any], output_json: Path, predictions_csv: Path) -> None:
    primary = report["positive_smell_classification"]
    print("\nBACKD00R MoE Formal Test Evaluation")
    print("===================================")
    print(f"Test dataset: {report['evaluation_dataset']}")
    print(f"Full test samples: {report['sample_count']}")
    print(f"Primary positive-smell samples: {report['primary_sample_count']}")
    print("")
    print("Chapter 4 Ready Summary")
    print("-----------------------")
    print(f"Accuracy:              {primary['accuracy']:.4f}")
    print(f"Macro Precision:       {primary['macro_precision']:.4f}")
    print(f"Macro Recall:          {primary['macro_recall']:.4f}")
    print(f"Macro F1-Score:        {primary['macro_f1_score']:.4f}")
    print(f"Weighted Precision:    {primary['weighted_precision']:.4f}")
    print(f"Weighted Recall:       {primary['weighted_recall']:.4f}")
    print(f"Weighted F1-Score:     {primary['weighted_f1_score']:.4f}")
    print(f"Macro Brier Score OVR: {primary['macro_brier_score_ovr']:.4f}")
    print(f"Multiclass Brier:      {primary['multiclass_brier_score']:.4f}")
    roc_auc = primary["macro_roc_auc_ovr"]
    print(f"Macro ROC-AUC OVR:   {'N/A' if roc_auc is None else f'{roc_auc:.4f}'}")
    print("")
    print("Per-Class Metrics")
    print("-----------------")
    for label, values in primary["per_class"].items():
        print(
            f"{label:13} precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} "
            f"f1={values['f1_score']:.4f} "
            f"support={values['support']}"
        )
    print("")
    print("Confusion Matrix")
    print("----------------")
    print("Rows=true labels, columns=predicted labels")
    print("Labels:", ", ".join(primary["confusion_matrix"]["labels"]))
    for row in primary["confusion_matrix"]["matrix"]:
        print(row)
    if report["confidence_ranking"] is not None:
        print("")
        print("Confidence Ranking Metrics")
        print("--------------------------")
        ranking = report["confidence_ranking"]
        print(f"Relevant smell-present rows: {ranking['total_relevant']}")
        for k in (3, 5, 10):
            print(
                f"P@{k}={ranking[f'precision_at_{k}']:.4f}, "
                f"NDCG@{k}={ranking[f'ndcg_at_{k}']:.4f}"
            )
    print("")
    print(f"Wrote evaluation report: {output_json}")
    print(f"Wrote prediction rows:   {predictions_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        default=DEFAULT_ARTIFACTS_DIR / "test_normalized.csv",
        type=Path,
    )
    parser.add_argument(
        "--artifacts-dir",
        default=DEFAULT_ARTIFACTS_DIR / "models",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_ARTIFACTS_DIR / "evaluation_report.json",
        type=Path,
    )
    parser.add_argument(
        "--predictions-out",
        default=DEFAULT_ARTIFACTS_DIR / "test_predictions.csv",
        type=Path,
    )
    args = parser.parse_args()
    evaluate_moe_pipeline(args.test, args.artifacts_dir, args.out, args.predictions_out)


if __name__ == "__main__":
    main()
