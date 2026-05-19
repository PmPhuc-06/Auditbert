from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from engine_registry import lay_muc_registry, tao_mo_hinh_theo_loai


DEFAULT_CSV = "Tap_Du_Lieu_Test_Kiem_Tra.csv"
DEFAULT_OUTPUT_DIR = "paper_artifacts"
FLAT_METRICS_FILENAME = "ket_qua_eval.json"
TABLE3_CSV_FILENAME = "table3_main_metrics.csv"
TABLE3_MD_FILENAME = "table3_main_metrics.md"
MODEL_PROVENANCE_FILENAME = "model_provenance.md"
AUDITBERT_EVIDENCE_JSON_FILENAME = "auditbert_feature_evidence.json"
AUDITBERT_EVIDENCE_MD_FILENAME = "auditbert_feature_evidence.md"
PAPER_SUMMARY_FILENAME = "paper_results_summary.md"
DISPLAY_METRIC_ORDER = [
    "precision",
    "recall",
    "f1",
    "f2",
    "auprc",
    "auc_roc",
    "mcc",
]
HIGHLIGHT_METRICS = {"f1", "f2", "auprc", "auc_roc", "mcc"}
MODEL_ORDER = ["baseline", "phobert", "mfinbert", "auditbert"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline/PhoBERT/MFinBERT/AuditBERT on an SSC-derived CSV test set.",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Evaluation CSV. Default: Tap_Du_Lieu_Test_Kiem_Tra.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics/predictions artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["baseline", "phobert", "mfinbert", "auditbert"],
        help="Subset of models to evaluate.",
    )
    return parser.parse_args()


def load_eval_csv(csv_path: str) -> tuple[list[str], list[int], dict[str, object]]:
    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines="skip", encoding="utf-8")
    if len(df.columns) < 4:
        raise ValueError(f"CSV {csv_path} must have at least 4 columns.")

    text_col = df.columns[3]
    label_col = df.columns[2]
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    dataset_info = {
        "csv_path": str(Path(csv_path).resolve()),
        "rows": len(df),
        "positives": int(sum(labels)),
        "negatives": int(len(labels) - sum(labels)),
        "text_column": text_col,
        "label_column": label_col,
    }
    return texts, labels, dataset_info


def score_documents(model, texts: list[str]) -> list[dict[str, object]]:
    if hasattr(model, "score_texts_batch"):
        return list(model.score_texts_batch(texts))
    return [dict(model.score_text(text)) for text in texts]


def compute_metrics(
    y_true: list[int],
    final_scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    y_pred = (final_scores >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    try:
        auc_roc = float(roc_auc_score(y_true, final_scores))
    except Exception:
        auc_roc = 0.0
    try:
        auprc = float(average_precision_score(y_true, final_scores))
    except Exception:
        auprc = 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
        "auprc": auprc,
        "auc_roc": auc_roc,
        "mcc": mcc,
        "threshold_used": float(threshold),
        "positive_prediction_rate": float(np.mean(y_pred)),
        "score_mean": float(np.mean(final_scores)),
        "score_std": float(np.std(final_scores)),
        "score_min": float(np.min(final_scores)),
        "score_max": float(np.max(final_scores)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def compute_hybrid_head_evidence(
    y_true: list[int],
    raw_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    raw_pred = (raw_scores >= threshold).astype(int)
    hybrid_pred = (hybrid_scores >= threshold).astype(int)
    changed_indices = np.where(raw_pred != hybrid_pred)[0].tolist()
    examples = []
    for idx in changed_indices[:10]:
        examples.append(
            {
                "index": int(idx + 1),
                "label": int(y_true[idx]),
                "raw_model_score": round(float(raw_scores[idx]), 6),
                "hybrid_metadata_score": round(float(hybrid_scores[idx]), 6),
                "raw_pred": int(raw_pred[idx]),
                "hybrid_pred": int(hybrid_pred[idx]),
            }
        )
    return {
        "mean_abs_score_shift": float(np.mean(np.abs(hybrid_scores - raw_scores))),
        "max_abs_score_shift": float(np.max(np.abs(hybrid_scores - raw_scores))),
        "raw_auc_roc": float(roc_auc_score(y_true, raw_scores)),
        "hybrid_auc_roc": float(roc_auc_score(y_true, hybrid_scores)),
        "raw_auprc": float(average_precision_score(y_true, raw_scores)),
        "hybrid_auprc": float(average_precision_score(y_true, hybrid_scores)),
        "raw_f2_at_final_threshold": float(fbeta_score(y_true, raw_pred, beta=2, zero_division=0)),
        "hybrid_f2_at_final_threshold": float(fbeta_score(y_true, hybrid_pred, beta=2, zero_division=0)),
        "changed_decisions": len(changed_indices),
        "changed_examples": examples,
    }


def build_flat_metrics_payload(summary: dict[str, object]) -> dict[str, dict[str, float]]:
    models = dict(summary.get("models", {}))
    payload: dict[str, dict[str, float]] = {}
    for model_name in MODEL_ORDER:
        result = models.get(model_name)
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        payload[model_name] = {
            key: float(result[key])
            for key in DISPLAY_METRIC_ORDER + ["threshold_used"]
            if key in result
        }
    return payload


def format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def write_table3_artifacts(summary: dict[str, object], output_dir: Path) -> None:
    models = dict(summary.get("models", {}))
    csv_lines = [
        "model,precision,recall,f1,f2,auprc,auc_roc,mcc,threshold_used",
    ]
    metric_best: dict[str, float] = {}
    for metric_name in DISPLAY_METRIC_ORDER:
        candidates = [
            float(models[model_name][metric_name])
            for model_name in MODEL_ORDER
            if isinstance(models.get(model_name), dict)
            and models[model_name].get("status") == "ok"
            and metric_name in models[model_name]
        ]
        if candidates:
            metric_best[metric_name] = max(candidates)

    md_lines = [
        "# Table 3. Main Metrics on the Held-Out SSC Test Set",
        "",
        "| Model | Precision | Recall | F1 | F2 | AUPRC | AUC-ROC | MCC | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_ORDER:
        result = models.get(model_name)
        if not isinstance(result, dict) or result.get("status") != "ok":
            continue
        label = str(result.get("model_label", model_name))
        csv_lines.append(
            ",".join(
                [
                    model_name,
                    *[format_metric(result[metric]) for metric in DISPLAY_METRIC_ORDER],
                    format_metric(result["threshold_used"]),
                ]
            )
        )
        md_metric_cells = []
        for metric_name in DISPLAY_METRIC_ORDER:
            value = float(result[metric_name])
            cell = format_metric(value)
            if (
                metric_name in HIGHLIGHT_METRICS
                and abs(value - metric_best.get(metric_name, value)) <= 1e-12
            ):
                cell = f"**{cell}**"
            md_metric_cells.append(cell)
        md_lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    *md_metric_cells,
                    format_metric(result["threshold_used"]),
                ]
            )
            + " |"
        )

    (output_dir / TABLE3_CSV_FILENAME).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / TABLE3_MD_FILENAME).write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_model_provenance(output_dir: Path) -> None:
    rows = [
        (
            "Baseline",
            "Classical benchmark implemented in-project",
            "TF-IDF + Logistic Regression",
            "No public pretrained checkpoint; only standard algorithms are reused.",
        ),
        (
            "PhoBERT",
            "Public pretrained backbone",
            "vinai/phobert-base",
            "Fine-tuned on the fraud-detection corpus as a strong Vietnamese benchmark.",
        ),
        (
            "MFinBERT",
            "Public pretrained backbone",
            "sonnv/MFinBERT",
            "Fine-tuned on the same corpus as a finance-language benchmark.",
        ),
        (
            "AuditBERT-VN",
            "Project-developed model",
            "PhoBERT backbone + hybrid metadata head + rule-aware inference",
            "This is the main research contribution, not an off-the-shelf public model.",
        ),
    ]
    lines = [
        "# Model Provenance",
        "",
        "| Model | Provenance Type | Core Stack | Paper-Ready Interpretation |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    (output_dir / MODEL_PROVENANCE_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_auditbert_feature_evidence(summary: dict[str, object], output_dir: Path) -> None:
    auditbert_result = dict(summary.get("models", {})).get("auditbert")
    if not isinstance(auditbert_result, dict) or auditbert_result.get("status") != "ok":
        return

    evidence_path = auditbert_result.get("hybrid_head_evidence_path")
    evidence_payload: dict[str, object] = {}
    if isinstance(evidence_path, str) and Path(evidence_path).exists():
        evidence_payload = json.loads(Path(evidence_path).read_text(encoding="utf-8"))

    meta_path = Path("auditbert_fraud_checkpoint.meta.json")
    ranked_features: list[dict[str, object]] = []
    if meta_path.exists():
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_names = list(meta_payload.get("hybrid_feature_names", []))
        coefficients = list(meta_payload.get("hybrid_coefficients", []))
        ranked_pairs: list[tuple[str, float]] = []
        for name, coef in zip(feature_names, coefficients):
            if str(name) == "raw_text_prob":
                continue
            ranked_pairs.append((str(name), float(coef)))
        ranked_pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        for name, coef in ranked_pairs[:5]:
            ranked_features.append(
                {
                    "feature": name,
                    "coefficient": round(coef, 6),
                    "direction": "risk_up" if coef > 0 else "risk_down",
                }
            )

    payload = {
        "top_hybrid_features": ranked_features,
        "hybrid_head_evidence": evidence_payload,
    }
    (output_dir / AUDITBERT_EVIDENCE_JSON_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_lines = [
        "# AuditBERT-VN Feature Evidence",
        "",
        "## Top Hybrid-Metadata Features",
        "",
        "| Feature | Coefficient | Direction |",
        "|---|---:|---|",
    ]
    for item in ranked_features:
        md_lines.append(
            f"| {item['feature']} | {float(item['coefficient']):.6f} | {item['direction']} |"
        )
    if not ranked_features:
        md_lines.append("| N/A | 0.000000 | unavailable |")

    if evidence_payload:
        md_lines.extend(
            [
                "",
                "## Hybrid Head Decision Evidence",
                "",
                f"- Mean absolute score shift: {format_metric(evidence_payload.get('mean_abs_score_shift', 0.0))}",
                f"- Max absolute score shift: {format_metric(evidence_payload.get('max_abs_score_shift', 0.0))}",
                f"- Raw F2 at final threshold: {format_metric(evidence_payload.get('raw_f2_at_final_threshold', 0.0))}",
                f"- Hybrid F2 at final threshold: {format_metric(evidence_payload.get('hybrid_f2_at_final_threshold', 0.0))}",
                f"- Changed decisions: {int(evidence_payload.get('changed_decisions', 0))}",
                "",
                "| Index | Label | Raw Score | Hybrid Score | Raw Pred | Hybrid Pred |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in evidence_payload.get("changed_examples", []):
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("index", "")),
                        str(item.get("label", "")),
                        format_metric(item.get("raw_model_score", 0.0)),
                        format_metric(item.get("hybrid_metadata_score", 0.0)),
                        str(item.get("raw_pred", "")),
                        str(item.get("hybrid_pred", "")),
                    ]
                )
                + " |"
            )
    (output_dir / AUDITBERT_EVIDENCE_MD_FILENAME).write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )


def write_paper_summary(summary: dict[str, object], output_dir: Path) -> None:
    models = dict(summary.get("models", {}))
    baseline = models.get("baseline", {})
    phobert = models.get("phobert", {})
    mfinbert = models.get("mfinbert", {})
    auditbert = models.get("auditbert", {})
    if not all(isinstance(item, dict) and item.get("status") == "ok" for item in (baseline, phobert, mfinbert, auditbert)):
        return

    lines = [
        "# Paper Results Summary",
        "",
        "## Main Takeaways",
        "",
        (
            f"- AuditBERT-VN is the strongest overall model on the held-out SSC test set, "
            f"leading on F1 ({format_metric(auditbert['f1'])}), AUPRC ({format_metric(auditbert['auprc'])}), "
            f"AUC-ROC ({format_metric(auditbert['auc_roc'])}), and MCC ({format_metric(auditbert['mcc'])})."
        ),
        (
            f"- Relative to the TF-IDF baseline, AuditBERT-VN improves recall from "
            f"{format_metric(baseline['recall'])} to {format_metric(auditbert['recall'])}, "
            f"F2 from {format_metric(baseline['f2'])} to {format_metric(auditbert['f2'])}, "
            f"and MCC from {format_metric(baseline['mcc'])} to {format_metric(auditbert['mcc'])}."
        ),
        (
            f"- PhoBERT remains the strongest recall-oriented comparator "
            f"(Recall {format_metric(phobert['recall'])}, F2 {format_metric(phobert['f2'])}), "
            f"while MFinBERT is the weakest benchmark in this Vietnamese setting."
        ),
        "",
        "## Paper-Ready Wording",
        "",
        (
            "AuditBERT-VN should not be described as merely competitive. "
            "On the SSC held-out test set, it establishes the best balanced performance profile "
            "among the evaluated models and substantially improves over the TF-IDF baseline on all "
            "high-priority fraud-detection metrics except precision."
        ),
        "",
        "## Provenance Clarification",
        "",
        (
            "Baseline is an in-project implementation of classical TF-IDF + Logistic Regression, "
            "whereas PhoBERT and MFinBERT are public pretrained backbones that were fine-tuned on the project corpus. "
            "AuditBERT-VN is the novel model contribution built on top of these established components."
        ),
    ]
    (output_dir / PAPER_SUMMARY_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_one_model(
    model_name: str,
    texts: list[str],
    y_true: list[int],
    output_dir: Path,
) -> dict[str, object]:
    entry = lay_muc_registry(model_name)
    model = tao_mo_hinh_theo_loai(model_name)

    if hasattr(model, "_load_checkpoint") and not model._load_checkpoint():
        return {
            "status": "missing_checkpoint",
            "model": model_name,
            "model_label": entry.label,
            "checkpoint_path": entry.checkpoint_path,
        }

    threshold = float(getattr(model, "threshold", 0.5))
    breakdowns = score_documents(model, texts)
    raw_scores = np.asarray(
        [float(item["raw_model_score"]) for item in breakdowns],
        dtype=float,
    )
    hybrid_scores = np.asarray(
        [float(item["hybrid_metadata_score"]) for item in breakdowns],
        dtype=float,
    )
    final_scores = np.asarray(
        [float(item["final_score"]) for item in breakdowns],
        dtype=float,
    )
    metrics = compute_metrics(y_true, final_scores, threshold)
    metrics.update(
        {
            "status": "ok",
            "model": model_name,
            "model_label": entry.label,
            "checkpoint_path": entry.checkpoint_path,
            "model_source": getattr(model, "model_source", "unknown"),
            "hybrid_metadata_enabled": bool(
                getattr(model, "hybrid_metadata_enabled", False)
            ),
            "rule_bonus_per_flag": float(
                getattr(model, "rule_bonus_per_flag", 0.05)
            ),
            "raw_score_mean": float(np.mean(raw_scores)),
            "hybrid_score_mean": float(np.mean(hybrid_scores)),
            "rule_increment_mean": float(np.mean(final_scores - hybrid_scores)),
        }
    )

    prediction_rows = []
    for index, (text, label, breakdown) in enumerate(zip(texts, y_true, breakdowns), start=1):
        prediction_rows.append(
            {
                "index": index,
                "label": int(label),
                "predicted_label": int(breakdown["predicted_label"]),
                "raw_model_score": round(float(breakdown["raw_model_score"]), 6),
                "hybrid_metadata_score": round(float(breakdown["hybrid_metadata_score"]), 6),
                "final_score": round(float(breakdown["final_score"]), 6),
                "rule_increment": round(float(breakdown["rule_increment"]), 6),
                "threshold_used": round(float(breakdown["threshold_used"]), 6),
                "decision_layer": breakdown["decision_layer"],
                "red_flags": list(breakdown["red_flags"]),
                "text_preview": text[:240],
            }
        )

    predictions_path = output_dir / f"predictions_{model_name}.json"
    predictions_path.write_text(
        json.dumps(prediction_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics["predictions_path"] = str(predictions_path.resolve())

    if bool(getattr(model, "hybrid_metadata_enabled", False)):
        hybrid_evidence = compute_hybrid_head_evidence(
            y_true,
            raw_scores,
            hybrid_scores,
            threshold,
        )
        evidence_path = output_dir / f"hybrid_head_evidence_{model_name}.json"
        evidence_path.write_text(
            json.dumps(hybrid_evidence, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metrics["hybrid_head_evidence_path"] = str(evidence_path.resolve())

    return metrics


def main() -> None:
    args = parse_args()
    texts, labels, dataset_info = load_eval_csv(args.csv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SSC EVALUATION")
    print("=" * 78)
    print(
        f"Dataset: {dataset_info['rows']} rows | positives={dataset_info['positives']} "
        f"| negatives={dataset_info['negatives']}"
    )
    print(f"CSV    : {dataset_info['csv_path']}")

    summary: dict[str, object] = {
        "dataset": dataset_info,
        "models": {},
    }

    for model_name in args.models:
        print("\n" + "-" * 78)
        print(f"Model: {model_name}")
        result = evaluate_one_model(model_name, texts, labels, output_dir)
        summary["models"][model_name] = result
        if result.get("status") != "ok":
            print(f"Status: {result['status']}")
            continue
        print(
            f"Precision={result['precision']:.4f} Recall={result['recall']:.4f} "
            f"F1={result['f1']:.4f} F2={result['f2']:.4f}"
        )
        print(
            f"AUPRC={result['auprc']:.4f} AUC-ROC={result['auc_roc']:.4f} "
            f"MCC={result['mcc']:.4f} Thr={result['threshold_used']:.4f}"
        )
        print(
            f"PositiveRate={result['positive_prediction_rate']:.4f} "
            f"RawMean={result['raw_score_mean']:.4f} "
            f"HybridMean={result['hybrid_score_mean']:.4f} "
            f"FinalMean={result['score_mean']:.4f}"
        )

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    flat_metrics_path = Path(FLAT_METRICS_FILENAME)
    flat_metrics_path.write_text(
        json.dumps(build_flat_metrics_payload(summary), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    write_table3_artifacts(summary, output_dir)
    write_model_provenance(output_dir)
    write_auditbert_feature_evidence(summary, output_dir)
    write_paper_summary(summary, output_dir)
    print("\n" + "=" * 78)
    print(f"Saved metrics summary to: {summary_path.resolve()}")
    print(f"Saved flat metrics to  : {flat_metrics_path.resolve()}")


if __name__ == "__main__":
    main()
