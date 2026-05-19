#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from engine_auditbert import MoHinhGianLanAuditBERT
from engine_common import tai_du_lieu_json, tim_nguong_toi_uu
from engine_metadata import (
    HYBRID_METADATA_FEATURE_NAMES,
    trich_xuat_metadata_features,
    vector_hoa_metadata_features,
)
from engine_phobert import MoHinhGianLanPhoBERT


DEFAULT_TRAIN = "auditbert_split_train.jsonl"
DEFAULT_VAL = "auditbert_split_val.jsonl"
DEFAULT_TEST = "auditbert_split_test.jsonl"
DEFAULT_OUTPUT_DIR = "paper_artifacts"
OUTPUT_JSON_NAME = "ablation_results.json"
OUTPUT_MD_NAME = "table_ablation.md"
OUTPUT_CSV_NAME = "table_ablation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AuditBERT-VN ablation study on the official train/val/test split.",
    )
    parser.add_argument("--train-jsonl", default=DEFAULT_TRAIN, help="Training split JSONL.")
    parser.add_argument("--val-jsonl", default=DEFAULT_VAL, help="Validation split JSONL.")
    parser.add_argument("--test-jsonl", default=DEFAULT_TEST, help="Test split JSONL.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for ablation tables and JSON artifacts.",
    )
    return parser.parse_args()


def compute_metrics(
    y_true: list[int],
    scores: list[float],
    threshold: float,
) -> dict[str, object]:
    y_scores = np.asarray(scores, dtype=float)
    y_pred = (y_scores >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    try:
        auc_roc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc_roc = 0.0
    try:
        auprc = float(average_precision_score(y_true, y_scores))
    except Exception:
        auprc = 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "auprc": auprc,
        "auc_roc": auc_roc,
        "mcc": mcc,
        "threshold_used": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def score_documents(model, texts: list[str]) -> list[dict[str, object]]:
    if hasattr(model, "score_texts_batch"):
        return list(model.score_texts_batch(texts))
    return [dict(model.score_text(text)) for text in texts]


def load_split(path: str) -> tuple[list[str], list[int]]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return tai_du_lieu_json(str(split_path))


def build_hybrid_head_evidence(
    y_true: list[int],
    raw_scores: list[float],
    final_scores: list[float],
    threshold: float,
) -> dict[str, object]:
    raw_arr = np.asarray(raw_scores, dtype=float)
    final_arr = np.asarray(final_scores, dtype=float)
    raw_pred = (raw_arr >= threshold).astype(int)
    final_pred = (final_arr >= threshold).astype(int)
    changed = np.where(raw_pred != final_pred)[0].tolist()
    return {
        "mean_abs_score_shift": float(np.mean(np.abs(final_arr - raw_arr))),
        "max_abs_score_shift": float(np.max(np.abs(final_arr - raw_arr))),
        "raw_f2_at_final_threshold": float(fbeta_score(y_true, raw_pred, beta=2, zero_division=0)),
        "final_f2_at_final_threshold": float(fbeta_score(y_true, final_pred, beta=2, zero_division=0)),
        "changed_decisions": len(changed),
        "changed_examples": [
            {
                "index": int(idx + 1),
                "label": int(y_true[idx]),
                "raw_model_score": round(float(raw_arr[idx]), 6),
                "final_score": round(float(final_arr[idx]), 6),
                "raw_pred": int(raw_pred[idx]),
                "final_pred": int(final_pred[idx]),
            }
            for idx in changed[:10]
        ],
    }


def evaluate_phobert_only(
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> dict[str, object]:
    model = MoHinhGianLanPhoBERT(hybrid_metadata_enabled=False)
    if not model._load_checkpoint():
        raise RuntimeError("Missing PhoBERT checkpoint.")
    val_scores = model._predict_raw_scores_batch(val_texts)
    threshold, _ = tim_nguong_toi_uu(val_labels, val_scores, metric="f2")
    test_scores = model._predict_raw_scores_batch(test_texts)
    result = compute_metrics(test_labels, test_scores, threshold)
    result["score_layer"] = "raw_backbone"
    return result


def evaluate_metadata_only(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> dict[str, object]:
    def to_matrix(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            feature_map = trich_xuat_metadata_features(text)
            rows.append(vector_hoa_metadata_features(feature_map, HYBRID_METADATA_FEATURE_NAMES))
        return np.asarray(rows, dtype=float)

    x_train = to_matrix(train_texts)
    x_val = to_matrix(val_texts)
    x_test = to_matrix(test_texts)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    classifier.fit(x_train_scaled, train_labels)

    val_scores = classifier.predict_proba(x_val_scaled)[:, 1].astype(float).tolist()
    threshold, _ = tim_nguong_toi_uu(val_labels, val_scores, metric="f2")
    test_scores = classifier.predict_proba(x_test_scaled)[:, 1].astype(float).tolist()
    result = compute_metrics(test_labels, test_scores, threshold)

    ranked_features = list(zip(HYBRID_METADATA_FEATURE_NAMES, classifier.coef_[0].astype(float).tolist()))
    ranked_features.sort(key=lambda item: abs(item[1]), reverse=True)
    result["top_features"] = [
        {
            "feature": name,
            "coefficient": round(float(coef), 6),
            "direction": "risk_up" if coef > 0 else "risk_down",
        }
        for name, coef in ranked_features[:5]
    ]
    result["zscore_source"] = {
        "fit_on": "train_split_only",
        "feature_count": len(HYBRID_METADATA_FEATURE_NAMES),
    }
    return result


def evaluate_auditbert_full(
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> dict[str, object]:
    model = MoHinhGianLanAuditBERT(hybrid_metadata_enabled=True)
    if not model._load_checkpoint():
        raise RuntimeError("Missing AuditBERT-VN checkpoint.")

    val_breakdowns = score_documents(model, val_texts)
    test_breakdowns = score_documents(model, test_texts)
    val_scores = [float(item["final_score"]) for item in val_breakdowns]
    threshold, _ = tim_nguong_toi_uu(val_labels, val_scores, metric="f2")
    test_scores = [float(item["final_score"]) for item in test_breakdowns]
    result = compute_metrics(test_labels, test_scores, threshold)
    result["score_layer"] = "hybrid_metadata_head"
    result["hybrid_head_evidence"] = build_hybrid_head_evidence(
        test_labels,
        [float(item["raw_model_score"]) for item in test_breakdowns],
        test_scores,
        threshold,
    )
    return result


def write_outputs(payload: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / OUTPUT_JSON_NAME
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    variants = dict(payload.get("variants", {}))
    csv_lines = [
        "variant,precision,recall,f1,f2,auprc,auc_roc,mcc,threshold_used",
    ]
    md_lines = [
        "# Ablation Study for AuditBERT-VN",
        "",
        "| Variant | Precision | Recall | F1 | F2 | AUPRC | AUC-ROC | MCC | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("phobert_only", "PhoBERT-only"),
        ("metadata_only", "Metadata-only"),
        ("auditbert_full", "AuditBERT-VN (Full)"),
    ]:
        result = variants.get(key, {})
        csv_lines.append(
            ",".join(
                [
                    key,
                    *[f"{float(result[metric]):.4f}" for metric in ("precision", "recall", "f1", "f2", "auprc", "auc_roc", "mcc")],
                    f"{float(result['threshold_used']):.4f}",
                ]
            )
        )
        md_lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    *[f"{float(result[metric]):.4f}" for metric in ("precision", "recall", "f1", "f2", "auprc", "auc_roc", "mcc")],
                    f"{float(result['threshold_used']):.4f}",
                ]
            )
            + " |"
        )

    (output_dir / OUTPUT_CSV_NAME).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (output_dir / OUTPUT_MD_NAME).write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    train_texts, train_labels = load_split(args.train_jsonl)
    val_texts, val_labels = load_split(args.val_jsonl)
    test_texts, test_labels = load_split(args.test_jsonl)

    phobert_only = evaluate_phobert_only(val_texts, val_labels, test_texts, test_labels)
    metadata_only = evaluate_metadata_only(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
    )
    auditbert_full = evaluate_auditbert_full(val_texts, val_labels, test_texts, test_labels)

    payload = {
        "data": {
            "train_jsonl": str(Path(args.train_jsonl).resolve()),
            "val_jsonl": str(Path(args.val_jsonl).resolve()),
            "test_jsonl": str(Path(args.test_jsonl).resolve()),
            "train_size": len(train_texts),
            "val_size": len(val_texts),
            "test_size": len(test_texts),
            "test_positives": int(sum(test_labels)),
            "test_negatives": int(len(test_labels) - sum(test_labels)),
        },
        "variants": {
            "phobert_only": phobert_only,
            "metadata_only": metadata_only,
            "auditbert_full": auditbert_full,
        },
        "delta": {
            "hybrid_metadata_head_vs_phobert_only": {
                "delta_f2": round(float(auditbert_full["f2"]) - float(phobert_only["f2"]), 6),
                "delta_auc_roc": round(float(auditbert_full["auc_roc"]) - float(phobert_only["auc_roc"]), 6),
                "delta_mcc": round(float(auditbert_full["mcc"]) - float(phobert_only["mcc"]), 6),
            },
            "full_model_vs_metadata_only": {
                "delta_f2": round(float(auditbert_full["f2"]) - float(metadata_only["f2"]), 6),
                "delta_auc_roc": round(float(auditbert_full["auc_roc"]) - float(metadata_only["auc_roc"]), 6),
                "delta_mcc": round(float(auditbert_full["mcc"]) - float(metadata_only["mcc"]), 6),
            },
        },
    }
    output_dir = Path(args.output_dir)
    write_outputs(payload, output_dir)

    print("=" * 88)
    print("ABLATION STUDY")
    print("=" * 88)
    print(f"Train={len(train_texts)} | Val={len(val_texts)} | Test={len(test_texts)}")
    print(f"Fraud test samples: {sum(test_labels)} / {len(test_labels)}")
    print("-" * 88)
    for key, label in [
        ("phobert_only", "PhoBERT-only"),
        ("metadata_only", "Metadata-only"),
        ("auditbert_full", "AuditBERT-VN (Full)"),
    ]:
        result = payload["variants"][key]
        print(
            f"{label:<22} "
            f"Precision={float(result['precision']):.4f} "
            f"Recall={float(result['recall']):.4f} "
            f"F1={float(result['f1']):.4f} "
            f"F2={float(result['f2']):.4f} "
            f"AUPRC={float(result['auprc']):.4f} "
            f"AUC={float(result['auc_roc']):.4f} "
            f"MCC={float(result['mcc']):.4f}"
        )
    print("-" * 88)
    print(
        "Hybrid metadata head vs PhoBERT-only: "
        f"delta F2={payload['delta']['hybrid_metadata_head_vs_phobert_only']['delta_f2']:+.4f}, "
        f"delta AUC={payload['delta']['hybrid_metadata_head_vs_phobert_only']['delta_auc_roc']:+.4f}, "
        f"delta MCC={payload['delta']['hybrid_metadata_head_vs_phobert_only']['delta_mcc']:+.4f}"
    )
    print(
        "Full model vs Metadata-only: "
        f"delta F2={payload['delta']['full_model_vs_metadata_only']['delta_f2']:+.4f}, "
        f"delta AUC={payload['delta']['full_model_vs_metadata_only']['delta_auc_roc']:+.4f}, "
        f"delta MCC={payload['delta']['full_model_vs_metadata_only']['delta_mcc']:+.4f}"
    )
    print(f"Saved ablation artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
