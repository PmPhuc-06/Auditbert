from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from engine_common import (
    AUDITBERT_BATCH_SIZE,
    AUDITBERT_CHECKPOINT,
    AUDITBERT_EPOCHS,
    AUDITBERT_LR,
    AUDITBERT_MAX_LEN,
    AUDITBERT_MODEL_NAME,
    AUDITBERT_PATIENCE,
    AUDITBERT_THRESHOLD_METRIC,
    BASELINE_CHECKPOINT,
    MFINBERT_BATCH_SIZE,
    MFINBERT_CHECKPOINT,
    MFINBERT_EPOCHS,
    MFINBERT_LR,
    MFINBERT_MAX_LEN,
    MFINBERT_MODEL_NAME,
    MFINBERT_PATIENCE,
    MFINBERT_THRESHOLD_METRIC,
    PHOBERT_BATCH_SIZE,
    PHOBERT_CHECKPOINT,
    PHOBERT_EPOCHS,
    PHOBERT_LR,
    PHOBERT_MAX_LEN,
    PHOBERT_MODEL_NAME,
    PHOBERT_PATIENCE,
    PHOBERT_THRESHOLD_METRIC,
    RULE_RED_FLAG_BONUS,
    luu_manifest_split,
    tao_split_mac_dinh_tu_json,
)
from engine_metadata import xuat_metadata_schema_json, xuat_metadata_schema_markdown


DEFAULT_DATASET = "samples.jsonl"
DEFAULT_OUTPUT_DIR = "paper_artifacts"
DEFAULT_FREEZE_LAYERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reproducibility artifacts for AuditBERT-VN paper tables.",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="JSON/JSONL dataset used for train/val/test splitting.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for schema/config/threshold artifacts.",
    )
    return parser.parse_args()


def load_transformer_meta(path: str) -> dict[str, object]:
    meta_path = Path(path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_baseline_threshold(path: str) -> dict[str, object]:
    ckpt = Path(path)
    if not ckpt.exists():
        return {}
    payload = pickle.loads(ckpt.read_bytes())
    return {
        "threshold": float(payload.get("threshold", 0.5)),
        "threshold_metric": str(payload.get("threshold_metric", "f2")),
        "best_val_metrics": payload.get("best_val_metrics", {}),
    }


def build_model_configs() -> dict[str, dict[str, object]]:
    return {
        "baseline": {
            "checkpoint_path": BASELINE_CHECKPOINT,
            "model_name": "tfidf_logistic_regression",
            "hybrid_metadata_enabled": False,
            "threshold_metric": "f2",
        },
        "phobert": {
            "checkpoint_path": PHOBERT_CHECKPOINT,
            "model_name": PHOBERT_MODEL_NAME,
            "batch_size": PHOBERT_BATCH_SIZE,
            "learning_rate": PHOBERT_LR,
            "epochs": PHOBERT_EPOCHS,
            "patience": PHOBERT_PATIENCE,
            "max_len": PHOBERT_MAX_LEN,
            "freeze_layers": DEFAULT_FREEZE_LAYERS,
            "hybrid_metadata_enabled": False,
            "threshold_metric": PHOBERT_THRESHOLD_METRIC,
        },
        "mfinbert": {
            "checkpoint_path": MFINBERT_CHECKPOINT,
            "model_name": MFINBERT_MODEL_NAME,
            "batch_size": MFINBERT_BATCH_SIZE,
            "learning_rate": MFINBERT_LR,
            "epochs": MFINBERT_EPOCHS,
            "patience": MFINBERT_PATIENCE,
            "max_len": MFINBERT_MAX_LEN,
            "freeze_layers": DEFAULT_FREEZE_LAYERS,
            "hybrid_metadata_enabled": False,
            "threshold_metric": MFINBERT_THRESHOLD_METRIC,
        },
        "auditbert": {
            "checkpoint_path": AUDITBERT_CHECKPOINT,
            "model_name": AUDITBERT_MODEL_NAME,
            "batch_size": AUDITBERT_BATCH_SIZE,
            "learning_rate": AUDITBERT_LR,
            "epochs": AUDITBERT_EPOCHS,
            "patience": AUDITBERT_PATIENCE,
            "max_len": AUDITBERT_MAX_LEN,
            "freeze_layers": DEFAULT_FREEZE_LAYERS,
            "hybrid_metadata_enabled": True,
            "threshold_metric": AUDITBERT_THRESHOLD_METRIC,
        },
    }


def build_thresholds(
    model_configs: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    thresholds = {
        "baseline": load_baseline_threshold(BASELINE_CHECKPOINT),
        "phobert": load_transformer_meta(PHOBERT_CHECKPOINT),
        "mfinbert": load_transformer_meta(MFINBERT_CHECKPOINT),
        "auditbert": load_transformer_meta(AUDITBERT_CHECKPOINT),
    }
    for model_name, threshold_info in thresholds.items():
        config = model_configs.get(model_name, {})
        if model_name == "baseline":
            threshold_info.setdefault("hybrid_metadata_enabled", False)
            threshold_info.setdefault("hybrid_metadata_requested", False)
            continue
        if "hybrid_metadata_enabled" in config:
            threshold_info["hybrid_metadata_enabled"] = bool(config["hybrid_metadata_enabled"])
            threshold_info["hybrid_metadata_requested"] = bool(config["hybrid_metadata_enabled"])
        for key in ("freeze_layers", "learning_rate", "batch_size", "epochs", "patience", "max_len", "model_name", "threshold_metric"):
            if key in config and key not in threshold_info:
                threshold_info[key] = config[key]
    return thresholds


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xuat_metadata_schema_json(str(output_dir / "metadata_schema.json"))
    xuat_metadata_schema_markdown(str(output_dir / "metadata_feature_list.md"))

    split, split_manifest = tao_split_mac_dinh_tu_json(
        args.dataset,
        ty_le_train=0.8,
        ty_le_val=0.1,
        seed=42,
    )
    split_manifest["summary"] = split.summary()
    luu_manifest_split(split_manifest, output_dir / "split_manifest.json")

    model_configs = build_model_configs()
    for model_name, config in model_configs.items():
        config_path = output_dir / f"train_config_{model_name}.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    thresholds = build_thresholds(model_configs)
    (output_dir / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rule_postprocessing = {
        "location": [
            "engine_common.py::tao_score_breakdown",
            "engine_baseline.py::predict",
            "engine_transformer.py::predict",
        ],
        "decision_scope": "pipeline_inference_and_evaluation",
        "rule_bonus_per_flag": RULE_RED_FLAG_BONUS,
        "rule_definition": "final_score = min(1.0, hybrid_metadata_score + rule_bonus_per_flag * len(red_flags))",
        "red_flag_source": "engine_common.py::phat_hien_co_do",
    }
    (output_dir / "rule_postprocessing.json").write_text(
        json.dumps(rule_postprocessing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved paper artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
