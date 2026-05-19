from __future__ import annotations

from collections import defaultdict

from engine_common import danh_gia_du_doan
from engine_metadata import trich_xuat_metadata_features


def tao_bias_flags_tu_features(feature_map: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if feature_map.get("ocr_quality", 1.0) < 0.55:
        flags.append("low_ocr_quality")
    if feature_map.get("digit_ratio", 0.0) > 0.12:
        flags.append("digit_heavy_text")
    if feature_map.get("log_char_len", 0.0) <= 5.0:
        flags.append("short_text")
    if feature_map.get("red_flag_count", 0.0) >= 2:
        flags.append("many_rule_flags")
    if feature_map.get("english_keyword_ratio", 0.0) > 0.08:
        flags.append("english_heavy_text")
    return flags


def danh_gia_bias_theo_nhom(
    texts: list[str],
    y_true: list[int],
    y_scores: list[float],
    *,
    threshold: float = 0.5,
    min_group_size: int = 5,
) -> dict:
    overall = danh_gia_du_doan(y_true, y_scores, threshold=threshold)
    grouped_indices: dict[str, list[int]] = defaultdict(list)

    for idx, text in enumerate(texts):
        features = trich_xuat_metadata_features(text)
        groups = tao_bias_flags_tu_features(features)
        if not groups:
            groups = ["default_population"]
        for group in groups:
            grouped_indices[group].append(idx)

    groups_report: dict[str, dict] = {}
    recall_gaps: list[float] = []
    precision_gaps: list[float] = []

    for group_name, indices in grouped_indices.items():
        if len(indices) < min_group_size:
            continue
        group_true = [y_true[i] for i in indices]
        group_scores = [y_scores[i] for i in indices]
        metrics = danh_gia_du_doan(group_true, group_scores, threshold=threshold)
        groups_report[group_name] = {
            "support": len(indices),
            "fraud_ratio": round(sum(group_true) / max(len(group_true), 1), 4),
            "metrics": {key: round(float(value), 4) for key, value in metrics.items()},
        }
        recall_gaps.append(abs(metrics.get("recall", 0.0) - overall.get("recall", 0.0)))
        precision_gaps.append(abs(metrics.get("precision", 0.0) - overall.get("precision", 0.0)))

    return {
        "threshold_used": round(float(threshold), 4),
        "overall_metrics": {key: round(float(value), 4) for key, value in overall.items()},
        "groups": groups_report,
        "max_recall_gap": round(max(recall_gaps, default=0.0), 4),
        "max_precision_gap": round(max(precision_gaps, default=0.0), 4),
    }
