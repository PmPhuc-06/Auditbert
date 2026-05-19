from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?84|0)(?:\d[\s.-]?){8,10}(?!\d)")
LONG_ID_PATTERN = re.compile(r"(?<!\d)\d{10,14}(?!\d)")


def tai_ban_ghi_dataset(json_path: str) -> list[dict]:
    raw = Path(json_path).read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("File dataset trống.")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        data = [json.loads(line) for line in lines]

    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            records = data["records"]
        elif "texts" in data and "labels" in data:
            records = [
                {"text": text, "label": label}
                for text, label in zip(data["texts"], data["labels"])
            ]
        else:
            raise ValueError("JSON object không có cấu trúc dataset được hỗ trợ.")
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Format dataset không hỗ trợ: {type(data)}")

    return [dict(record) for record in records]


def _ti_le_co_truong(records: list[dict], field_name: str) -> float:
    if not records:
        return 0.0
    count = sum(1 for record in records if str(record.get(field_name, "")).strip())
    return count / len(records)


def _scan_pii(text: str) -> dict[str, int]:
    return {
        "emails": len(EMAIL_PATTERN.findall(text)),
        "phones": len(PHONE_PATTERN.findall(text)),
        "long_ids": len(LONG_ID_PATTERN.findall(text)),
    }


def _cohen_kappa_from_records(records: list[dict]) -> float | None:
    try:
        from sklearn.metrics import cohen_kappa_score
    except Exception:
        return None

    candidate_pairs = [
        ("labeler_a", "labeler_b"),
        ("label_manual", "label_ai"),
        ("label_reviewer", "label"),
    ]
    for left, right in candidate_pairs:
        values_left = []
        values_right = []
        for record in records:
            if left in record and right in record:
                values_left.append(record[left])
                values_right.append(record[right])
        if values_left and len(values_left) == len(values_right):
            try:
                return float(cohen_kappa_score(values_left, values_right))
            except Exception:
                continue
    return None


def kiem_tra_governance_dataset(json_path: str) -> dict:
    records = tai_ban_ghi_dataset(json_path)
    texts = [str(record.get("text", "")) for record in records]
    labels = [record.get("label") for record in records if "label" in record]

    duplicates = len(texts) - len(set(texts))
    pii_counter = Counter()
    for text in texts:
        pii_counter.update(_scan_pii(text))

    fraud_count = sum(int(label) for label in labels if str(label).strip())
    total = len(records)
    minority_ratio = min(fraud_count, max(total - fraud_count, 0)) / max(total, 1)
    kappa = _cohen_kappa_from_records(records)

    step_1 = {
        "status": "ok" if _ti_le_co_truong(records, "source") >= 0.5 or _ti_le_co_truong(records, "doc_id") >= 0.5 else "warning",
        "collection_field_coverage": {
            "source": round(_ti_le_co_truong(records, "source"), 4),
            "doc_id": round(_ti_le_co_truong(records, "doc_id"), 4),
            "version": round(_ti_le_co_truong(records, "version"), 4),
            "collected_at": round(_ti_le_co_truong(records, "collected_at"), 4),
        },
    }
    step_2 = {
        "status": "ok" if sum(pii_counter.values()) == 0 else "warning",
        "privacy_field_coverage": {
            "anonymized": round(_ti_le_co_truong(records, "anonymized"), 4),
            "privacy_reviewed": round(_ti_le_co_truong(records, "privacy_reviewed"), 4),
        },
        "pii_scan": dict(pii_counter),
    }
    step_3 = {
        "status": "ok" if kappa is not None or total > 0 else "warning",
        "label_count": len(labels),
        "cohen_kappa": round(kappa, 4) if kappa is not None else None,
        "label_fields_detected": [
            field_name
            for field_name in ("label", "labeler_a", "labeler_b", "label_manual", "label_ai")
            if any(field_name in record for record in records)
        ],
    }

    ready_for_training = (
        total > 0
        and len(labels) == total
        and duplicates / max(total, 1) <= 0.1
        and sum(pii_counter.values()) == 0
        and minority_ratio >= 0.1
    )
    step_4 = {
        "status": "ok" if ready_for_training else "warning",
        "total_records": total,
        "duplicates": duplicates,
        "fraud_ratio": round(fraud_count / max(total, 1), 4),
        "minority_ratio": round(minority_ratio, 4),
        "ready_for_training": ready_for_training,
    }

    return {
        "dataset_path": str(Path(json_path).resolve()),
        "step_1_collect_data": step_1,
        "step_2_ethics_privacy": step_2,
        "step_3_label_quality": step_3,
        "step_4_ready_gate": step_4,
    }
