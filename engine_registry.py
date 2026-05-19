from __future__ import annotations

from dataclasses import dataclass

from engine_auditbert import MoHinhGianLanAuditBERT
from engine_baseline import MoHinhGianLan
from engine_common import (
    AUDITBERT_CHECKPOINT,
    BASELINE_CHECKPOINT,
    MFINBERT_CHECKPOINT,
    PHOBERT_CHECKPOINT,
)
from engine_mfinbert import MoHinhGianLanMFinBERT
from engine_phobert import MoHinhGianLanPhoBERT


MODEL_BASELINE = "baseline"
MODEL_PHOBERT = "phobert"
MODEL_MFINBERT = "mfinbert"
MODEL_AUDITBERT = "auditbert"


@dataclass(frozen=True)
class ModelRegistryEntry:
    key: str
    label: str
    description: str
    model_class: type
    is_transformer: bool = False
    checkpoint_path: str | None = None


MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    MODEL_BASELINE: ModelRegistryEntry(
        key=MODEL_BASELINE,
        label="Baseline",
        description=(
            "Classical benchmark implemented in-project with TF-IDF + "
            "Logistic Regression for risk-screening; not a public pretrained checkpoint."
        ),
        model_class=MoHinhGianLan,
        is_transformer=False,
        checkpoint_path=BASELINE_CHECKPOINT,
    ),
    MODEL_PHOBERT: ModelRegistryEntry(
        key=MODEL_PHOBERT,
        label="PhoBERT",
        description=(
            "Public pretrained vinai/phobert-base backbone, fine-tuned as a "
            "Vietnamese risk-screening benchmark without hybrid metadata."
        ),
        model_class=MoHinhGianLanPhoBERT,
        is_transformer=True,
        checkpoint_path=PHOBERT_CHECKPOINT,
    ),
    MODEL_MFINBERT: ModelRegistryEntry(
        key=MODEL_MFINBERT,
        label="MFinBERT",
        description=(
            "Public pretrained sonnv/MFinBERT backbone, fine-tuned as a "
            "cross-lingual finance stress test without hybrid metadata."
        ),
        model_class=MoHinhGianLanMFinBERT,
        is_transformer=True,
        checkpoint_path=MFINBERT_CHECKPOINT,
    ),
    MODEL_AUDITBERT: ModelRegistryEntry(
        key=MODEL_AUDITBERT,
        label="AuditBERT-VN",
        description=(
            "Project-developed final model: PhoBERT backbone + hybrid metadata "
            "late fusion + rule-aware risk-screening in a single checkpoint."
        ),
        model_class=MoHinhGianLanAuditBERT,
        is_transformer=True,
        checkpoint_path=AUDITBERT_CHECKPOINT,
    ),
}

MODEL_CHOICES = tuple(MODEL_REGISTRY.keys())
TRANSFORMER_MODEL_CHOICES = tuple(
    key for key, entry in MODEL_REGISTRY.items() if entry.is_transformer
)
MODEL_QUERY_DESCRIPTION = "Một trong: " + ", ".join(MODEL_CHOICES)


def lay_muc_registry(loai: str) -> ModelRegistryEntry:
    if loai not in MODEL_REGISTRY:
        raise KeyError(f"Model không được hỗ trợ: {loai}")
    return MODEL_REGISTRY[loai]


def lay_lop_mo_hinh(loai: str) -> type:
    return lay_muc_registry(loai).model_class


def tao_mo_hinh_theo_loai(loai: str):
    return lay_lop_mo_hinh(loai)()


def la_model_transformer(loai: str) -> bool:
    return lay_muc_registry(loai).is_transformer


def danh_sach_model_ho_tro() -> list[str]:
    return list(MODEL_CHOICES)


def tao_ket_qua_du_doan(
    loai: str,
    prediction,
    model_source: str = "unknown",
    top_terms_method: str = "unknown",
) -> dict:
    entry = lay_muc_registry(loai)
    risk_label_int = int(getattr(prediction, "risk_label", None) or prediction.label)
    score_breakdown = dict(getattr(prediction, "score_breakdown", {}) or {})

    def _prediction_float(name: str, fallback: float) -> float:
        value = getattr(prediction, name, None)
        if value is None:
            value = fallback
        return float(value)

    return {
        "model": loai,
        "model_label": entry.label,
        "model_source": model_source,
        "top_terms_method": top_terms_method,
        "label": "Cảnh báo rủi ro" if prediction.label == 1 else "Bình thường",
        "legacy_label": "Gian lận" if prediction.label == 1 else "Bình thường",
        "label_int": prediction.label,
        "risk_label": "Rủi ro cao" if risk_label_int == 1 else "Rủi ro thấp",
        "risk_label_int": risk_label_int,
        "muc_do_rui_ro": prediction.muc_do_rui_ro,
        "fraud_probability": round(prediction.probability_fraud, 4),
        "fraud_risk_probability": round(prediction.probability_fraud, 4),
        "non_fraud_probability": round(prediction.probability_non_fraud, 4),
        "model_fraud_probability": round(prediction.model_probability_fraud, 4),
        "raw_text_probability_fraud": round(
            _prediction_float("raw_text_probability_fraud", prediction.model_probability_fraud),
            4,
        ),
        "raw_text_probability_non_fraud": round(
            _prediction_float("raw_text_probability_non_fraud", prediction.model_probability_non_fraud),
            4,
        ),
        "hybrid_probability_fraud": round(
            _prediction_float("hybrid_probability_fraud", prediction.model_probability_fraud),
            4,
        ),
        "hybrid_probability_non_fraud": round(
            _prediction_float("hybrid_probability_non_fraud", prediction.model_probability_non_fraud),
            4,
        ),
        "threshold_used": round(prediction.threshold_used, 4),
        "chat_luong_van_ban": round(prediction.chat_luong_van_ban, 4),
        "red_flags": prediction.red_flags,
        "risk_signals": list(getattr(prediction, "risk_signals", prediction.red_flags)),
        "score_breakdown": score_breakdown,
        "score_semantics": {
            "raw_text_probability_fraud": "raw_backbone_probability",
            "hybrid_probability_fraud": "after_hybrid_metadata_head",
            "fraud_probability": "after_rule_based_postprocessing",
        },
        "explanation": prediction.explanation,
        "top_terms": prediction.top_terms,
        "van_ban_sach": prediction.van_ban_sach,
        "doan_nghi_ngo": prediction.doan_nghi_ngo,
        "metadata_features": {
            key: round(float(value), 4)
            for key, value in dict(getattr(prediction, "metadata_features", {}) or {}).items()
        },
        "bias_flags": getattr(prediction, "bias_flags", []),
        "explainability": getattr(prediction, "explainability", {}),
    }
