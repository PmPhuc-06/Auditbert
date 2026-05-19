"""
AuditBERT-VN — mô hình sàng lọc cảnh báo rủi ro báo cáo tài chính.

Kiến trúc Feature Fusion (nhẹ, chỉ 1 model khi inference):
 • Backbone : PhoBERT (vinai/phobert-base) — tốt nhất cho tiếng Việt
 • Head      : Hybrid Metadata mở rộng với 20 features gồm:
     - 14 features gốc (thống kê, rule-based, language signal)
     - 6 features mới (financial_term_density_vn, financial_term_density_en,
       round_number_ratio, accounting_code_count, loss_keyword_count,
       abnormal_profit_signal)
 • Signal tài chính từ MFinBERT được proxy hóa thành feature
   "financial_term_density_en" — không cần load MFinBERT khi inference.
 • Signal từ Baseline TF-IDF được proxy hóa thành "round_number_ratio"
   và "financial_term_density_vn".

Kết quả: Chỉ cần load 1 checkpoint (~400 MB) khi chạy, không cần
chạy song song PhoBERT + MFinBERT + Baseline.
"""
from __future__ import annotations

from engine_common import (
    AUDITBERT_BATCH_SIZE,
    AUDITBERT_CHECKPOINT,
    AUDITBERT_EPOCHS,
    AUDITBERT_LR,
    AUDITBERT_MAX_LEN,
    AUDITBERT_MODEL_NAME,
    AUDITBERT_PATIENCE,
    AUDITBERT_THRESHOLD_METRIC,
)
from engine_transformer import FOCAL_GAMMA_DEFAULT, MoHinhGianLanTransformer


class MoHinhGianLanAuditBERT(MoHinhGianLanTransformer):
    """
    AuditBERT-VN — kiến trúc phân loại lai cho sàng lọc rủi ro BCTC Việt Nam.

    Tích hợp tinh hoa của 3 model:
      1. PhoBERT backbone  → Hiểu sâu ngữ nghĩa tiếng Việt (cấu trúc câu, văn phong kế toán)
      2. Baseline signal   → round_number_ratio (Benford's Law), financial_term_density_vn
      3. MFinBERT signal   → financial_term_density_en (proxy thuật ngữ tài chính Anh)

    Chỉ load 1 checkpoint khi inference — nhẹ và nhanh.
    """

    DISPLAY_NAME = "AuditBERT-VN"

    def __init__(
        self,
        model_name: str = AUDITBERT_MODEL_NAME,
        checkpoint_path: str = AUDITBERT_CHECKPOINT,
        max_len: int = AUDITBERT_MAX_LEN,
        batch_size: int = AUDITBERT_BATCH_SIZE,
        epochs: int = AUDITBERT_EPOCHS,
        learning_rate: float = AUDITBERT_LR,
        threshold_metric: str = AUDITBERT_THRESHOLD_METRIC,
        patience: int = AUDITBERT_PATIENCE,
        hybrid_metadata_enabled: bool = True,   # LUÔN BẬT — đây là điểm mấu chốt
        gamma: float = FOCAL_GAMMA_DEFAULT,
        ig_n_steps: int = 50,
        ig_internal_batch_size: int = 10,
    ) -> None:
        super().__init__(
            display_name=self.DISPLAY_NAME,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            max_len=max_len,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            threshold_metric=threshold_metric,
            patience=patience,
            hybrid_metadata_enabled=hybrid_metadata_enabled,
            gamma=gamma,
            ig_n_steps=ig_n_steps,
            ig_internal_batch_size=ig_internal_batch_size,
        )
