from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from engine_bias import tao_bias_flags_tu_features
from engine_common import (
    CHUNK_OVERLAP,
    DataSplit,
    PredictionResult,
    RULE_RED_FLAG_BONUS,
    danh_gia_du_doan,
    luu_split_ra_file,
    phat_hien_co_do,
    tai_du_lieu_json,
    tach_chunks_recursive,
    tao_score_breakdown,
    tao_score_breakdown_tu_van_ban,
    tien_xu_ly_day_du,
    tao_van_ban_hien_thi,
    tim_doan_nghi_ngo,
    tim_nguong_toi_uu,
    tinh_chat_luong_van_ban,
    xep_muc_do_rui_ro,
)
from engine_explainability import PerturbationTextExplainer
from engine_governance import kiem_tra_governance_dataset
from engine_metadata import (
    HYBRID_METADATA_FEATURE_NAMES,
    lam_tron_metadata_features,
    tao_hybrid_feature_vector,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import transformers

    transformers.logging.set_verbosity_error()
    TRANSFORMER_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    nn = SimpleNamespace(Module=object)
    DataLoader = None
    TorchDataset = object
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMER_AVAILABLE = False

try:
    from engine_explainability import IntegratedGradientsExplainer

    IG_AVAILABLE = True
except Exception:
    IG_AVAILABLE = False


FOCAL_GAMMA_DEFAULT = 1.0


class FocalLoss(nn.Module):
    """
    Focal Loss cho phân loại nhị phân 2 class (fraud / non-fraud).
    """

    def __init__(self, alpha: "torch.Tensor", gamma: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        import torch.nn.functional as F

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha.gather(0, targets)

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
        return loss.mean()


_TorchDatasetBase = TorchDataset if TRANSFORMER_AVAILABLE else object  # type: ignore[misc]


class _TransformerFraudDataset(_TorchDatasetBase):  # type: ignore[misc]
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {"text": self.texts[idx], "labels": self.labels[idx]}


def _tao_transformer_collate_fn(tokenizer, max_len: int):
    def collate(batch):
        texts = [item["text"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return collate


class MoHinhGianLanTransformer:
    """
    Core transformer-based fraud model dùng chung cho PhoBERT/MFinBERT.
    """

    DISPLAY_NAME = "Transformer"

    def __init__(
        self,
        *,
        display_name: str,
        model_name: str,
        checkpoint_path: str,
        max_len: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        threshold_metric: str,
        patience: int,
        hybrid_metadata_enabled: bool = True,
        gamma: float = FOCAL_GAMMA_DEFAULT,
        ig_n_steps: int = 50,
        ig_internal_batch_size: int = 10,
        freeze_layers: int = 8,
    ) -> None:
        # freeze_layers: số transformer encoder layers giữ cố định
        # (0 = fine-tune tất cả; 8 = chỉ fine-tune 4 layers cuối + head)
        self.freeze_layers = freeze_layers
        if not TRANSFORMER_AVAILABLE:
            raise RuntimeError(
                "Thiếu thư viện transformer. Hãy cài: pip install torch transformers sentencepiece"
            )

        self.display_name = display_name
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = self.checkpoint_path.with_suffix(".meta.json")
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.threshold_metric = threshold_metric
        self.patience = patience
        self.gamma = gamma
        self.ig_n_steps = ig_n_steps
        self.ig_internal_batch_size = ig_internal_batch_size
        self.chunk_max_chars = max(256, self.max_len * 3)
        self.is_trained = False
        self.threshold = 0.5
        self.best_val_metrics: dict[str, float] = {}
        self.model_source = "untrained"
        self.top_terms_method = "integrated_gradients"
        self.hybrid_metadata_requested = bool(hybrid_metadata_enabled)
        self.hybrid_metadata_enabled = bool(hybrid_metadata_enabled)
        self.hybrid_feature_names = ["raw_text_prob"] + HYBRID_METADATA_FEATURE_NAMES
        self.hybrid_scaler_mean: list[float] = []
        self.hybrid_scaler_scale: list[float] = []
        self.hybrid_coefficients: list[float] = []
        self.hybrid_intercept = 0.0
        self.hybrid_model_is_ready = False
        self.rule_bonus_per_flag = RULE_RED_FLAG_BONUS

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log(f"Device: {self.device}")

        local_only = self.checkpoint_path.exists()
        if local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

        self._log(f"Đang load tokenizer '{model_name}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_only,
        )
        self._log(f"Đang load model '{model_name}' ...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            attn_implementation="eager",
            local_files_only=local_only,
            use_safetensors=False,
        )
        self.model.to(self.device)

        self._ig_explainer: "IntegratedGradientsExplainer | None" = None

        # Áp dụng layer freezing ngay sau khi load model
        if self.freeze_layers > 0:
            self._freeze_backbone_layers(self.freeze_layers)

    def _freeze_backbone_layers(self, n_freeze: int) -> None:
        """
        Đóng băng n_freeze encoder layers đầu tiên của transformer.
        Chỉ fine-tune phần classifier head và các layers cuối.

        Giải thích: PhoBERT có 12 encoder layers. Nếu fine-tune tất cả
        cùng lúc với LR = 2e-5, các layers thấp (học kiến thức ngôn ngữ
        chung) sẽ bị phá vỡ → catastrophic forgetting → AUC sụp đổ.
        Giải pháp: chỉ unfreeze 4 layers cuối + classifier head.
        """
        frozen_count = 0
        # Đóng băng embeddings
        for param in self.model.base_model.embeddings.parameters():
            param.requires_grad = False
            frozen_count += 1

        # Đóng băng n_freeze encoder layers đầu
        encoder_layers = self.model.base_model.encoder.layer
        n_total = len(encoder_layers)
        n_freeze_actual = min(n_freeze, n_total)
        for i, layer in enumerate(encoder_layers):
            if i < n_freeze_actual:
                for param in layer.parameters():
                    param.requires_grad = False
                    frozen_count += 1

        self._log(
            f"Layer freezing: {n_freeze_actual}/{n_total} encoder layers đóng băng "
            f"| Chỉ fine-tune layers {n_freeze_actual}–{n_total - 1} + classifier head."
        )

    def _unfreeze_all_layers(self) -> None:
        """Giải phóng tất cả layers (gọi trước phase 2 fine-tuning)."""
        for param in self.model.parameters():
            param.requires_grad = True
        self._log("Đã giải phóng tất cả layers cho fine-tuning đầy đủ.")

    def _log(self, message: str) -> None:
        line = f"[{self.display_name}] {message}"
        try:
            print(line)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            fallback = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
            print(fallback)

    @property
    def ig_explainer(self) -> "IntegratedGradientsExplainer":
        if self._ig_explainer is None:
            if not IG_AVAILABLE:
                raise RuntimeError(
                    "Thiếu Captum hoặc engine_explainability.py.\n"
                    "Hãy cài: pip install captum\n"
                    "Và đảm bảo engine_explainability.py cùng thư mục."
                )
            if not self.is_trained:
                raise RuntimeError("Model chưa được huấn luyện. Gọi fit() trước.")
            self._ig_explainer = IntegratedGradientsExplainer(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                n_steps=self.ig_n_steps,
                internal_batch_size=self.ig_internal_batch_size,
            )
            self._log(f"IG Explainer khởi tạo (n_steps={self.ig_n_steps})")
        return self._ig_explainer

    @classmethod
    def load_dataset_json(cls, json_path: str) -> tuple[list[str], list[int]]:
        texts, labels = tai_du_lieu_json(json_path)
        n_fraud = sum(labels)
        print(
            f"[{getattr(cls, 'DISPLAY_NAME', 'Transformer')}] Dataset: {len(texts)} mau "
            f"({n_fraud} fraud / {len(labels) - n_fraud} non-fraud)"
        )
        return texts, labels

    @staticmethod
    def governance_report(json_path: str) -> dict:
        return kiem_tra_governance_dataset(json_path)

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        text_list = list(texts)
        label_list = list(labels)
        if self._load_checkpoint():
            if self.hybrid_metadata_enabled and not self.hybrid_model_is_ready:
                self._log("Checkpoint cũ chưa có hybrid metadata head → hiệu chỉnh bổ sung.")
                self._fit_hybrid_metadata_model(text_list, label_list)
                self._save_metadata()
            return
        self._train(text_list, label_list)

    def fit_from_json(self, json_path: str) -> None:
        texts, labels = self.load_dataset_json(json_path)
        if self._load_checkpoint():
            if self.hybrid_metadata_enabled and not self.hybrid_model_is_ready:
                self._log("Checkpoint cũ chưa có hybrid metadata head → hiệu chỉnh từ dataset.")
                self._fit_hybrid_metadata_model(texts, labels)
                self._save_metadata()
            return
        self._train(texts, labels)

    def fit_from_split(
        self,
        split: DataSplit,
        luu_split: bool = True,
        thu_muc_split: str = ".",
        ten_file_prefix: str = "transformer_split",
        min_auc_threshold: float = 0.60,
    ) -> dict[str, float]:
        """
        Train từ pre-split data với validation + early stopping.

        min_auc_threshold: Nếu checkpoint đã tồn tại nhưng AUC trên val set
        thấp hơn ngưỡng này (mặc định 0.60), checkpoint bị coi là COLLAPSED
        và sẽ bị retrain lại tự động thay vì dùng weights xấu.
        """
        checkpoint_ok = False
        if self._load_checkpoint():
            # Kiểm tra checkpoint có bị collapse không
            val_scores = self._predict_scores_batch(split.val_texts)
            try:
                from sklearn.metrics import roc_auc_score as _auc
                auc_check = _auc(split.val_labels, val_scores)
            except Exception:
                auc_check = 0.0

            if auc_check >= min_auc_threshold:
                self._log(
                    f"Checkpoint hợp lệ (val AUC={auc_check:.4f} ≥ {min_auc_threshold}) "
                    "→ bỏ qua fine-tune backbone."
                )
                checkpoint_ok = True
                if self.hybrid_metadata_enabled and not self.hybrid_model_is_ready:
                    self._log("Hiệu chỉnh hybrid metadata head từ split hiện tại.")
                    self._fit_hybrid_metadata_model(split.train_texts, split.train_labels)
                self.threshold, self.best_val_metrics = tim_nguong_toi_uu(
                    split.val_labels,
                    val_scores,
                    metric=self.threshold_metric,
                )
                self._save_metadata()
                return self._evaluate(split.test_texts, split.test_labels, "Test")
            else:
                self._log(
                    f"⚠ Checkpoint COLLAPSED (val AUC={auc_check:.4f} < {min_auc_threshold}) "
                    "→ XÓA checkpoint cũ và RETRAIN lại từ đầu."
                )
                # Xóa checkpoint bị hỏng
                for f in [
                    self.checkpoint_path,
                    self.metadata_path,
                    self.checkpoint_path.with_suffix(".best.pt"),
                ]:
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
                # Reload model weights gốc từ HuggingFace (reset về pretrained)
                self._log(f"Reload pretrained weights: {self.model_name} ...")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=2,
                    attn_implementation="eager",
                    use_safetensors=False,
                )
                self.model.to(self.device)
                # Áp dụng lại layer freezing
                if self.freeze_layers > 0:
                    self._freeze_backbone_layers(self.freeze_layers)
                self.hybrid_model_is_ready = False

        if luu_split:
            luu_split_ra_file(split, thu_muc=thu_muc_split, ten_file_prefix=ten_file_prefix)
        self._train_with_validation(
            train_texts=split.train_texts,
            train_labels=split.train_labels,
            val_texts=split.val_texts,
            val_labels=split.val_labels,
        )
        return self._evaluate(split.test_texts, split.test_labels, "Test")

    def _build_loss(self, labels: list[int]) -> "FocalLoss":
        n = len(labels)
        n_fraud = sum(labels)
        n_non = n - n_fraud
        alpha_fraud = n_non / n if n else 0.5
        alpha_non_fraud = n_fraud / n if n else 0.5
        alpha = torch.tensor([alpha_non_fraud, alpha_fraud], dtype=torch.float).to(self.device)
        self._log(
            f"FocalLoss — γ={self.gamma} | "
            f"α_non_fraud={alpha_non_fraud:.3f} α_fraud={alpha_fraud:.3f}"
        )
        return FocalLoss(alpha=alpha, gamma=self.gamma)

    def _predict_raw_scores_batch(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if any(len(text) > self.chunk_max_chars for text in texts):
            return [self._predict_raw_fraud_score(text) for text in texts]

        self.model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                out = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                )
                probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().tolist()
                scores.extend(float(p) for p in probs)
        return scores

    def _hybrid_sigmoid(self, value: float) -> float:
        value = max(min(value, 30.0), -30.0)
        return 1.0 / (1.0 + math.exp(-value))

    def _apply_hybrid_vector(self, vector: list[float]) -> float:
        if not self.hybrid_model_is_ready or not self.hybrid_coefficients:
            return float(vector[0])

        normalized: list[float] = []
        for value, mean, scale in zip(vector, self.hybrid_scaler_mean, self.hybrid_scaler_scale):
            safe_scale = scale if scale not in (0.0, None) else 1.0
            normalized.append((float(value) - float(mean)) / float(safe_scale))
        linear = self.hybrid_intercept
        for coef, value in zip(self.hybrid_coefficients, normalized):
            linear += float(coef) * float(value)
        return self._hybrid_sigmoid(linear)

    def _ap_dung_hybrid_metadata(
        self,
        text: str,
        raw_prob: float,
    ) -> tuple[float, dict[str, float]]:
        feature_map, vector = tao_hybrid_feature_vector(text, raw_prob)
        if not self.hybrid_metadata_enabled or not self.hybrid_model_is_ready:
            return float(raw_prob), feature_map
        return self._apply_hybrid_vector(vector), feature_map

    def _fit_hybrid_metadata_model(self, texts: list[str], labels: list[int]) -> None:
        if not self.hybrid_metadata_enabled or not texts:
            return
        if len(set(labels)) < 2:
            self._log("Bỏ qua hybrid metadata head vì dataset chỉ có 1 class.")
            return

        raw_scores = self._predict_raw_scores_batch(texts)
        rows: list[list[float]] = []
        for text, raw_score in zip(texts, raw_scores):
            _, vector = tao_hybrid_feature_vector(text, raw_score)
            rows.append(vector)

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(rows)
        hybrid_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        hybrid_model.fit(x_scaled, labels)

        self.hybrid_scaler_mean = [float(v) for v in scaler.mean_.tolist()]
        self.hybrid_scaler_scale = [
            float(v) if float(v) != 0.0 else 1.0
            for v in scaler.scale_.tolist()
        ]
        self.hybrid_coefficients = [float(v) for v in hybrid_model.coef_[0].tolist()]
        self.hybrid_intercept = float(hybrid_model.intercept_[0])
        self.hybrid_model_is_ready = True
        self._log(
            f"Đã fit hybrid metadata head "
            f"({len(self.hybrid_feature_names)} features, {len(texts)} mẫu)."
        )

    def _predict_scores_batch(self, texts: list[str]) -> list[float]:
        raw_scores = self._predict_raw_scores_batch(texts)
        if not self.hybrid_metadata_enabled or not self.hybrid_model_is_ready:
            return raw_scores
        final_scores: list[float] = []
        for text, raw_score in zip(texts, raw_scores):
            final_score, _ = self._ap_dung_hybrid_metadata(text, raw_score)
            final_scores.append(final_score)
        return final_scores

    def score_text(self, text: str, threshold: float | None = None) -> dict[str, object]:
        threshold_used = self.threshold if threshold is None else threshold
        _, raw_fraud_prob = self.predict_raw_proba(text)
        hybrid_fraud_prob, metadata_features = self._ap_dung_hybrid_metadata(
            text,
            raw_fraud_prob,
        )
        breakdown = tao_score_breakdown_tu_van_ban(
            text,
            raw_model_score=raw_fraud_prob,
            hybrid_metadata_score=hybrid_fraud_prob,
            threshold_used=threshold_used,
            hybrid_metadata_enabled=(
                self.hybrid_metadata_enabled and self.hybrid_model_is_ready
            ),
            bonus_moi_flag=self.rule_bonus_per_flag,
        )
        breakdown["metadata_features"] = lam_tron_metadata_features(metadata_features)
        return breakdown

    def score_texts_batch(
        self,
        texts: list[str],
        threshold: float | None = None,
    ) -> list[dict[str, object]]:
        threshold_used = self.threshold if threshold is None else threshold
        raw_scores = self._predict_raw_scores_batch(texts)
        outputs: list[dict[str, object]] = []
        for text, raw_score in zip(texts, raw_scores):
            hybrid_score, metadata_features = self._ap_dung_hybrid_metadata(text, raw_score)
            breakdown = tao_score_breakdown_tu_van_ban(
                text,
                raw_model_score=raw_score,
                hybrid_metadata_score=hybrid_score,
                threshold_used=threshold_used,
                hybrid_metadata_enabled=(
                    self.hybrid_metadata_enabled and self.hybrid_model_is_ready
                ),
                bonus_moi_flag=self.rule_bonus_per_flag,
            )
            breakdown["metadata_features"] = lam_tron_metadata_features(metadata_features)
            outputs.append(breakdown)
        return outputs

    def _save_metadata(self) -> None:
        payload = {
            "threshold": self.threshold,
            "threshold_metric": self.threshold_metric,
            "best_val_metrics": self.best_val_metrics,
            "gamma": self.gamma,
            "hybrid_metadata_enabled": self.hybrid_metadata_enabled,
            "hybrid_metadata_requested": self.hybrid_metadata_requested,
            "hybrid_feature_names": self.hybrid_feature_names,
            "hybrid_scaler_mean": self.hybrid_scaler_mean,
            "hybrid_scaler_scale": self.hybrid_scaler_scale,
            "hybrid_coefficients": self.hybrid_coefficients,
            "hybrid_intercept": self.hybrid_intercept,
            "hybrid_model_is_ready": self.hybrid_model_is_ready,
            "freeze_layers": self.freeze_layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "max_len": self.max_len,
            "model_name": self.model_name,
            "rule_bonus_per_flag": self.rule_bonus_per_flag,
        }
        self.metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            return
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.threshold = float(payload.get("threshold", 0.5))
            self.threshold_metric = str(payload.get("threshold_metric", self.threshold_metric))
            self.best_val_metrics = dict(payload.get("best_val_metrics", {}))
            self.gamma = float(payload.get("gamma", FOCAL_GAMMA_DEFAULT))
            checkpoint_hybrid_enabled = bool(
                payload.get("hybrid_metadata_enabled", self.hybrid_metadata_enabled)
            )
            self.hybrid_metadata_enabled = bool(
                self.hybrid_metadata_requested and checkpoint_hybrid_enabled
            )
            self.hybrid_feature_names = list(
                payload.get("hybrid_feature_names", self.hybrid_feature_names)
            )
            self.hybrid_scaler_mean = [
                float(value) for value in payload.get("hybrid_scaler_mean", [])
            ]
            self.hybrid_scaler_scale = [
                float(value) if float(value) != 0.0 else 1.0
                for value in payload.get("hybrid_scaler_scale", [])
            ]
            self.hybrid_coefficients = [
                float(value) for value in payload.get("hybrid_coefficients", [])
            ]
            self.hybrid_intercept = float(payload.get("hybrid_intercept", 0.0))
            self.hybrid_model_is_ready = bool(
                payload.get(
                    "hybrid_model_is_ready",
                    bool(self.hybrid_coefficients and self.hybrid_scaler_mean),
                )
            )
            self.rule_bonus_per_flag = float(
                payload.get("rule_bonus_per_flag", self.rule_bonus_per_flag)
            )
        except Exception as exc:
            self._log(f"Không load được metadata: {exc}")

    def _train(self, texts: list[str], labels: list[int]) -> None:
        """
        Train không có validation set.
        Nếu dataset đủ lớn (>= 10 mẫu), tự động tách 10% val để early stopping.
        Luôn dùng LinearScheduler với warm-up để tránh gradient explosion.
        """
        # Tự động tách val set 10% để early stopping nếu đủ dữ liệu
        n = len(texts)
        if n >= 10:
            n_val = max(1, int(n * 0.1))
            val_texts  = texts[-n_val:]
            val_labels = labels[-n_val:]
            train_texts  = texts[:-n_val]
            train_labels = labels[:-n_val]
            self._log(f"Tự tách val: train={len(train_texts)} | val={len(val_texts)}")
            self._train_with_validation(train_texts, train_labels, val_texts, val_labels)
            return

        # Fallback: dataset rất nhỏ — train thuần không val
        dataset = _TransformerFraudDataset(texts, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_tao_transformer_collate_fn(self.tokenizer, self.max_len),
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        total_steps   = len(loader) * self.epochs
        warmup_steps  = max(1, int(total_steps * 0.1))
        try:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        except ImportError:
            scheduler = None

        criterion = self._build_loss(labels)
        self._log(
            f"Bắt đầu fine-tune (no-val): {len(texts)} mẫu, "
            f"{self.epochs} epochs | warmup={warmup_steps} steps"
        )
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item()
            avg = total_loss / max(len(loader), 1)
            self._log(f"Epoch [{epoch + 1}/{self.epochs}] — Loss: {avg:.4f}")
        self.is_trained = True
        self._fit_hybrid_metadata_model(texts, labels)
        self.model_source = "runtime_fit:training_data"
        torch.save(self.model.state_dict(), self.checkpoint_path)
        self._save_metadata()
        self._log(f"Đã lưu checkpoint → {self.checkpoint_path}")

    def _train_with_validation(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
    ) -> None:
        train_dataset = _TransformerFraudDataset(train_texts, train_labels)
        val_dataset = _TransformerFraudDataset(val_texts, val_labels)
        collate_fn = _tao_transformer_collate_fn(self.tokenizer, self.max_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        # ── Differential Learning Rate ─────────────────────────────────────
        # Backbone (frozen layers sẽ tự skip do requires_grad=False)
        # Classifier head dùng LR cao hơn 10x để hội tụ nhanh hơn
        backbone_params = [
            p for n, p in self.model.named_parameters()
            if "classifier" not in n and p.requires_grad
        ]
        head_params = [
            p for n, p in self.model.named_parameters()
            if "classifier" in n and p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.learning_rate},
                {"params": head_params,    "lr": self.learning_rate * 10},
            ],
            weight_decay=0.01,
        )
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self.model.parameters())
        self._log(
            f"Trainable params: {n_trainable:,}/{n_total:,} "
            f"({n_trainable/max(n_total,1)*100:.1f}%) "
            f"| backbone LR={self.learning_rate} | head LR={self.learning_rate*10}"
        )

        # ── Linear Warm-up + Linear Decay scheduler ────────────────────────
        total_steps  = len(train_loader) * self.epochs
        warmup_steps = max(1, int(total_steps * 0.06))  # 6% warm-up (nhẹ hơn)
        try:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            self._log(
                f"Scheduler: LinearWarmup | total={total_steps} steps "
                f"| warmup={warmup_steps} steps"
            )
        except ImportError:
            scheduler = None
            self._log("⚠ transformers scheduler không khả dụng.")
        criterion = self._build_loss(train_labels)

        best_checkpoint = self.checkpoint_path.with_suffix(".best.pt")
        best_metric = -1.0
        best_auprc = -1.0
        best_threshold = 0.5
        no_improve = 0

        self._log(
            "Train với validation:\n"
            f"  Train: {len(train_texts)} | Val: {len(val_texts)} | Epochs: {self.epochs}"
        )
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'F2':>8} {'AUPRC':>8} {'Thr':>6} {'Best':>6}")
        print("-" * 70)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()   # cập nhật LR sau mỗi step (không phải epoch)
                train_loss += loss.item()
            avg_train = train_loss / max(len(train_loader), 1)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    batch_labels = batch["labels"].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    val_loss += criterion(outputs.logits, batch_labels).item()
            avg_val = val_loss / max(len(val_loader), 1)
            val_scores = self._predict_scores_batch(val_texts)
            threshold, val_metrics = tim_nguong_toi_uu(
                val_labels,
                val_scores,
                metric=self.threshold_metric,
            )
            metric_val = val_metrics.get(self.threshold_metric, 0.0)
            current_auprc = val_metrics.get("auprc", 0.0)

            is_best = (
                metric_val > best_metric + 1e-8
                or (
                    abs(metric_val - best_metric) <= 1e-8
                    and current_auprc > best_auprc + 1e-8
                )
            )
            if is_best:
                best_metric = metric_val
                best_auprc = current_auprc
                best_threshold = threshold
                self.best_val_metrics = val_metrics
                torch.save(self.model.state_dict(), best_checkpoint)
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"{epoch + 1:>6} {avg_train:>12.4f} {avg_val:>10.4f} "
                f"{val_metrics['f2']:>8.4f} {val_metrics['auprc']:>8.4f} {threshold:>6.3f} "
                f"{'✓' if is_best else '':>6}"
            )
            if no_improve >= self.patience:
                self._log(f"Early stopping sau {self.patience} epoch không cải thiện.")
                break

        if best_checkpoint.exists():
            state = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(state)
            self._fit_hybrid_metadata_model(train_texts, train_labels)
            final_val_scores = self._predict_scores_batch(val_texts)
            self.threshold, self.best_val_metrics = tim_nguong_toi_uu(
                val_labels,
                final_val_scores,
                metric=self.threshold_metric,
            )
            torch.save(self.model.state_dict(), self.checkpoint_path)
            self._save_metadata()
            self._log(
                f"Best checkpoint backbone: {self.threshold_metric}={best_metric:.4f} "
                f"auprc={best_auprc:.4f} threshold(raw)={best_threshold:.3f}"
            )
            self._log(
                f"Hybrid validation threshold={self.threshold:.3f} "
                f"{self.threshold_metric}={self.best_val_metrics.get(self.threshold_metric, 0.0):.4f}"
            )
        self.is_trained = True
        self.model_source = "runtime_fit:training_data"

    def _evaluate(
        self,
        texts: list[str],
        labels: list[int],
        split_name: str = "Eval",
    ) -> dict[str, float]:
        y_scores = self._predict_scores_batch(texts)
        metrics = danh_gia_du_doan(labels, y_scores, threshold=self.threshold)
        metrics["threshold"] = self.threshold
        self._log(f"Kết quả {split_name} set ({len(texts)} mẫu):")
        for key, val in metrics.items():
            print(f"  {key:>10}: {val:.4f}")
        return metrics

    def _load_checkpoint(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        try:
            state = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.is_trained = True
            self.model_source = f"checkpoint:{self.checkpoint_path.name}"
            self._load_metadata()
            self._log(f"Đã load checkpoint '{self.checkpoint_path}'")
            return True
        except Exception as exc:
            self._log(f"Không load được checkpoint: {exc}")
            return False

    def _predict_raw_proba_single(self, text: str) -> tuple[float, float]:
        self.model.eval()
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().tolist()
        return float(probs[0]), float(probs[1])

    def predict_raw_proba(self, text: str) -> tuple[float, float]:
        # Bỏ kiểm tra is_trained ở đây do khi validation ở epoch đầu, 
        # model đang huấn luyện dở dang nhưng vẫn cần predict để tính metric.

        if len(text) <= self.chunk_max_chars:
            return self._predict_raw_proba_single(text)

        chunks = tach_chunks_recursive(
            text,
            max_chars=self.chunk_max_chars,
            overlap=CHUNK_OVERLAP,
        )
        if not chunks:
            return self._predict_raw_proba_single(text)

        fraud_scores: list[float] = []
        chunk_lens: list[int] = []
        for chunk in chunks:
            _, fraud_prob = self._predict_raw_proba_single(chunk)
            fraud_scores.append(fraud_prob)
            chunk_lens.append(len(chunk))

        max_score = max(fraud_scores)
        total_len = sum(chunk_lens) or 1
        weighted_mean = sum(score * length for score, length in zip(fraud_scores, chunk_lens)) / total_len
        final_fraud = 0.6 * max_score + 0.4 * weighted_mean

        self._log(
            f"Chunking: {len(chunks)} chunks | "
            f"max={max_score:.3f} w_mean={weighted_mean:.3f} final={final_fraud:.3f}"
        )
        return 1.0 - final_fraud, final_fraud

    def _predict_raw_fraud_score(self, text: str) -> float:
        return float(self.predict_raw_proba(text)[1])

    def predict_proba(self, text: str) -> tuple[float, float]:
        raw_non_fraud_prob, raw_fraud_prob = self.predict_raw_proba(text)
        final_fraud_prob, _ = self._ap_dung_hybrid_metadata(text, raw_fraud_prob)
        return 1.0 - final_fraud_prob, final_fraud_prob

    def _explain_top_terms(self, text: str, top_k: int = 5) -> list[str]:
        if not self.is_trained:
            return []
        if not IG_AVAILABLE:
            self._log("Captum chưa cài — bỏ qua IG explain. pip install captum")
            return []

        explain_text = text
        if len(text) > self.chunk_max_chars:
            chunks = tach_chunks_recursive(text, max_chars=self.chunk_max_chars, overlap=0)
            if chunks:
                scores = self._predict_raw_scores_batch(chunks)
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                explain_text = chunks[best_idx]
                self._log(
                    f"Text dài → explain chunk {best_idx + 1}/{len(chunks)} "
                    f"(fraud_score={scores[best_idx]:.3f})"
                )

        try:
            return self.ig_explainer.explain_top_terms(
                text=explain_text,
                top_k=top_k,
                max_len=self.max_len,
            )
        except Exception as exc:
            self._log(f"Không thể tạo IG explain: {exc}")
            return []

    def explain_all_methods(self, text: str, top_k: int = 5) -> dict[str, object]:
        perturbation = PerturbationTextExplainer(self._predict_raw_scores_batch)
        ig_terms = self._explain_top_terms(text, top_k=top_k)
        shap_terms = perturbation.explain_shap_approx(text, top_k=top_k)
        lime_terms = perturbation.explain_lime_surrogate(text, top_k=top_k)
        return {
            "available_methods": [
                "integrated_gradients",
                "shap_token_approx",
                "lime_surrogate_approx",
                "bias_group_report",
            ],
            "integrated_gradients": ig_terms,
            "shap_token_approx": shap_terms,
            "lime_surrogate_approx": lime_terms,
        }

    def predict(self, text: str, threshold: float | None = None) -> PredictionResult:
        raw_non_fraud_prob, raw_fraud_prob = self.predict_raw_proba(text)
        non_fraud_prob, fraud_prob = self.predict_proba(text)
        threshold_used = self.threshold if threshold is None else threshold
        breakdown = tao_score_breakdown(
            raw_model_score=raw_fraud_prob,
            hybrid_metadata_score=fraud_prob,
            red_flags=phat_hien_co_do(text),
            threshold_used=threshold_used,
            hybrid_metadata_enabled=(
                self.hybrid_metadata_enabled and self.hybrid_model_is_ready
            ),
            bonus_moi_flag=self.rule_bonus_per_flag,
        )
        red_flags = list(breakdown["red_flags"])
        adjusted_score = float(breakdown["final_score"])
        label = int(breakdown["predicted_label"])
        tokens_sach, van_ban_embedding = tien_xu_ly_day_du(text)
        van_ban_sach = tao_van_ban_hien_thi(text)
        doan_nghi_ngo = tim_doan_nghi_ngo(text)
        chat_luong = tinh_chat_luong_van_ban(text, tokens_sach)
        muc_do_rui_ro = xep_muc_do_rui_ro(adjusted_score)
        final_fraud_from_hybrid, metadata_features = self._ap_dung_hybrid_metadata(
            text,
            raw_fraud_prob,
        )
        bias_flags = tao_bias_flags_tu_features(metadata_features)
        explainability = self.explain_all_methods(text)
        ig_terms = list(explainability.get("integrated_gradients", []))
        shap_terms = list(explainability.get("shap_token_approx", []))
        lime_terms = list(explainability.get("lime_surrogate_approx", []))
        if ig_terms:
            top_terms = ig_terms
            self.top_terms_method = "integrated_gradients"
        elif shap_terms:
            top_terms = shap_terms
            self.top_terms_method = "shap_token_approx"
        else:
            top_terms = lime_terms
            self.top_terms_method = "lime_surrogate_approx" if lime_terms else "integrated_gradients"
        explainability["selected_top_terms_method"] = self.top_terms_method
        explanation: list[str] = []
        if red_flags:
            explanation.append(f"Các red flags theo rule-based: {', '.join(red_flags)}")
        explanation.append(f"Xác suất gian lận raw backbone: {raw_fraud_prob:.3f}")
        explanation.append(f"Xác suất sau hybrid metadata: {final_fraud_from_hybrid:.3f}")
        explanation.append(f"Điểm rủi ro sau rule-based: {adjusted_score:.3f}")
        explanation.append(f"Chất lượng văn bản/OCR: {chat_luong:.3f}")
        if top_terms:
            explanation.append("Các token/từ tác động mạnh nhất: " + ", ".join(top_terms))
        if bias_flags:
            explanation.append(f"Các bias/data flags cần lưu ý: {', '.join(bias_flags)}")
        return PredictionResult(
            label=label,
            muc_do_rui_ro=muc_do_rui_ro,
            model_probability_fraud=fraud_prob,
            model_probability_non_fraud=non_fraud_prob,
            probability_fraud=adjusted_score,
            probability_non_fraud=1 - adjusted_score,
            threshold_used=threshold_used,
            chat_luong_van_ban=chat_luong,
            red_flags=red_flags,
            explanation=explanation,
            top_terms=top_terms,
            van_ban_sach=van_ban_sach,
            van_ban_embedding=van_ban_embedding,
            doan_nghi_ngo=doan_nghi_ngo,
            metadata_features=lam_tron_metadata_features(metadata_features),
            raw_text_probability_fraud=raw_fraud_prob,
            raw_text_probability_non_fraud=raw_non_fraud_prob,
            hybrid_probability_fraud=final_fraud_from_hybrid,
            hybrid_probability_non_fraud=1 - final_fraud_from_hybrid,
            risk_label=label,
            risk_signals=red_flags,
            score_breakdown=breakdown,
            bias_flags=bias_flags,
            explainability=explainability,
        )
