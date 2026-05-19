from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from engine_bias import tao_bias_flags_tu_features
from engine_common import (
    BASELINE_CHECKPOINT,
    DataSplit,
    PredictionResult,
    RULE_RED_FLAG_BONUS,
    danh_gia_du_doan,
    luu_split_ra_file,
    phat_hien_co_do,
    tai_du_lieu_json,
    tao_score_breakdown,
    tao_score_breakdown_tu_van_ban,
    tien_xu_ly_day_du,
    tao_van_ban_hien_thi,
    tim_doan_nghi_ngo,
    tim_nguong_toi_uu,
    tinh_chat_luong_van_ban,
    xep_muc_do_rui_ro,
)
from engine_metadata import lam_tron_metadata_features, trich_xuat_metadata_features


class BoTFIDF:
    """Wrapper nhe quanh sklearn TfidfVectorizer de train nhanh cho demo/API."""

    def __init__(
        self,
        *,
        max_features: int = 20000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.feature_names: list[str] = []

    @staticmethod
    def _chuan_bi_text(text: str) -> str:
        tokens, _ = tien_xu_ly_day_du(text)
        return " ".join(tokens)

    def fit(self, texts: Iterable[str]) -> None:
        processed = [self._chuan_bi_text(text) for text in texts]
        self.vectorizer.fit(processed)
        self.feature_names = list(self.vectorizer.get_feature_names_out())

    def transform(self, texts: Iterable[str]):
        processed = [self._chuan_bi_text(text) for text in texts]
        return self.vectorizer.transform(processed)

    def fit_transform(self, texts: Iterable[str]):
        processed = [self._chuan_bi_text(text) for text in texts]
        matrix = self.vectorizer.fit_transform(processed)
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        return matrix

    def explain_top_terms(
        self,
        text: str,
        weights: list[float],
        top_k: int = 5,
    ) -> list[str]:
        if not self.feature_names or not weights:
            return []

        row = self.transform([text])
        if row.shape[0] == 0:
            return []

        scored: list[tuple[float, str]] = []
        for idx, value in zip(row.indices, row.data):
            if idx < len(weights):
                scored.append((float(value) * weights[idx], self.feature_names[idx]))
        scored.sort(key=lambda item: abs(item[0]), reverse=True)
        return [term for _, term in scored[:top_k]]


class MoHinhGianLan:
    """Model 1 — TF-IDF + Logistic Regression (baseline nhanh cho CLI/API)."""

    def __init__(
        self,
        learning_rate: float = 0.25,
        epochs: int = 300,
        checkpoint_path: str = BASELINE_CHECKPOINT,
        max_features: int = 20000,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_features = max_features
        self.checkpoint_path = Path(checkpoint_path)
        self.vectorizer = BoTFIDF(max_features=max_features)
        self.classifier: LogisticRegression | None = None
        self.weights: list[float] = []
        self.bias = 0.0
        self.is_trained = False
        self.threshold = 0.5
        self.model_source = "untrained"
        self.top_terms_method = "tfidf_logreg_contribution"
        self.best_val_metrics: dict[str, float] = {}
        self.threshold_metric = "f2"
        self.rule_bonus_per_flag = RULE_RED_FLAG_BONUS

    def _tao_classifier(self) -> LogisticRegression:
        return LogisticRegression(
            max_iter=max(200, self.epochs),
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        text_list = list(texts)
        y = list(labels)
        X = self.vectorizer.fit_transform(text_list)
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise RuntimeError("Không có đặc trưng TF-IDF hợp lệ để huấn luyện.")

        self.classifier = self._tao_classifier()
        self.classifier.fit(X, y)
        self.weights = self.classifier.coef_[0].astype(float).tolist()
        self.bias = float(self.classifier.intercept_[0])
        self.is_trained = True
        self.model_source = "runtime_fit:training_data"
        self._save_checkpoint()

    @staticmethod
    def load_dataset_json(json_path: str) -> tuple[list[str], list[int]]:
        texts, labels = tai_du_lieu_json(json_path)
        n_fraud = sum(labels)
        print(
            f"[Baseline] Dataset: {len(texts)} mau "
            f"({n_fraud} fraud / {len(labels) - n_fraud} non-fraud)"
        )
        return texts, labels

    def fit_from_json(self, json_path: str) -> None:
        texts, labels = self.load_dataset_json(json_path)
        self.fit(texts, labels)
        self.model_source = f"runtime_fit:{json_path}"
        self._save_checkpoint()

    def _bat_buoc_da_huan_luyen(self) -> LogisticRegression:
        if not self.is_trained or self.classifier is None:
            raise RuntimeError("Model chưa được huấn luyện.")
        return self.classifier

    def _predict_scores_batch(self, texts: list[str]) -> list[float]:
        classifier = self._bat_buoc_da_huan_luyen()
        if not texts:
            return []
        features_batch = self.vectorizer.transform(texts)
        return classifier.predict_proba(features_batch)[:, 1].astype(float).tolist()

    def score_text(self, text: str, threshold: float | None = None) -> dict[str, object]:
        threshold_used = self.threshold if threshold is None else threshold
        _, fraud_prob = self.predict_proba(text)
        breakdown = tao_score_breakdown_tu_van_ban(
            text,
            raw_model_score=fraud_prob,
            hybrid_metadata_score=fraud_prob,
            threshold_used=threshold_used,
            hybrid_metadata_enabled=False,
            bonus_moi_flag=self.rule_bonus_per_flag,
        )
        metadata_features = trich_xuat_metadata_features(text)
        breakdown["metadata_features"] = lam_tron_metadata_features(metadata_features)
        return breakdown

    def score_texts_batch(
        self,
        texts: list[str],
        threshold: float | None = None,
    ) -> list[dict[str, object]]:
        threshold_used = self.threshold if threshold is None else threshold
        raw_scores = self._predict_scores_batch(texts)
        outputs: list[dict[str, object]] = []
        for text, raw_score in zip(texts, raw_scores):
            metadata_features = trich_xuat_metadata_features(text)
            breakdown = tao_score_breakdown_tu_van_ban(
                text,
                raw_model_score=raw_score,
                hybrid_metadata_score=raw_score,
                threshold_used=threshold_used,
                hybrid_metadata_enabled=False,
                bonus_moi_flag=self.rule_bonus_per_flag,
            )
            breakdown["metadata_features"] = lam_tron_metadata_features(metadata_features)
            outputs.append(breakdown)
        return outputs

    def fit_from_split(
        self,
        split: DataSplit,
        luu_split: bool = True,
        thu_muc_split: str = ".",
        ten_file_prefix: str = "baseline_split",
    ) -> dict[str, float]:
        if luu_split:
            luu_split_ra_file(split, thu_muc=thu_muc_split, ten_file_prefix=ten_file_prefix)

        self.fit(split.train_texts, split.train_labels)
        self.model_source = "runtime_fit:split_train"

        val_scores = self._predict_scores_batch(split.val_texts)
        self.threshold, self.best_val_metrics = tim_nguong_toi_uu(
            split.val_labels,
            val_scores,
            metric=self.threshold_metric,
        )
        self._save_checkpoint()
        print(
            f"[Baseline] Validation best {self.threshold_metric}="
            f"{self.best_val_metrics.get(self.threshold_metric, 0.0):.4f} "
            f"threshold={self.threshold:.3f}"
        )
        return self._evaluate(split.test_texts, split.test_labels, "Test")

    def _evaluate(
        self,
        texts: list[str],
        labels: list[int],
        split_name: str = "Eval",
    ) -> dict[str, float]:
        y_scores = self._predict_scores_batch(texts)
        metrics = danh_gia_du_doan(labels, y_scores, threshold=self.threshold)
        metrics["threshold"] = self.threshold
        print(f"\n[Baseline] Ket qua {split_name} set ({len(texts)} mau):")
        for key, val in metrics.items():
            print(f"  {key:>10}: {val:.4f}")
        return metrics

    def predict_proba(self, text: str) -> tuple[float, float]:
        classifier = self._bat_buoc_da_huan_luyen()
        features = self.vectorizer.transform([text])
        fraud_prob = float(classifier.predict_proba(features)[0][1])
        return 1.0 - fraud_prob, fraud_prob

    def predict(self, text: str, threshold: float | None = None) -> PredictionResult:
        non_fraud_prob, fraud_prob = self.predict_proba(text)
        threshold_used = self.threshold if threshold is None else threshold
        breakdown = tao_score_breakdown(
            raw_model_score=fraud_prob,
            hybrid_metadata_score=fraud_prob,
            red_flags=phat_hien_co_do(text),
            threshold_used=threshold_used,
            hybrid_metadata_enabled=False,
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
        top_terms = self.vectorizer.explain_top_terms(text, self.weights)
        metadata_features = trich_xuat_metadata_features(text)
        bias_flags = tao_bias_flags_tu_features(metadata_features)

        explanation: list[str] = []
        if red_flags:
            explanation.append(f"Các red flags theo rule-based: {', '.join(red_flags)}")
        explanation.append(f"Xác suất gian lận của mô hình: {fraud_prob:.3f}")
        explanation.append(f"Điểm rủi ro sau rule-based: {adjusted_score:.3f}")
        explanation.append(f"Chất lượng văn bản/OCR: {chat_luong:.3f}")
        if top_terms:
            explanation.append(f"Từ khóa tác động mạnh: {', '.join(top_terms)}")

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
            raw_text_probability_fraud=fraud_prob,
            raw_text_probability_non_fraud=non_fraud_prob,
            hybrid_probability_fraud=fraud_prob,
            hybrid_probability_non_fraud=non_fraud_prob,
            risk_label=label,
            risk_signals=red_flags,
            score_breakdown=breakdown,
            bias_flags=bias_flags,
            explainability={
                "available_methods": ["tfidf_logreg_contribution"],
                "top_terms": top_terms,
            },
        )

    def _save_checkpoint(self) -> None:
        payload = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "max_features": self.max_features,
            "threshold": self.threshold,
            "model_source": self.model_source,
            "top_terms_method": self.top_terms_method,
            "best_val_metrics": self.best_val_metrics,
            "threshold_metric": self.threshold_metric,
            "rule_bonus_per_flag": self.rule_bonus_per_flag,
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "weights": self.weights,
            "bias": self.bias,
        }
        self.checkpoint_path.write_bytes(pickle.dumps(payload))

    def _load_checkpoint(self) -> bool:
        if not self.checkpoint_path.exists():
            return False

        payload = pickle.loads(self.checkpoint_path.read_bytes())
        self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
        self.epochs = int(payload.get("epochs", self.epochs))
        self.max_features = int(payload.get("max_features", self.max_features))
        self.threshold = float(payload.get("threshold", self.threshold))
        self.model_source = f"checkpoint:{self.checkpoint_path.name}"
        self.top_terms_method = str(payload.get("top_terms_method", self.top_terms_method))
        self.best_val_metrics = {
            str(key): float(value)
            for key, value in payload.get("best_val_metrics", {}).items()
        }
        self.threshold_metric = str(
            payload.get("threshold_metric", self.threshold_metric)
        )
        self.rule_bonus_per_flag = float(
            payload.get("rule_bonus_per_flag", self.rule_bonus_per_flag)
        )
        self.vectorizer = payload.get("vectorizer", BoTFIDF(max_features=self.max_features))
        self.classifier = payload.get("classifier")
        self.weights = [float(item) for item in payload.get("weights", [])]
        self.bias = float(payload.get("bias", 0.0))
        self.is_trained = self.classifier is not None
        return self.is_trained
