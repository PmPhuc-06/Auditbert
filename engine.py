"""
engine.py — facade re-export.

app.py và main.py import từ đây như cũ, không cần sửa gì.
Nội bộ đã tách thành 3 file:
  - engine_common.py   → preprocessing, red flags, utils, dataclass, metrics
  - engine_baseline.py → TF-IDF + Logistic Regression
  - engine_phobert.py  → PhoBERT fine-tune
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAU_TACH_TU = re.compile(r"\w+", re.UNICODE)
MAU_HTML = re.compile(r"<[^>]+>")
MAU_KHOANG_TRANG = re.compile(r"\s+")

try:
    from underthesea import word_tokenize as tach_tu_underthesea
except Exception:  # pragma: no cover - optional dependency
    tach_tu_underthesea = None

try:
    from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional dependency
    GoogleTranslator = None

# PhoBERT dependencies — chỉ cần cho Model 2
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    PHOBERT_AVAILABLE = True
except Exception:  # pragma: no cover
    PHOBERT_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_SPLITTER_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGCHAIN_SPLITTER_AVAILABLE = False

from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Hằng số chung
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "và", "la", "là", "của", "cua", "các", "cac", "những", "nhung",
    "được", "duoc", "cho", "với", "voi", "một", "mot", "có", "co",
    "trong", "khi", "đã", "da", "đang", "dang", "từ", "tu",
    "the", "and", "or", "of", "to", "is", "are",
}

TU_KHOA_TIENG_ANH = {
    "the", "and", "of", "to", "is", "are", "with", "from",
    "revenue", "invoice", "liabilities", "audit", "transactions", "profit",
}

CU_M_TU_GHEP = [
    "báo cáo tài chính", "bao cao tai chinh",
    "bên liên quan", "ben lien quan",
    "doanh thu khống", "doanh thu tăng",
    "hóa đơn giả", "hoa don gia",
    "dòng tiền", "dong tien",
    "kiểm toán viên", "kiem toan vien",
    "chính sách kế toán", "chinh sach ke toan",
    "ngoài bảng cân đối", "ngoai bang can doi",
]

# Hằng số riêng cho Model 2 — PhoBERT
PHOBERT_MODEL_NAME      = "vinai/phobert-base"
PHOBERT_CHECKPOINT      = "phobert_fraud_checkpoint.pt"
PHOBERT_MAX_LEN         = 256
PHOBERT_BATCH_SIZE      = 8
PHOBERT_EPOCHS          = 5
PHOBERT_LR              = 2e-5
PHOBERT_PATIENCE        = 2
PHOBERT_THRESHOLD_METRIC = "f2"

# Recursive chunking — ký tự tối đa mỗi chunk (≈ PHOBERT_MAX_LEN tokens)
CHUNK_MAX_CHARS  = PHOBERT_MAX_LEN * 3   # ~768 ký tự
CHUNK_OVERLAP    = 120                    # overlap giữa 2 chunk liên tiếp


# ===========================================================================
# BƯỚC 2 — PREPROCESSING PIPELINE  (giữ nguyên)
# ===========================================================================

def xu_ly_van_ban(text: str) -> list[str]:
    return MAU_TACH_TU.findall(text.lower())


def bo_dau(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def chuan_hoa_text(text: str) -> str:
    lowered = text.lower()
    lowered = MAU_KHOANG_TRANG.sub(" ", lowered).strip()
    return lowered


def chuan_hoa_khong_dau(text: str) -> str:
    return chuan_hoa_text(bo_dau(text))


def co_cum_tu(texts: list[str], phrases: list[str]) -> bool:
    for phrase in phrases:
        normalized_phrase       = chuan_hoa_text(phrase)
        normalized_ascii_phrase = chuan_hoa_khong_dau(phrase)
        if any(
            normalized_phrase in text_variant or normalized_ascii_phrase in text_variant
            for text_variant in texts
        ):
            return True
    return False


def lam_sach_van_ban(text: str) -> str:
    text = MAU_HTML.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ỹà-ỹ_\s]", " ", text)
    text = chuan_hoa_text(text)
    return text


def tach_tu_tieng_viet(text: str) -> list[str]:
    if not text:
        return []
    if tach_tu_underthesea is not None:
        try:
            tokenized = tach_tu_underthesea(text, format="text")
            return tokenized.split()
        except Exception:
            pass
    text_thay_the = f" {text} "
    for cum_tu in CU_M_TU_GHEP:
        mau = re.escape(cum_tu)
        text_thay_the = re.sub(mau, cum_tu.replace(" ", "_"), text_thay_the)
    return xu_ly_van_ban(text_thay_the)


def bo_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]


def tien_xu_ly_day_du(text: str) -> tuple[list[str], str]:
    van_ban_sach = lam_sach_van_ban(text)
    tokens = tach_tu_tieng_viet(van_ban_sach)
    tokens = bo_stopwords(tokens)
    van_ban_san_sang = " ".join(tokens)
    return tokens, van_ban_san_sang


def tao_van_ban_hien_thi(text: str) -> str:
    van_ban = lam_sach_van_ban(text)
    return dich_hien_thi_sang_viet(van_ban)


def la_van_ban_tieng_anh(text: str) -> bool:
    tokens = xu_ly_van_ban(text)
    if not tokens:
        return False
    ascii_ratio  = sum(1 for ch in text if ord(ch) < 128) / max(len(text), 1)
    english_hits = sum(1 for token in tokens if token.lower() in TU_KHOA_TIENG_ANH)
    return ascii_ratio > 0.9 and english_hits >= 3


def dich_hien_thi_sang_viet(text: str) -> str:
    if not text:
        return text
    if not la_van_ban_tieng_anh(text):
        return text
    if GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source="auto", target="vi").translate(text)
    except Exception:
        return text


# ===========================================================================
# RED FLAGS — rule-based  (giữ nguyên)
# ===========================================================================

def phat_hien_co_do(text: str) -> list[str]:
    lowered       = chuan_hoa_text(text)
    lowered_ascii = chuan_hoa_khong_dau(text)
    text_variants = [lowered, lowered_ascii]
    flags: list[str] = []

    revenue_growth_phrases = [
        "doanh thu tăng", "doanh thu tăng mạnh", "doanh thu tăng đột biến",
    ]
    weak_cashflow_phrases = [
        "dòng tiền âm", "dòng tiền từ hoạt động kinh doanh âm",
        "dòng tiền hoạt động âm", "tiền từ hoạt động kinh doanh âm",
    ]
    if co_cum_tu(text_variants, revenue_growth_phrases) and co_cum_tu(text_variants, weak_cashflow_phrases):
        flags.append("doanh_thu_khong_di_kem_dong_tien")

    checks = {
        "doanh_thu_tang_nhung_loi_nhuan_giam": [
            "doanh thu tăng nhưng lợi nhuận gộp giảm",
            "doanh thu tăng nhưng lợi nhuận giảm",
            "revenue increased but gross profit decreased",
        ],
        "hoa_don_gia_hoac_thoi_phong_doanh_thu": [
            "hóa đơn giả", "hoa don gia",
            "thổi phồng doanh thu", "ghi nhận doanh thu khống", "doanh thu khống",
        ],
        "che_giau_no_phai_tra": [
            "che giấu nợ phải trả", "nợ ngoài bảng cân đối",
            "ngoài bảng cân đối", "off-balance",
        ],
        "dau_hieu_quan_tri_rui_ro": [
            "bên liên quan", "ben lien quan",
            "thay đổi kiểm toán viên", "thay doi kiem toan vien",
            "thay đổi chính sách kế toán", "thay doi chinh sach ke toan",
        ],
    }
    for rule_id, phrases in checks.items():
        if co_cum_tu(text_variants, phrases):
            flags.append(rule_id)
    return flags


def tach_doan(text: str) -> list[str]:
    parts = re.split(r"(?:\n{2,}|[\n\r]+|(?<=[\.\?\!])\s+)", text)
    return [part.strip() for part in parts if part and part.strip()]


def tim_doan_nghi_ngo(text: str) -> list[dict[str, str]]:
    ket_qua: list[dict[str, str]] = []
    for doan in tach_doan(text):
        danh_sach_co_do = phat_hien_co_do(doan)
        if danh_sach_co_do:
            ket_qua.append({"snippet": doan[:300], "flags": ", ".join(danh_sach_co_do)})
    return ket_qua


# ===========================================================================
# RECURSIVE CHUNKING — dùng langchain-text-splitters
# ===========================================================================

def tach_chunks_recursive(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Tách text thành các chunk ngữ nghĩa bằng RecursiveCharacterTextSplitter.

    Ưu tiên ranh giới: đoạn văn (\\n\\n) → dòng (\\n) → câu (. ) → từ ( ) → ký tự.
    Overlap giữa chunk liên tiếp để không mất ngữ cảnh biên.

    Fallback về cắt cứng nếu langchain-text-splitters chưa được cài.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    if LANGCHAIN_SPLITTER_AVAILABLE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)
        return [c for c in chunks if c.strip()]

    # Fallback: cắt cứng có overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


# ===========================================================================
# DATACLASS — dùng chung cho cả 2 model  (giữ nguyên)
# ===========================================================================

@dataclass
class PredictionResult:
    label: int
    muc_do_rui_ro: str
    model_probability_fraud: float
    model_probability_non_fraud: float
    probability_fraud: float
    probability_non_fraud: float
    threshold_used: float
    chat_luong_van_ban: float
    red_flags: list[str]
    explanation: list[str]
    top_terms: list[str]
    van_ban_sach: str
    van_ban_embedding: str
    doan_nghi_ngo: list[dict[str, str]]


# ===========================================================================
# DATA SPLIT UTILITIES  (giữ nguyên)
# ===========================================================================

@dataclass
class DataSplit:
    """Kết quả sau khi split dataset thành train / val / test."""
    train_texts:  list[str]
    train_labels: list[int]
    val_texts:    list[str]
    val_labels:   list[int]
    test_texts:   list[str]
    test_labels:  list[int]

    def summary(self) -> dict:
        def _stats(labels: list[int]) -> dict:
            n = len(labels)
            f = sum(labels)
            return {"total": n, "fraud": f, "non_fraud": n - f,
                    "fraud_ratio": round(f / n, 4) if n else 0.0}
        return {
            "train": _stats(self.train_labels),
            "val":   _stats(self.val_labels),
            "test":  _stats(self.test_labels),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("=" * 55)
        print(f"{'Split':<8} {'Total':>7} {'Fraud':>7} {'Non-F':>7} {'%Fraud':>8}")
        print("-" * 55)
        for split_name in ("train", "val", "test"):
            st = s[split_name]
            print(f"{split_name:<8} {st['total']:>7} {st['fraud']:>7} "
                  f"{st['non_fraud']:>7} {st['fraud_ratio']:>7.2%}")
        print("=" * 55)


def _loc_bo_trung_lap(
    texts: list[str],
    labels: list[int],
) -> tuple[list[str], list[int]]:
    seen: set[str] = set()
    clean_texts, clean_labels = [], []
    for text, label in zip(texts, labels):
        if text not in seen:
            seen.add(text)
            clean_texts.append(text)
            clean_labels.append(label)
    removed = len(texts) - len(clean_texts)
    if removed:
        print(f"[DataSplit] Đã loại {removed} mẫu exact-duplicate.")
    return clean_texts, clean_labels


def tao_train_val_test_split(
    texts: list[str],
    labels: list[int],
    ty_le_train: float = 0.8,
    ty_le_val:   float = 0.1,
    seed: int = 42,
    loc_trung_lap: bool = True,
) -> DataSplit:
    from sklearn.model_selection import train_test_split

    texts_list  = list(texts)
    labels_list = list(labels)

    if loc_trung_lap:
        texts_list, labels_list = _loc_bo_trung_lap(texts_list, labels_list)

    ty_le_val_test = 1.0 - ty_le_train
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts_list, labels_list,
        test_size=ty_le_val_test,
        random_state=seed,
        stratify=labels_list,
    )

    ty_le_val_trong_tmp = ty_le_val / ty_le_val_test
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=1.0 - ty_le_val_trong_tmp,
        random_state=seed,
        stratify=y_tmp,
    )

    split = DataSplit(
        train_texts=X_train, train_labels=y_train,
        val_texts=X_val,     val_labels=y_val,
        test_texts=X_test,   test_labels=y_test,
    )
    split.print_summary()
    return split


def luu_split_ra_file(
    split: DataSplit,
    thu_muc: str = ".",
    ten_file_prefix: str = "split",
) -> dict[str, str]:
    out_dir = Path(thu_muc)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split_name, texts, labels in [
        ("train", split.train_texts, split.train_labels),
        ("val",   split.val_texts,   split.val_labels),
        ("test",  split.test_texts,  split.test_labels),
    ]:
        file_path = out_dir / f"{ten_file_prefix}_{split_name}.jsonl"
        with file_path.open("w", encoding="utf-8") as fh:
            for text, label in zip(texts, labels):
                fh.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
        paths[split_name] = str(file_path.resolve())
        print(f"[DataSplit] Đã lưu {split_name}: {file_path} ({len(texts)} mẫu)")
    return paths


def tai_split_tu_file(
    thu_muc: str = ".",
    ten_file_prefix: str = "split",
) -> DataSplit:
    in_dir = Path(thu_muc)
    result: dict[str, tuple[list[str], list[int]]] = {}
    for split_name in ("train", "val", "test"):
        file_path = in_dir / f"{ten_file_prefix}_{split_name}.jsonl"
        texts, labels = [], []
        with file_path.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                texts.append(obj["text"])
                labels.append(int(obj["label"]))
        result[split_name] = (texts, labels)
        print(f"[DataSplit] Đã tải {split_name}: {len(texts)} mẫu")
    return DataSplit(
        train_texts=result["train"][0], train_labels=result["train"][1],
        val_texts=result["val"][0],     val_labels=result["val"][1],
        test_texts=result["test"][0],   test_labels=result["test"][1],
    )


# ===========================================================================
# MODEL 1 — TF-IDF + Logistic Regression  (giữ nguyên)
# ===========================================================================

class BoTFIDF:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []

    def fit(self, texts: Iterable[str]) -> None:
        documents = [tien_xu_ly_day_du(text)[0] for text in texts]
        doc_count = len(documents)
        doc_freq  = Counter()
        for tokens in documents:
            for token in set(tokens):
                doc_freq[token] += 1
        vocabulary      = sorted(doc_freq.keys())
        self.vocabulary = {word: idx for idx, word in enumerate(vocabulary)}
        self.idf = [
            math.log((1 + doc_count) / (1 + doc_freq[word])) + 1.0
            for word in vocabulary
        ]

    def transform(self, texts: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        vocab_size = len(self.vocabulary)
        for text in texts:
            tokens, _ = tien_xu_ly_day_du(text)
            counts    = Counter(tokens)
            total     = sum(counts.values()) or 1
            vector    = [0.0] * vocab_size
            for token, count in counts.items():
                if token in self.vocabulary:
                    idx         = self.vocabulary[token]
                    vector[idx] = (count / total) * self.idf[idx]
            vectors.append(vector)
        return vectors

    def fit_transform(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        self.fit(text_list)
        return self.transform(text_list)

    def explain_top_terms(self, text: str, weights: list[float], top_k: int = 5) -> list[str]:
        vector   = self.transform([text])[0]
        inv_vocab = {idx: word for word, idx in self.vocabulary.items()}
        scored   = [
            (value * weights[idx], inv_vocab[idx])
            for idx, value in enumerate(vector)
            if value > 0
        ]
        scored.sort(key=lambda item: abs(item[0]), reverse=True)
        return [word for _, word in scored[:top_k]]


class MoHinhGianLan:
    """Model 1 — TF-IDF + Logistic Regression (baseline)."""

    def __init__(self, learning_rate: float = 0.25, epochs: int = 500) -> None:
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.vectorizer    = BoTFIDF()
        self.weights: list[float] = []
        self.bias       = 0.0
        self.is_trained = False
        self.threshold  = 0.5

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(min(value, 30.0), -30.0)
        return 1.0 / (1.0 + math.exp(-value))

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        text_list = list(texts)
        y = list(labels)
        X = self.vectorizer.fit_transform(text_list)
        if not X:
            raise RuntimeError("Không có dữ liệu huấn luyện.")
        feature_count = len(X[0])
        self.weights  = [0.0] * feature_count
        self.bias     = 0.0
        positives = sum(y)
        negatives = len(y) - positives
        pos_w = len(y) / (2 * positives) if positives else 1.0
        neg_w = len(y) / (2 * negatives) if negatives else 1.0
        for _ in range(self.epochs):
            grad_w = [0.0] * feature_count
            grad_b = 0.0
            for features, target in zip(X, y):
                linear     = sum(w * v for w, v in zip(self.weights, features)) + self.bias
                prediction = self._sigmoid(linear)
                sw         = pos_w if target == 1 else neg_w
                error      = (prediction - target) * sw
                for idx, value in enumerate(features):
                    grad_w[idx] += error * value
                grad_b += error
            scale = 1 / len(X)
            for idx in range(feature_count):
                self.weights[idx] -= self.learning_rate * grad_w[idx] * scale
            self.bias -= self.learning_rate * grad_b * scale
        self.is_trained = True

    def predict_proba(self, text: str) -> tuple[float, float]:
        if not self.is_trained:
            raise RuntimeError("Model chưa được huấn luyện.")
        features   = self.vectorizer.transform([text])[0]
        linear     = sum(w * v for w, v in zip(self.weights, features)) + self.bias
        fraud_prob = self._sigmoid(linear)
        return 1.0 - fraud_prob, fraud_prob

    def predict(self, text: str, threshold: float | None = None) -> PredictionResult:
        non_fraud_prob, fraud_prob = self.predict_proba(text)
        threshold_used = self.threshold if threshold is None else threshold
        red_flags      = phat_hien_co_do(text)
        adjusted_score = min(1.0, fraud_prob + 0.05 * len(red_flags))
        label          = 1 if adjusted_score >= threshold_used else 0
        tokens_sach, van_ban_embedding = tien_xu_ly_day_du(text)
        van_ban_sach   = tao_van_ban_hien_thi(text)
        doan_nghi_ngo  = tim_doan_nghi_ngo(text)
        chat_luong     = tinh_chat_luong_van_ban(text, tokens_sach)
        muc_do_rui_ro  = xep_muc_do_rui_ro(adjusted_score)
        top_terms      = self.vectorizer.explain_top_terms(text, self.weights)
        explanation: list[str] = []
        if red_flags:
            explanation.append(f"Các red flags theo rule-based: {', '.join(red_flags)}")
        explanation.append(f"Xác suất gian lận của mô hình: {fraud_prob:.3f}")
        explanation.append(f"Điểm rủi ro sau rule-based: {adjusted_score:.3f}")
        explanation.append(f"Chất lượng văn bản/OCR: {chat_luong:.3f}")
        if top_terms:
            explanation.append(f"Từ khóa tác động mạnh: {', '.join(top_terms)}")
        return PredictionResult(
            label=label, muc_do_rui_ro=muc_do_rui_ro,
            model_probability_fraud=fraud_prob,
            model_probability_non_fraud=non_fraud_prob,
            probability_fraud=adjusted_score, probability_non_fraud=1 - adjusted_score,
            threshold_used=threshold_used,
            chat_luong_van_ban=chat_luong, red_flags=red_flags,
            explanation=explanation, top_terms=top_terms,
            van_ban_sach=van_ban_sach, van_ban_embedding=van_ban_embedding,
            doan_nghi_ngo=doan_nghi_ngo,
        )


# ===========================================================================
# MODEL 2 — PhoBERT fine-tune  (thêm recursive chunking vào predict_proba)
# ===========================================================================

_TorchDatasetBase = TorchDataset if PHOBERT_AVAILABLE else object  # type: ignore[misc]


class _PhoBERTDataset(_TorchDatasetBase):  # type: ignore[misc]
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts  = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {"text": self.texts[idx], "labels": self.labels[idx]}


def _tao_phobert_collate_fn(tokenizer, max_len: int):
    def collate(batch):
        texts  = [item["text"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])
        enc    = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt",
        )
        enc["labels"] = labels
        return enc
    return collate



class MoHinhGianLanPhoBERT:
    """
    Model 2 — PhoBERT fine-tune (main model).

    Cải tiến so với bản trước:
    - Recursive chunking trong predict_proba() xử lý văn bản dài (PDF nhiều trang).
    - Tổng hợp điểm theo weighted mean (trọng số = độ dài chunk) + max.
    - DataSplit + early stopping + focal loss + optimal threshold.
    """

    def __init__(
        self,
        model_name:       str   = PHOBERT_MODEL_NAME,
        checkpoint_path:  str   = PHOBERT_CHECKPOINT,
        max_len:          int   = PHOBERT_MAX_LEN,
        batch_size:       int   = PHOBERT_BATCH_SIZE,
        epochs:           int   = PHOBERT_EPOCHS,
        learning_rate:    float = PHOBERT_LR,
        threshold_metric: str   = PHOBERT_THRESHOLD_METRIC,
        patience:         int   = PHOBERT_PATIENCE,
    ) -> None:
        if not PHOBERT_AVAILABLE:
            raise RuntimeError(
                "Thiếu thư viện PhoBERT. Hãy cài: pip install torch transformers sentencepiece"
            )
        self.checkpoint_path  = Path(checkpoint_path)
        self.max_len          = max_len
        self.batch_size       = batch_size
        self.epochs           = epochs
        self.learning_rate    = learning_rate
        self.threshold_metric = threshold_metric
        self.patience         = patience
        self.is_trained       = False
        self.threshold        = 0.5
        self.best_val_metrics: dict[str, float] = {}
        self.metadata_path    = self.checkpoint_path.with_suffix(".meta.json")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PhoBERT] Device: {self.device}")

        local_only = self.checkpoint_path.exists()
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        print(f"[PhoBERT] Đang load tokenizer '{model_name}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_only,
        )
        print(f"[PhoBERT] Đang load model '{model_name}' ...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2,
            attn_implementation="eager",
            local_files_only=local_only,
            use_safetensors=False,
        )
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Load dataset JSON / JSONL
    # ------------------------------------------------------------------

    @staticmethod
    def load_dataset_json(json_path: str) -> tuple[list[str], list[int]]:
        raw = Path(json_path).read_text(encoding="utf-8").strip()
        if raw.startswith("{"):
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            data  = [json.loads(ln) for ln in lines]
        else:
            data = json.loads(raw)
        if isinstance(data, list):
            texts  = [str(item["text"])  for item in data]
            labels = [int(item["label"]) for item in data]
        elif isinstance(data, dict):
            texts  = [str(t)  for t in data["texts"]]
            labels = [int(lb) for lb in data["labels"]]
        else:
            raise ValueError(f"Format JSON không hỗ trợ: {type(data)}")
        n_fraud = sum(labels)
        print(f"[PhoBERT] Dataset: {len(texts)} mẫu ({n_fraud} fraud / {len(labels) - n_fraud} non-fraud)")
        return texts, labels

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        if self._load_checkpoint():
            return
        self._train(list(texts), list(labels))

    def fit_from_json(self, json_path: str) -> None:
        if self._load_checkpoint():
            return
        texts, labels = self.load_dataset_json(json_path)
        self._train(texts, labels)

    def fit_from_split(
        self,
        split: DataSplit,
        luu_split: bool = True,
        thu_muc_split: str = ".",
    ) -> dict[str, float]:
        if self._load_checkpoint():
            print("[PhoBERT] Checkpoint sẵn có → bỏ qua train, đánh giá test set.")
            return self._evaluate(split.test_texts, split.test_labels, "Test")
        if luu_split:
            luu_split_ra_file(split, thu_muc=thu_muc_split, ten_file_prefix="phobert_split")
        self._train_with_validation(
            train_texts=split.train_texts, train_labels=split.train_labels,
            val_texts=split.val_texts,     val_labels=split.val_labels,
        )
        return self._evaluate(split.test_texts, split.test_labels, "Test")

    # ------------------------------------------------------------------
    # Internal train helpers
    # ------------------------------------------------------------------

    def _build_class_weights(self, labels: list[int]) -> "torch.Tensor":
        positives = sum(labels)
        negatives = len(labels) - positives
        pos_w = len(labels) / (2 * positives) if positives else 1.0
        neg_w = len(labels) / (2 * negatives) if negatives else 1.0
        return torch.tensor([neg_w, pos_w], dtype=torch.float).to(self.device)

    def _build_loss(self, labels: list[int]) -> "nn.CrossEntropyLoss":
        """Dùng nn.CrossEntropyLoss với class weights — đủ cho imbalance 65/35 đến 70/30."""
        return nn.CrossEntropyLoss(weight=self._build_class_weights(labels))

    def _predict_scores_batch(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        self.model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start:start + self.batch_size]
                enc = self.tokenizer(
                    batch_texts, truncation=True, padding=True,
                    max_length=self.max_len, return_tensors="pt",
                )
                out   = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                )
                probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().tolist()
                scores.extend(float(p) for p in probs)
        return scores

    def _save_metadata(self) -> None:
        payload = {
            "threshold":         self.threshold,
            "threshold_metric":  self.threshold_metric,
            "best_val_metrics":  self.best_val_metrics,
        }
        self.metadata_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            return
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.threshold        = float(payload.get("threshold", 0.5))
            self.best_val_metrics = dict(payload.get("best_val_metrics", {}))
        except Exception as exc:
            print(f"[PhoBERT] Không load được metadata: {exc}")

    def _train(self, texts: list[str], labels: list[int]) -> None:
        dataset    = _PhoBERTDataset(texts, labels)
        loader     = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=_tao_phobert_collate_fn(self.tokenizer, self.max_len),
        )
        optimizer  = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion  = self._build_loss(labels)
        print(f"[PhoBERT] Bắt đầu fine-tune: {len(texts)} mẫu, {self.epochs} epochs ...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels   = batch["labels"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(len(loader), 1)
            print(f"[PhoBERT] Epoch [{epoch + 1}/{self.epochs}] — Loss: {avg:.4f}")
        self.is_trained = True
        torch.save(self.model.state_dict(), self.checkpoint_path)
        self._save_metadata()
        print(f"[PhoBERT] Đã lưu checkpoint → {self.checkpoint_path}")

    def _train_with_validation(
        self,
        train_texts: list[str], train_labels: list[int],
        val_texts:   list[str], val_labels:   list[int],
    ) -> None:
        train_dataset = _PhoBERTDataset(train_texts, train_labels)
        val_dataset   = _PhoBERTDataset(val_texts,   val_labels)
        collate_fn    = _tao_phobert_collate_fn(self.tokenizer, self.max_len)
        train_loader  = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  collate_fn=collate_fn)
        val_loader    = DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        optimizer     = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion     = self._build_loss(train_labels)

        best_checkpoint = self.checkpoint_path.with_suffix(".best.pt")
        best_metric = best_auprc = -1.0
        best_threshold = 0.5
        no_improve = 0

        print(
            f"[PhoBERT] Train với validation:\n"
            f"  Train: {len(train_texts)} | Val: {len(val_texts)} | Epochs: {self.epochs}"
        )
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'F2':>8} {'AUPRC':>8} {'Thr':>6} {'Best':>6}")
        print("-" * 70)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels   = batch["labels"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            avg_train = train_loss / max(len(train_loader), 1)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids      = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    batch_labels   = batch["labels"].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    val_loss += criterion(outputs.logits, batch_labels).item()
            avg_val    = val_loss / max(len(val_loader), 1)
            val_scores = self._predict_scores_batch(val_texts)
            threshold, val_metrics = tim_nguong_toi_uu(
                val_labels, val_scores, metric=self.threshold_metric
            )
            metric_val    = val_metrics.get(self.threshold_metric, 0.0)
            current_auprc = val_metrics.get("auprc", 0.0)

            is_best = (
                metric_val > best_metric + 1e-8 or
                (abs(metric_val - best_metric) <= 1e-8 and current_auprc > best_auprc + 1e-8)
            )
            if is_best:
                best_metric = metric_val
                best_auprc  = current_auprc
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
                print(f"[PhoBERT] Early stopping sau {self.patience} epoch không cải thiện.")
                break

        if best_checkpoint.exists():
            state = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(state)
            self.threshold = best_threshold
            torch.save(self.model.state_dict(), self.checkpoint_path)
            self._save_metadata()
            print(
                f"[PhoBERT] Best checkpoint: {self.threshold_metric}={best_metric:.4f} "
                f"auprc={best_auprc:.4f} threshold={best_threshold:.3f}"
            )
        self.is_trained = True

    def _evaluate(
        self, texts: list[str], labels: list[int], split_name: str = "Eval"
    ) -> dict[str, float]:
        y_scores = self._predict_scores_batch(texts)
        metrics  = danh_gia_du_doan(labels, y_scores, threshold=self.threshold)
        metrics["threshold"] = self.threshold
        print(f"\n[PhoBERT] Kết quả {split_name} set ({len(texts)} mẫu):")
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
            self._load_metadata()
            print(f"[PhoBERT] Đã load checkpoint '{self.checkpoint_path}'")
            return True
        except Exception as exc:
            print(f"[PhoBERT] Không load được checkpoint: {exc}")
            return False

    # ------------------------------------------------------------------
    # Predict — với recursive chunking cho văn bản dài
    # ------------------------------------------------------------------

    def _predict_proba_single(self, text: str) -> tuple[float, float]:
        """Predict trên 1 đoạn đơn lẻ (nội bộ)."""
        self.model.eval()
        enc = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt",
        )
        with torch.no_grad():
            out   = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().tolist()
        return float(probs[0]), float(probs[1])

    def predict_proba(self, text: str) -> tuple[float, float]:
        """
        Trả về (prob_non_fraud, prob_fraud).

        Nếu text ngắn  → predict trực tiếp.
        Nếu text dài   → recursive chunking → predict từng chunk
                       → tổng hợp: 0.6×max + 0.4×weighted_mean(theo độ dài chunk).
        """
        if not self.is_trained:
            raise RuntimeError("Model chưa được huấn luyện.")

        # Ngưỡng: ước tính ~3 ký tự/token
        if len(text) <= CHUNK_MAX_CHARS:
            return self._predict_proba_single(text)

        # --- Recursive chunking ---
        chunks = tach_chunks_recursive(
            text,
            max_chars=CHUNK_MAX_CHARS,
            overlap=CHUNK_OVERLAP,
        )
        if not chunks:
            return self._predict_proba_single(text)

        fraud_scores: list[float] = []
        chunk_lens:   list[int]   = []
        for chunk in chunks:
            _, fraud_prob = self._predict_proba_single(chunk)
            fraud_scores.append(fraud_prob)
            chunk_lens.append(len(chunk))

        max_score = max(fraud_scores)

        # Weighted mean theo độ dài chunk (đoạn dài → trọng số cao hơn)
        total_len     = sum(chunk_lens) or 1
        weighted_mean = sum(s * l for s, l in zip(fraud_scores, chunk_lens)) / total_len

        # Tổng hợp: 60% max + 40% weighted_mean
        final_fraud = 0.6 * max_score + 0.4 * weighted_mean

        print(
            f"[PhoBERT] Chunking: {len(chunks)} chunks | "
            f"max={max_score:.3f} w_mean={weighted_mean:.3f} final={final_fraud:.3f}"
        )
        return 1.0 - final_fraud, final_fraud

    def _explain_top_terms(self, text: str, top_k: int = 5) -> list[str]:
        if not self.is_trained:
            return []
        # Với text dài, chỉ lấy chunk đầu tiên để explain (chunk quan trọng nhất về cấu trúc)
        if len(text) > CHUNK_MAX_CHARS:
            chunks = tach_chunks_recursive(text, max_chars=CHUNK_MAX_CHARS, overlap=0)
            text = chunks[0] if chunks else text[:CHUNK_MAX_CHARS]

        self.model.eval()
        enc    = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=self.max_len, return_tensors="pt",
        )
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
                output_attentions=True,
            )
        if not out.attentions:
            return []
        cls_attn = out.attentions[-1][0][:, 0, :].mean(dim=0).cpu().tolist()
        scored = []
        for score, tok in zip(cls_attn, tokens):
            if tok in {"<s>", "</s>", "<pad>", "<unk>"}:
                continue
            clean = tok.lstrip("▁")
            if clean.endswith("@@") or not all(c.isalpha() for c in clean) or len(clean) < 2:
                continue
            scored.append((score, clean))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        result: list[str] = []
        for _, tok in scored:
            if tok not in seen:
                seen.add(tok)
                result.append(tok)
            if len(result) >= top_k:
                break
        return result

    def predict(self, text: str, threshold: float | None = None) -> PredictionResult:
        non_fraud_prob, fraud_prob = self.predict_proba(text)
        threshold_used = self.threshold if threshold is None else threshold
        red_flags      = phat_hien_co_do(text)
        adjusted_score = min(1.0, fraud_prob + 0.05 * len(red_flags))
        label          = 1 if adjusted_score >= threshold_used else 0
        tokens_sach, van_ban_embedding = tien_xu_ly_day_du(text)
        van_ban_sach   = tao_van_ban_hien_thi(text)
        doan_nghi_ngo  = tim_doan_nghi_ngo(text)
        chat_luong     = tinh_chat_luong_van_ban(text, tokens_sach)
        muc_do_rui_ro  = xep_muc_do_rui_ro(adjusted_score)
        top_terms      = self._explain_top_terms(text)
        explanation: list[str] = []
        if red_flags:
            explanation.append(f"Các red flags theo rule-based: {', '.join(red_flags)}")
        explanation.append(f"Xác suất gian lận của mô hình: {fraud_prob:.3f}")
        explanation.append(f"Điểm rủi ro sau rule-based: {adjusted_score:.3f}")
        explanation.append(f"Chất lượng văn bản/OCR: {chat_luong:.3f}")
        if top_terms:
            explanation.append(
                "Các token được attention ưu tiên (chỉ mang tính gợi ý, không phải giải thích nhân quả): "
                + ", ".join(top_terms)
            )
        return PredictionResult(
            label=label, muc_do_rui_ro=muc_do_rui_ro,
            model_probability_fraud=fraud_prob,
            model_probability_non_fraud=non_fraud_prob,
            probability_fraud=adjusted_score, probability_non_fraud=1 - adjusted_score,
            threshold_used=threshold_used,
            chat_luong_van_ban=chat_luong, red_flags=red_flags,
            explanation=explanation, top_terms=top_terms,
            van_ban_sach=van_ban_sach, van_ban_embedding=van_ban_embedding,
            doan_nghi_ngo=doan_nghi_ngo,
        )


# ===========================================================================
# HELPER FUNCTIONS — metrics, utils  (giữ nguyên)
# ===========================================================================

def tinh_chat_luong_van_ban(text: str, tokens_sach: list[str]) -> float:
    raw_tokens = xu_ly_van_ban(text)
    if not raw_tokens:
        return 0.0
    clean_ratio      = len(tokens_sach) / len(raw_tokens)
    long_token_ratio = sum(1 for t in tokens_sach if len(t) >= 3) / max(len(tokens_sach), 1)
    ascii_noise      = sum(1 for t in raw_tokens if any(c.isdigit() for c in t) and len(t) > 6) / len(raw_tokens)
    score = 0.6 * clean_ratio + 0.5 * long_token_ratio - 0.3 * ascii_noise
    return max(0.0, min(1.0, score))


def xep_muc_do_rui_ro(score: float) -> str:
    if score >= 0.75:
        return "Cao"
    if score >= 0.45:
        return "Trung bình"
    return "Thấp"


def tim_nguong_toi_uu(
    y_true: list[int],
    y_scores: list[float],
    metric: str = "f2",
) -> tuple[float, dict[str, float]]:
    """
    Tìm threshold tối ưu dùng sklearn.metrics.precision_recall_curve.
    Duyệt tất cả threshold trên PR curve, chọn cái cho metric cao nhất.
    """
    if not y_scores or sum(y_true) == 0:
        return 0.5, danh_gia_du_doan(y_true, y_scores, threshold=0.5)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    best_threshold = 0.5
    best_score     = -1.0
    best_metrics: dict[str, float] = {}

    for p, r, thr in zip(precisions[:-1], recalls[:-1], thresholds):
        if metric == "f2":
            score = fbeta_score_from_pr(p, r, beta=2.0)
        elif metric == "f1":
            score = fbeta_score_from_pr(p, r, beta=1.0)
        elif metric == "f0_5":
            score = fbeta_score_from_pr(p, r, beta=0.5)
        elif metric == "recall":
            score = r
        elif metric == "precision":
            score = p
        else:
            score = fbeta_score_from_pr(p, r, beta=2.0)

        if score > best_score:
            best_score     = score
            best_threshold = float(thr)
            best_metrics   = danh_gia_du_doan(y_true, y_scores, threshold=best_threshold)

    if not best_metrics:
        best_metrics = danh_gia_du_doan(y_true, y_scores, threshold=best_threshold)

    return best_threshold, best_metrics


def fbeta_score_from_pr(precision: float, recall: float, beta: float) -> float:
    """Tính F-beta từ precision và recall (dùng nội bộ trong tim_nguong_toi_uu)."""
    beta_sq = beta * beta
    denom   = beta_sq * precision + recall
    return (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0


def danh_gia_du_doan(
    y_true: list[int],
    y_scores: list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Tính toàn bộ metrics dùng sklearn.metrics."""
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    zero_div = 0.0

    # sklearn yêu cầu ít nhất 1 positive để tính một số metrics
    has_pos = sum(y_true) > 0
    has_pred_pos = sum(y_pred) > 0

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_div)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=zero_div)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=zero_div)),
        "f0_5":      float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=zero_div)),
        "f2":        float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=zero_div)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "auc_roc":   float(roc_auc_score(y_true, y_scores)) if has_pos else 0.0,
        "auprc":     float(average_precision_score(y_true, y_scores)) if has_pos else 0.0,
        "mcc":       float(matthews_corrcoef(y_true, y_pred)) if (has_pos and has_pred_pos) else 0.0,
    }


def doc_file_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


# ===========================================================================
# TRAINING DATA MẶC ĐỊNH  (fallback khi không có samples.jsonl)
# ===========================================================================

TRAINING_TEXTS = [
    "Doanh thu tăng nhờ hợp đồng hợp lệ và đã được đối chiếu với sao kê ngân hàng.",
    "Chi phí được ghi nhận chính xác và kiểm soát nội bộ vận hành hiệu quả.",
    "Tất cả giao dịch đều có xác nhận bên thứ ba và chứng từ đầy đủ.",
    "Kiểm kê hàng tồn kho khớp với sổ kho và dòng tiền từ hoạt động kinh doanh dương.",
    "Thuyết minh minh bạch, biên lợi nhuận gộp ổn định và không có giao dịch bất thường.",
    "Doanh thu bị thổi phồng từ hóa đơn giả và xác nhận khách hàng đáng ngờ.",
    "Doanh nghiệp che giấu nợ phải trả trong các thỏa thuận ngoài bảng cân đối với bên liên quan.",
    "Lợi nhuận bị thao túng để đạt chỉ tiêu và thay đổi chính sách kế toán không có giải trình.",
    "Dòng tiền từ hoạt động kinh doanh âm dù doanh thu tăng mạnh vào cuối kỳ.",
    "Phát hiện giao dịch bất thường với bên liên quan và thiếu phụ lục hợp đồng.",
    "Doanh thu được ghi nhận sớm kèm thỏa thuận ngầm và dấu hiệu hóa đơn giả.",
    "Ban lãnh đạo thay đổi kiểm toán viên liên tục và áp lực đạt KPI lợi nhuận rất cao.",
]

TRAINING_LABELS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

TEST_CASES = [
    ("Doanh thu tăng hợp lệ nhờ hợp đồng đã ký và tiền thu về tốt từ khách hàng.", 0),
    ("Doanh thu khống từ hóa đơn giả được dùng để thổi phồng lợi nhuận và che giấu nợ phải trả.", 1),
    ("Doanh nghiệp thay đổi chính sách kế toán và có nhiều giao dịch với bên liên quan cần lưu ý.", 1),
    ("Thuyết minh rõ ràng, chi phí bình thường và số dư các tài khoản đã được xác minh.", 0),
]


# Compatibility facade for older imports from engine.py.
from engine_common import PredictionResult as PredictionResult  # noqa: E402,F401
from engine_registry import tao_ket_qua_du_doan as _tao_ket_qua_du_doan_registry  # noqa: E402


def tao_ket_qua_du_doan(
    loai: str,
    prediction,
    model_source: str = "unknown",
    top_terms_method: str = "unknown",
) -> dict:
    result = _tao_ket_qua_du_doan_registry(
        loai,
        prediction,
        model_source=model_source,
        top_terms_method=top_terms_method,
    )
    result.setdefault("label_id", result.get("label_int"))
    result.setdefault("risk_score", result.get("fraud_probability"))
    result.setdefault("score_semantics", {})
    result["score_semantics"].setdefault(
        "model_fraud_probability",
        "raw_model_probability",
    )
    result["score_semantics"].setdefault(
        "fraud_probability",
        "rule_adjusted_risk_score",
    )
    return result
