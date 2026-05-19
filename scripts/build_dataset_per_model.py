#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine_common import (
    phat_hien_co_do,
    tach_doan,
    tien_xu_ly_day_du,
)

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


SEED = 42
DATASET_VERSION = "v1.0"
LABEL_TO_INT = {"BINH_THUONG": 0, "CANH_BAO": 1, "VI_PHAM": 2, "KHONG": 3}
RISK_TO_INT = {"LOW_RISK": 0, "MEDIUM_RISK": 1, "HIGH_RISK": 2}

M1_SYSTEM_PROMPT = (
    "Ban la kiem toan vien AI. Chi ket luan khi co bang chung cu the. "
    "Tra loi dang JSON voi label, risk_level, standard_ref, finding, confidence."
)

M1_STANDARDS = [
    {
        "chunk_id": "TT48-Art6-chunk01",
        "source": "TT48/2019/TT-BTC",
        "article": "Dieu 6",
        "title": "Du phong no phai thu kho doi",
        "text": (
            "Doanh nghiep phai trich lap du phong no phai thu kho doi doi voi "
            "cac khoan no qua han thanh toan tren 6 thang hoac co bang chung "
            "khach no mat kha nang thanh toan."
        ),
    },
    {
        "chunk_id": "TT200-TK2293-chunk01",
        "source": "TT200/2014/TT-BTC",
        "article": "TK2293",
        "title": "Du phong ton that tai san va no phai thu",
        "text": (
            "Tai khoan 2293 phan anh gia tri du phong no phai thu kho doi va "
            "viec hoan nhap du phong khi rui ro khong con ton tai."
        ),
    },
    {
        "chunk_id": "VAS14-chunk01",
        "source": "VAS 14",
        "article": "Doanh thu va thu nhap khac",
        "title": "Dieu kien ghi nhan doanh thu",
        "text": (
            "Doanh thu chi duoc ghi nhan khi doanh nghiep da chuyen giao phan "
            "lon rui ro va loi ich, doanh thu duoc xac dinh tuong doi chac chan "
            "va co kha nang thu duoc loi ich kinh te."
        ),
    },
    {
        "chunk_id": "VAS01-chunk01",
        "source": "VAS 01",
        "article": "Chuan muc chung",
        "title": "Trung thuc va hop ly",
        "text": (
            "Bao cao tai chinh phai trinh bay trung thuc, hop ly tinh hinh tai "
            "chinh, ket qua kinh doanh va luu chuyen tien te cua doanh nghiep."
        ),
    },
]

M1_RULES = [
    {
        "id": "R001",
        "name": "Khong trich du phong no kho doi",
        "standard": "TT48/2019 Dieu 6",
        "trigger_pattern": "du phong.{0,30}(0 dong|khong co|chua trich)",
        "context_require": "no phai thu|khoan phai thu|kho doi",
        "label": "CANH_BAO",
        "risk": "CAO",
    },
    {
        "id": "R002",
        "name": "Doanh thu ghi nhan sai ky",
        "standard": "VAS 14",
        "trigger_pattern": "ghi nhan doanh thu.{0,50}(truoc|sai ky|khong du dieu kien)",
        "context_require": "doanh thu|hop dong|hoa don",
        "label": "VI_PHAM",
        "risk": "CAO",
    },
    {
        "id": "R003",
        "name": "Rui ro ben lien quan hoac ngoai bang",
        "standard": "VAS 01",
        "trigger_pattern": "ben lien quan|ngoai bang can doi|che giau no",
        "context_require": "thuyet minh|giao dich|no phai tra",
        "label": "CANH_BAO",
        "risk": "TRUNG_BINH",
    },
]

M3_SIGNAL_LEXICON = {
    "vagueness": [
        "nhat dinh",
        "mot so",
        "nhin chung",
        "tuong doi",
        "o mot muc do nao do",
        "co the noi",
        "phan lon",
    ],
    "hedging": [
        "co the",
        "du kien",
        "hy vong",
        "ky vong",
        "chung toi tin rang",
        "co kha nang",
        "theo du bao",
        "chung toi cho rang",
    ],
    "negation": ["khong", "chua", "chang", "khong the", "khong co", "chua duoc", "khong phai"],
    "passive_markers": ["duoc thuc hien", "da duoc", "se duoc tien hanh", "duoc xem xet"],
}

M3_FEATURE_NAMES = [
    "vague_r",
    "hedge_r",
    "neg_r",
    "passive_r",
    "fog_idx",
    "avg_len",
    "len_var",
    "num_den",
    *[f"topic_{i}" for i in range(8)],
    "sent_pos",
    "sent_neg",
    "sent_neu",
    "time_r",
    "excl_r",
    "quest_r",
    "drift_sc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-model datasets for AuditLLM M1 and FraudLens M3 from the "
            "canonical JSON/JSONL source."
        )
    )
    parser.add_argument("--input", default="samples.jsonl", help="Canonical JSON/JSONL input.")
    parser.add_argument("--output-dir", default="dataset", help="Output dataset root.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--version", default=DATASET_VERSION)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--max-features", type=int, default=5000)
    return parser.parse_args()


def read_json_records(path: Path) -> list[dict[str, object]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset is empty: {path}")
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        if isinstance(payload, dict) and "texts" in payload and "labels" in payload:
            return [
                {"text": text, "label": label}
                for text, label in zip(payload["texts"], payload["labels"])
            ]
        if isinstance(payload, list):
            return [dict(item) for item in payload]
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    raise ValueError(f"Unsupported dataset format: {path}")


def normalize_label(record: dict[str, object]) -> tuple[str, int]:
    raw = record.get("label", record.get("risk_label", 0))
    if isinstance(raw, str):
        value = raw.strip().upper()
        if value in LABEL_TO_INT:
            return value, LABEL_TO_INT[value]
        if value in {"1", "RISK", "HIGH_RISK"}:
            return "CANH_BAO", 1
        return "BINH_THUONG", 0
    label_int = int(raw)
    return ("CANH_BAO", 1) if label_int > 0 else ("BINH_THUONG", 0)


def infer_risk_level(label: str, text: str) -> str:
    if label == "BINH_THUONG":
        return "THAP"
    flags = phat_hien_co_do(text)
    if len(flags) >= 2 or label == "VI_PHAM":
        return "CAO"
    return "TRUNG_BINH"


def infer_standard_refs(text: str, label: str) -> list[str]:
    normalized = " ".join(tien_xu_ly_day_du(text)[0])
    refs: list[str] = []
    if any(term in normalized for term in ("du_phong", "du phong", "kho_doi", "no_phai_thu")):
        refs.extend(["TT48/2019 Dieu 6", "TT200/2014 TK2293"])
    if any(term in normalized for term in ("doanh_thu", "doanh thu", "hoa_don", "hoa don")):
        refs.append("VAS 14")
    if any(term in normalized for term in ("ben_lien_quan", "ngoai_bang", "che_giau")):
        refs.append("VAS 01")
    if not refs and label != "BINH_THUONG":
        refs.append("VAS 01")
    return refs


def build_finding(text: str, label: str, refs: list[str]) -> str:
    if label == "BINH_THUONG":
        return "Khong phat hien canh bao rui ro trong doan van."
    flags = phat_hien_co_do(text)
    if flags:
        return "Phat hien tin hieu rui ro: " + ", ".join(flags)
    if refs:
        return "Can doi chieu them voi " + ", ".join(refs)
    return "Canh bao rui ro sai lech trong yeu can kiem tra bo sung."


def normalize_m1_record(record: dict[str, object], idx: int) -> dict[str, object]:
    text = str(record.get("text", "")).strip()
    label, label_int = normalize_label(record)
    refs = record.get("standard_ref")
    if isinstance(refs, str):
        refs = [refs]
    if not refs:
        refs = infer_standard_refs(text, label)
    return {
        "id": str(record.get("id") or record.get("doc_id") or f"AV-{idx:04d}"),
        "text": text,
        "label": label,
        "label_int": int(label_int),
        "risk_level": str(record.get("risk_level") or infer_risk_level(label, text)),
        "standard_ref": list(refs),
        "finding": str(record.get("finding") or build_finding(text, label, list(refs))),
        "source": str(record.get("source") or record.get("file") or record.get("doc_id") or "unknown"),
        "company_id": str(record.get("company_id") or record.get("ticker") or f"CTY_ANON_{idx:04d}"),
        "dataset_version": DATASET_VERSION,
    }


def split_records(
    records: list[dict[str, object]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    labels = [int(item["label_int"]) for item in records]
    stratify = labels if len(set(labels)) > 1 and min(Counter(labels).values()) >= 2 else None
    train, tmp = train_test_split(
        records,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify,
    )
    tmp_labels = [int(item["label_int"]) for item in tmp]
    tmp_stratify = (
        tmp_labels
        if len(set(tmp_labels)) > 1 and min(Counter(tmp_labels).values()) >= 2
        else None
    )
    dev_share = dev_ratio / max(1.0 - train_ratio, 1e-9)
    dev, test = train_test_split(
        tmp,
        train_size=dev_share,
        random_state=seed,
        stratify=tmp_stratify,
    )
    return {"train": list(train), "dev": list(dev), "test": list(test)}


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def standard_for_ref(ref: str) -> dict[str, object]:
    ref_lower = ref.lower()
    for chunk in M1_STANDARDS:
        haystack = f"{chunk['source']} {chunk['article']} {chunk['title']}".lower()
        if any(part in haystack for part in ref_lower.replace("/", " ").split()):
            return chunk
    return M1_STANDARDS[0]


def pick_negative_standards(positive: dict[str, object], count: int, rng: random.Random) -> list[dict[str, object]]:
    candidates = [chunk for chunk in M1_STANDARDS if chunk["chunk_id"] != positive["chunk_id"]]
    rng.shuffle(candidates)
    return candidates[:count]


def build_m1_proposed(root: Path, splits: dict[str, list[dict[str, object]]], seed: int) -> None:
    rng = random.Random(seed)
    proposed_dir = root / "per_model" / "M1" / "proposed"
    kb_chunks = []
    for idx, chunk in enumerate(M1_STANDARDS):
        item = dict(chunk)
        item["chunk_index"] = idx
        item["total_chunks"] = len(M1_STANDARDS)
        item["chunk_size_tokens"] = 400
        item["chunk_overlap_tokens"] = 80
        kb_chunks.append(item)
    write_jsonl(proposed_dir / "kb_chunks.jsonl", kb_chunks)

    embedding_pairs = []
    reranker_pairs = []
    llm_rows = []
    for index, record in enumerate(splits["train"], start=1):
        refs = list(record.get("standard_ref") or [])
        positive = standard_for_ref(str(refs[0])) if refs else M1_STANDARDS[0]
        negatives = pick_negative_standards(positive, 3, rng)
        embedding_pairs.append(
            {
                "id": f"EP-{index:04d}",
                "anchor": record["text"],
                "positive": f"{positive['source']} {positive['article']}: {positive['text']}",
                "hard_negative": f"{negatives[0]['source']} {negatives[0]['article']}: {negatives[0]['text']}",
                "hard_negatives": [
                    f"{item['source']} {item['article']}: {item['text']}" for item in negatives
                ],
            }
        )
        reranker_pairs.append({"query": record["text"], "passage": positive["text"], "label": 1})
        for neg in negatives:
            reranker_pairs.append({"query": record["text"], "passage": neg["text"], "label": 0})
        llm_rows.append(
            {
                "id": f"LLM-{index:04d}",
                "messages": [
                    {"role": "system", "content": M1_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"[DOAN VAN]\n{record['text']}\n\n"
                            f"[CHUAN MUC]\n{positive['source']} {positive['article']}: {positive['text']}\n"
                            "[YEU CAU] Phan tich va tra loi JSON."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "label": record["label"],
                                "risk_level": record["risk_level"],
                                "standard_ref": record["standard_ref"],
                                "finding": record["finding"],
                                "confidence": 0.82 if record["label"] != "BINH_THUONG" else 0.76,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            }
        )
    write_jsonl(proposed_dir / "embedding_pairs.jsonl", embedding_pairs)
    write_jsonl(proposed_dir / "reranker_pairs.jsonl", reranker_pairs)
    write_jsonl(proposed_dir / "llm_finetune.jsonl", llm_rows)


def build_m1_baselines(root: Path, splits: dict[str, list[dict[str, object]]], max_features: int) -> None:
    b1_dir = root / "per_model" / "M1" / "B1_rule_based"
    write_json(
        b1_dir / "standards_dict.json",
        {
            "TT48/2019": {
                "keywords": ["du phong", "no phai thu", "kho doi", "qua han", "trich lap"],
                "pattern": "(du phong).{0,50}(no phai thu|khoan phai thu)",
                "violation_signals": [
                    {"pattern": "du phong.{0,20}0 dong", "label": "CANH_BAO"},
                    {"pattern": "khong trich|chua trich", "label": "CANH_BAO"},
                ],
            },
            "VAS14": {
                "keywords": ["doanh thu", "ghi nhan", "dieu kien", "hoa don"],
                "pattern": "ghi nhan doanh thu.{0,80}(sai ky|truoc|khong du dieu kien)",
            },
            "VAS01": {
                "keywords": ["trung thuc", "hop ly", "ben lien quan", "ngoai bang can doi"],
                "pattern": "ben lien quan|ngoai bang can doi|che giau",
            },
        },
    )
    rules_yaml = ["rules:"]
    for rule in M1_RULES:
        rules_yaml.extend(
            [
                f"- id: {rule['id']}",
                f"  name: {rule['name']}",
                f"  standard: {rule['standard']}",
                f"  trigger_pattern: '{rule['trigger_pattern']}'",
                f"  context_require: '{rule['context_require']}'",
                f"  label: {rule['label']}",
                f"  risk: {rule['risk']}",
            ]
        )
    (b1_dir / "rules.yaml").write_text("\n".join(rules_yaml) + "\n", encoding="utf-8")
    write_jsonl(b1_dir / "test_inputs.jsonl", splits["test"])

    b2_dir = root / "per_model" / "M1" / "B2_tfidf_lr"
    b2_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), lowercase=True)
    train_texts = [str(item["text"]) for item in splits["train"]]
    vectorizer.fit(train_texts)
    if joblib is not None:
        joblib.dump(vectorizer, b2_dir / "vectorizer.joblib")
    for split_name in ("train", "dev", "test"):
        rows = []
        matrix = vectorizer.transform([str(item["text"]) for item in splits[split_name]])
        for record, vector in zip(splits[split_name], matrix):
            row = {"id": record["id"], "label_int": record["label_int"], "label": record["label"]}
            for col_idx, value in zip(vector.indices, vector.data):
                row[f"tfidf_feat_{col_idx + 1}"] = round(float(value), 8)
            rows.append(row)
        fieldnames = ["id", "label_int", "label"] + [f"tfidf_feat_{i + 1}" for i in range(len(vectorizer.vocabulary_))]
        write_csv(b2_dir / f"{split_name}_features.csv", rows, fieldnames)

    b3_dir = root / "per_model" / "M1" / "B3_bm25_gpt4"
    kb_plain = [{"chunk_id": item["chunk_id"], "text": item["text"]} for item in M1_STANDARDS]
    write_jsonl(b3_dir / "kb_plain.jsonl", kb_plain)
    write_json(
        b3_dir / "bm25_corpus_tokens.json",
        [{"chunk_id": item["chunk_id"], "tokens": str(item["text"]).lower().split()} for item in M1_STANDARDS],
    )
    write_jsonl(
        b3_dir / "prompts_test.jsonl",
        [
            {
                "id": record["id"],
                "system": "Ban la kiem toan vien. Tra loi dang JSON.",
                "user": f"[DOAN VAN]\n{record['text']}\n[TOP5 BM25]\n{{bm25_top5}}",
                "gold_label": record["label"],
            }
            for record in splits["test"]
        ],
    )

    b4_dir = root / "per_model" / "M1" / "B4_rag_norerank"
    (b4_dir / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (b4_dir / "config.yaml").write_text(
        "\n".join(
            [
                "embedding_model: BAAI/bge-m3",
                "vector_db: chromadb",
                "top_k: 5",
                "reranker: null",
                "llm: VinaLLaMA-7B-chat",
                "kb_source: ../proposed/kb_chunks.jsonl",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        b4_dir / "retrieval_test.jsonl",
        [{"id": record["id"], "text": record["text"], "gold_label": record["label"], "top_k": 5} for record in splits["test"]],
    )

    b5_dir = root / "per_model" / "M1" / "B5_gpt4_direct"
    write_jsonl(
        b5_dir / "prompts_test.jsonl",
        [
            {
                "id": record["id"],
                "system": (
                    "Ban la kiem toan vien AI. Phan tich doan van va tra loi JSON "
                    "voi label, risk_level, standard_ref, finding, confidence."
                ),
                "user": f"[TOAN BO DOAN VAN]\n{record['text']}\n\nNote: Khong co chuan muc tham chieu duoc cung cap.",
                "gold_label": record["label"],
                "char_count": len(str(record["text"])),
            }
            for record in splits["test"]
        ],
    )


def ratio(text: str, phrases: list[str]) -> float:
    low = text.lower()
    hits = sum(low.count(item) for item in phrases)
    tokens = max(len(low.split()), 1)
    return hits / tokens


def extract_numbers(text: str) -> list[float]:
    values = []
    for match in re.findall(r"\d+(?:[.,]\d+)*", text):
        try:
            values.append(float(match.replace(".", "").replace(",", ".")))
        except ValueError:
            continue
    return values


def m3_features(record: dict[str, object], previous_text: str = "") -> dict[str, float]:
    text = str(record["text"])
    segments = tach_doan(text) or [text]
    tokenized = [seg.split() for seg in segments]
    sent_lengths = [len(tokens) for tokens in tokenized if tokens]
    avg_len = sum(sent_lengths) / max(len(sent_lengths), 1)
    len_var = sum((value - avg_len) ** 2 for value in sent_lengths) / max(len(sent_lengths), 1)
    words = text.split()
    syllables_proxy = sum(max(1, len(word) // 3) for word in words)
    fog_idx = 0.4 * (avg_len + 100 * syllables_proxy / max(len(words), 1))
    numbers = extract_numbers(text)
    red_flags = phat_hien_co_do(text)
    text_low = text.lower()
    topic_values = [0.0] * 8
    for flag in red_flags:
        topic_values[abs(hash(flag)) % 8] += 1.0 / max(len(red_flags), 1)
    pos_terms = ["tang truong", "cai thien", "hieu qua", "on dinh", "loi nhuan"]
    neg_terms = ["lo", "giam", "rui ro", "kho khan", "thieu", "khong"]
    pos = ratio(text_low, pos_terms)
    neg = ratio(text_low, neg_terms)
    neu = max(0.0, 1.0 - pos - neg)
    years = re.findall(r"\b20\d{2}\b", text)
    drift_score = 0.0
    if previous_text:
        current_terms = set(tien_xu_ly_day_du(text)[0])
        previous_terms = set(tien_xu_ly_day_du(previous_text)[0])
        overlap = len(current_terms & previous_terms) / max(len(current_terms | previous_terms), 1)
        drift_score = 1.0 - overlap
    values = {
        "vague_r": ratio(text_low, M3_SIGNAL_LEXICON["vagueness"]),
        "hedge_r": ratio(text_low, M3_SIGNAL_LEXICON["hedging"]),
        "neg_r": ratio(text_low, M3_SIGNAL_LEXICON["negation"]),
        "passive_r": ratio(text_low, M3_SIGNAL_LEXICON["passive_markers"]),
        "fog_idx": fog_idx,
        "avg_len": avg_len,
        "len_var": len_var,
        "num_den": len(numbers) / max(len(words), 1),
        "sent_pos": pos,
        "sent_neg": neg,
        "sent_neu": neu,
        "time_r": len(years) / max(len(words), 1),
        "excl_r": text.count("!") / max(len(text), 1),
        "quest_r": text.count("?") / max(len(text), 1),
        "drift_sc": float(record.get("drift_score", drift_score) or drift_score),
    }
    for idx, value in enumerate(topic_values):
        values[f"topic_{idx}"] = value
    return {name: float(values.get(name, 0.0)) for name in M3_FEATURE_NAMES}


def normalize_m3_record(record: dict[str, object], idx: int) -> dict[str, object]:
    label, label_int = normalize_label(record)
    risk_label = "HIGH_RISK" if label_int > 0 else "LOW_RISK"
    return {
        "id": str(record.get("id") or f"FS-{idx:04d}"),
        "company_id": str(record.get("company_id") or f"CTY_ANON_{idx:04d}"),
        "year": int(record.get("year") or 2024),
        "doc_section": str(record.get("doc_section") or "financial_statement_note"),
        "text": str(record["text"]),
        "risk_label": str(record.get("risk_label") or risk_label),
        "risk_label_int": int(record.get("risk_label_int") or RISK_TO_INT[risk_label]),
        "signals": [{"type": flag, "span": flag, "weight": 0.5} for flag in phat_hien_co_do(str(record["text"]))],
        "drift_from_prev": bool(record.get("drift_from_prev") or False),
        "drift_score": float(record.get("drift_score") or 0.0),
    }


def build_m3(root: Path, records: list[dict[str, object]], train_ratio: float, dev_ratio: float, seed: int) -> None:
    m3_records = [normalize_m3_record(record, idx) for idx, record in enumerate(records, start=1)]
    for record in m3_records:
        record["label_int"] = record["risk_label_int"]
    splits = split_records(m3_records, train_ratio, dev_ratio, seed)
    split_root = root / "splits"
    for split_name, rows in splits.items():
        write_jsonl(split_root / f"M3_{split_name}.jsonl", rows)

    proposed_dir = root / "per_model" / "M3" / "proposed"
    write_json(proposed_dir / "signal_lexicon.json", M3_SIGNAL_LEXICON)
    all_feature_rows = []
    for split_name in ("train", "dev", "test"):
        phobert_rows = []
        feature_rows = []
        for record in splits[split_name]:
            features = m3_features(record)
            phobert_rows.append({"id": record["id"], "text": record["text"], "label_int": record["risk_label_int"]})
            feature_row = {"id": record["id"], **{name: round(features[name], 8) for name in M3_FEATURE_NAMES}, "label_int": record["risk_label_int"]}
            feature_rows.append(feature_row)
            all_feature_rows.append({**feature_row, "split": split_name})
        write_jsonl(proposed_dir / f"phobert_input_{split_name}.jsonl", phobert_rows)
        write_csv(proposed_dir / f"features_{split_name}.csv", feature_rows, ["id", *M3_FEATURE_NAMES, "label_int"])
    write_csv(proposed_dir / "features.csv", all_feature_rows, ["split", "id", *M3_FEATURE_NAMES, "label_int"])

    b1_dir = root / "per_model" / "M3" / "B1_benford"
    rows = []
    for record in splits["test"]:
        numbers = extract_numbers(str(record["text"]))
        first_digits = [int(str(int(abs(value)))[0]) for value in numbers if abs(value) >= 1]
        observed = Counter(first_digits)
        benford = [math.log10(1 + 1 / digit) for digit in range(1, 10)]
        total = max(sum(observed.values()), 1)
        chi2 = sum(((observed.get(d, 0) / total) - benford[d - 1]) ** 2 / benford[d - 1] for d in range(1, 10))
        rows.append(
            {
                "company_id": record["company_id"],
                "year": record["year"],
                "numbers": numbers[:200],
                "first_digits": first_digits[:200],
                "benford_chi2": round(chi2, 6),
                "benford_pvalue": None,
                "risk_label": record["risk_label"],
            }
        )
    write_jsonl(b1_dir / "financial_numbers.jsonl", rows)

    b2_dir = root / "per_model" / "M3" / "B2_features_xgb"
    write_csv(b2_dir / "features.csv", all_feature_rows, ["split", "id", *M3_FEATURE_NAMES, "label_int"])
    (b2_dir / "config.yaml").write_text(
        "model: XGBoost_or_SVM\ninput: features.csv\nnote: features only, no PhoBERT embedding\n",
        encoding="utf-8",
    )

    b3_dir = root / "per_model" / "M3" / "B3_phobert_only"
    for split_name in ("train", "dev", "test"):
        write_jsonl(
            b3_dir / f"phobert_input_{split_name}.jsonl",
            [{"id": record["id"], "text": record["text"], "label_int": record["risk_label_int"]} for record in splits[split_name]],
        )

    b4_dir = root / "per_model" / "M3" / "B4_gpt4"
    write_jsonl(
        b4_dir / "prompts_test.jsonl",
        [
            {
                "id": record["id"],
                "system": "Ban la chuyen gia phan tich rui ro tai chinh. Tra loi JSON: {risk_level, signals, reason}.",
                "user": f"[DOAN VAN]\n{record['text']}",
                "gold_label": record["risk_label"],
            }
            for record in splits["test"]
        ],
    )

    b5_dir = root / "per_model" / "M3" / "B5_finbert"
    write_jsonl(
        b5_dir / "train_en.jsonl",
        [
            {"id": record["id"], "text_vi": record["text"], "text_en": "", "label_int": record["risk_label_int"]}
            for record in splits["train"]
        ],
    )


def write_stats(root: Path, splits: dict[str, list[dict[str, object]]], version: str, seed: int) -> None:
    payload = {
        "dataset_version": version,
        "seed": seed,
        "module": "M1",
        "splits": {},
    }
    for split_name, rows in splits.items():
        labels = Counter(str(row["label"]) for row in rows)
        payload["splits"][split_name] = {
            "count": len(rows),
            "label_distribution": dict(labels),
            "companies": len({row["company_id"] for row in rows}),
        }
    write_json(root / "stats" / "M1_distribution.json", payload)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    raw_records = read_json_records(input_path)
    m1_records = [normalize_m1_record(record, idx) for idx, record in enumerate(raw_records, start=1)]
    splits = split_records(m1_records, args.train_ratio, args.dev_ratio, args.seed)

    (output_root / "raw").mkdir(parents=True, exist_ok=True)
    (output_root / "processed").mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "processed" / "M1_canonical.jsonl", m1_records)
    for split_name, rows in splits.items():
        write_jsonl(output_root / "splits" / f"M1_{split_name}.jsonl", rows)

    build_m1_proposed(output_root, splits, args.seed)
    build_m1_baselines(output_root, splits, args.max_features)
    build_m3(output_root, raw_records, args.train_ratio, args.dev_ratio, args.seed)
    write_stats(output_root, splits, args.version, args.seed)
    write_json(
        output_root / "dataset_manifest.json",
        {
            "dataset_version": args.version,
            "seed": args.seed,
            "source": str(input_path.resolve()),
            "modules_built": ["M1", "M3"],
            "policy": {
                "test_set_shared": True,
                "augmentation_train_only": True,
                "tfidf_fit_train_only": True,
                "per_model_format_required": True,
            },
        },
    )
    print(f"Built per-model dataset at: {output_root.resolve()}")


if __name__ == "__main__":
    main()
