from __future__ import annotations
# python main.py --eval-split --model phobert --dataset samples.jsonl
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
# pip freeze > requirements-web.txt
# python -m uvicorn app:app --reload
# http://127.0.0.1:8000/docs
from engine import tao_ket_qua_du_doan
from engine_bias import danh_gia_bias_theo_nhom
from engine_common import (
    TEST_CASES,
    TRAINING_LABELS,
    TRAINING_TEXTS,
    danh_gia_du_doan,
    luu_manifest_split,
    luu_split_ra_file,
    tao_split_mac_dinh_tu_json,
    tai_split_tu_file,
)
from engine_document_io import doc_tai_lieu_tu_duong_dan
from engine_governance import kiem_tra_governance_dataset
from engine_registry import (
    MODEL_BASELINE,
    MODEL_CHOICES,
    la_model_transformer,
    lay_lop_mo_hinh,
    tao_mo_hinh_theo_loai,
)

def tao_ten_split_prefix(loai: str, json_path: str) -> str:
    dataset_path = Path(json_path)
    dataset_key = str(dataset_path.resolve() if dataset_path.exists() else dataset_path)
    dataset_slug = dataset_path.stem or "dataset"
    dataset_hash = hashlib.md5(dataset_key.encode("utf-8")).hexdigest()[:8]
    return f"{loai}_split_{dataset_slug}_{dataset_hash}"


def cau_hinh_console_utf8() -> None:
    """
    Tránh UnicodeEncodeError trên Windows console (cp1252/cp437)
    khi in tiếng Việt.
    """
    if os.name != "nt":
        return

    try:
        # Python 3.7+ hỗ trợ reconfigure trực tiếp cho TextIO.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Không làm crash chương trình nếu môi trường không hỗ trợ.
        pass


def _tai_hoac_tao_split(
    loai: str,
    json_path: str,
    seed: int = 42,
):
    split_prefix = tao_ten_split_prefix(loai, json_path)
    split_train = Path(f"{split_prefix}_train.jsonl")
    if split_train.exists():
        print(f"[main] Phát hiện split đã lưu cho {loai} → load lại.")
        return tai_split_tu_file(thu_muc=".", ten_file_prefix=split_prefix)

    split, manifest = tao_split_mac_dinh_tu_json(
        json_path,
        ty_le_train=0.8,
        ty_le_val=0.1,
        seed=seed,
    )
    luu_manifest_split(manifest, f"{split_prefix}_manifest.json")
    return split


def tao_mo_hinh(
    loai: str,
    json_path: str = "samples.jsonl",
    dung_split: bool = True,
    seed: int = 42,
):
    """
    Khởi tạo và huấn luyện model theo --model.
      baseline → TF-IDF + Logistic Regression
      phobert  → PhoBERT fine-tune
      mfinbert → MFinBERT fine-tune

    Cả 3 model:
      - Nếu dung_split=True  → dùng train/val/test split đúng cách.
      - Nếu dung_split=False → fit trên toàn bộ dataset nếu có.
    """
    model = tao_mo_hinh_theo_loai(loai)

    if dung_split and Path(json_path).exists():
        split = _tai_hoac_tao_split(loai, json_path=json_path, seed=seed)
        model.fit_from_split(
            split,
            luu_split=True,
            ten_file_prefix=tao_ten_split_prefix(loai, json_path),
        )
        return model

    if Path(json_path).exists():
        model.fit_from_json(json_path)
    else:
        model.fit(TRAINING_TEXTS, TRAINING_LABELS)
    return model


# ---------------------------------------------------------------------------
# Chức năng chính
# ---------------------------------------------------------------------------

def chay_demo(
    loai: str,
    dung_split: bool = True,
    json_path: str = "samples.jsonl",
    seed: int = 42,
) -> dict:
    model = tao_mo_hinh(loai, json_path=json_path, dung_split=dung_split, seed=seed)
    y_true   = []
    y_scores = []

    print("=" * 72)
    print(f"MASTER FRAUD DETECTION DEMO - TIENG VIET  [model: {loai.upper()}]")
    print("=" * 72)
    print("Bo du lieu: phân loại nhị phân cho phát hiện gian lận kiểm toán")
    print("Metric chính: Precision, Recall, F1, F0.5, F2, AUC-ROC, AUPRC, MCC")
    print("-" * 72)

    for index, (text, label) in enumerate(TEST_CASES, start=1):
        prediction = model.predict(text)
        y_true.append(label)
        y_scores.append(prediction.probability_fraud)
        print(f"Truong hop {index}")
        print(f"Van ban   : {text}")
        print(f"Nhan dung : {'Gian lan' if label == 1 else 'Khong gian lan'}")
        print(f"Du doan   : {'Gian lan' if prediction.label == 1 else 'Khong gian lan'}")
        print(f"Model prob: {prediction.model_probability_fraud:.3f}")
        print(f"Risk score: {prediction.probability_fraud:.3f} (threshold={prediction.threshold_used:.3f})")
        if prediction.red_flags:
            print(f"Red flags : {', '.join(prediction.red_flags)}")
        if prediction.top_terms:
            print(f"Top terms : {', '.join(prediction.top_terms)}")
        print("-" * 72)

    metrics = danh_gia_du_doan(y_true, y_scores)
    print("Danh gia (tren TEST_CASES nhanh)")
    for key, value in metrics.items():
        print(f"{key:>10}: {value:.4f}")

    result = {"model": loai, "metrics": metrics, "cases": len(TEST_CASES)}
    output_path = Path(f"demo_result_{loai}.json")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nDa luu ket qua vao {output_path.resolve()}")
    return result


def phan_tich_text(
    text: str,
    loai: str,
    dung_split: bool = True,
    json_path: str = "samples.jsonl",
    seed: int = 42,
    in_ket_qua: bool = True,
) -> dict:
    model = tao_mo_hinh(loai, json_path=json_path, dung_split=dung_split, seed=seed)
    prediction = model.predict(text)
    result = tao_ket_qua_du_doan(
        loai,
        prediction,
        model_source=getattr(model, "model_source", "unknown"),
        top_terms_method=getattr(model, "top_terms_method", "unknown"),
    )
    if in_ket_qua:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def phan_tich_tai_lieu(
    file_path: str,
    loai: str,
    dung_split: bool = True,
    json_path: str = "samples.jsonl",
    seed: int = 42,
) -> dict:
    document = doc_tai_lieu_tu_duong_dan(file_path)
    result = phan_tich_text(
        document.text,
        loai,
        dung_split=dung_split,
        json_path=json_path,
        seed=seed,
        in_ket_qua=False,
    )
    result["file_path"] = str(Path(file_path))
    result["source_type"] = document.source_type
    result["extraction_method"] = document.extraction_method
    result["ocr_used"] = document.ocr_used
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def quet_thu_muc(
    folder: str,
    loai: str,
    dung_split: bool = True,
    json_path: str = "samples.jsonl",
    seed: int = 42,
) -> dict:
    model = tao_mo_hinh(loai, json_path=json_path, dung_split=dung_split, seed=seed)
    folder_path = Path(folder)
    files = sorted(
        path for path in folder_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
    )

    if not files:
        raise FileNotFoundError(f"Khong tim thay file .txt hoac .pdf trong thu muc: {folder}")

    rows = []
    for file_path in files:
        try:
            document = doc_tai_lieu_tu_duong_dan(str(file_path))
            prediction = model.predict(document.text)
            rows.append({
                 "file_name":           file_path.name,
                 "file_path":           str(file_path),
                 "source_type":         document.source_type,
                 "extraction_method":   document.extraction_method,
                 "ocr_used":            document.ocr_used,
                 "model":               loai,
                 "model_source":        getattr(model, "model_source", "unknown"),
                 "label":               "Gian lan" if prediction.label == 1 else "Khong gian lan",
                 "muc_do_rui_ro":       prediction.muc_do_rui_ro,
                 "model_fraud_probability": round(prediction.model_probability_fraud, 4),
                 "model_non_fraud_probability": round(prediction.model_probability_non_fraud, 4),
                 "raw_text_probability_fraud": round(float(getattr(prediction, "raw_text_probability_fraud", prediction.model_probability_fraud)), 4),
                 "raw_text_probability_non_fraud": round(float(getattr(prediction, "raw_text_probability_non_fraud", prediction.model_probability_non_fraud)), 4),
                 "fraud_probability":   round(prediction.probability_fraud, 4),
                 "non_fraud_probability": round(prediction.probability_non_fraud, 4),
                 "threshold_used":      round(prediction.threshold_used, 4),
                 "chat_luong_van_ban":  round(prediction.chat_luong_van_ban, 4),
                 "van_ban_sach":        prediction.van_ban_sach[:500],
                "van_ban_embedding":   prediction.van_ban_embedding[:500],
                "red_flags":           "; ".join(prediction.red_flags),
                "doan_nghi_ngo":       " || ".join(
                    f"{item['flags']} => {item['snippet']}"
                    for item in prediction.doan_nghi_ngo
                 ),
                 "explanation":         " | ".join(prediction.explanation),
                 "top_terms":           "; ".join(prediction.top_terms),
                 "top_terms_method":    getattr(model, "top_terms_method", "unknown"),
                 "bias_flags":          "; ".join(getattr(prediction, "bias_flags", [])),
             })
        except Exception as exc:
            rows.append({
                 "file_name":           file_path.name,
                 "file_path":           str(file_path),
                 "source_type":         file_path.suffix.lower().lstrip("."),
                 "extraction_method":   "",
                 "ocr_used":            "",
                 "model":               loai,
                 "model_source":        getattr(model, "model_source", "unknown"),
                 "label":               "LOI_DOC_FILE",
                 "muc_do_rui_ro":       "",
                 "model_fraud_probability": "",
                 "model_non_fraud_probability": "",
                 "raw_text_probability_fraud": "",
                 "raw_text_probability_non_fraud": "",
                 "fraud_probability":   "",
                 "non_fraud_probability": "",
                 "threshold_used":      "",
                 "chat_luong_van_ban":  "",
                 "van_ban_sach":        "",
                "van_ban_embedding":   "",
                "red_flags":           "",
                 "doan_nghi_ngo":       "",
                 "explanation":         str(exc),
                 "top_terms":           "",
                 "top_terms_method":    "",
                 "bias_flags":          "",
             })

    csv_path  = Path(f"results_{loai}.csv")
    json_path = Path(f"results_{loai}.json")

    fieldnames = [
        "file_name", "file_path", "source_type", "extraction_method", "ocr_used",
        "model", "model_source", "label", "muc_do_rui_ro",
        "model_fraud_probability", "model_non_fraud_probability",
        "raw_text_probability_fraud", "raw_text_probability_non_fraud",
        "fraud_probability", "non_fraud_probability", "chat_luong_van_ban",
        "threshold_used", "van_ban_sach", "van_ban_embedding", "red_flags",
        "doan_nghi_ngo", "explanation", "top_terms", "top_terms_method", "bias_flags",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    fraud_count = sum(1 for r in rows if r["label"] == "Gian lan")
    normal_count = sum(1 for r in rows if r["label"] == "Khong gian lan")
    error_count = sum(1 for r in rows if r["label"] == "LOI_DOC_FILE")
    txt_count = sum(1 for r in rows if r["source_type"] == "txt")
    pdf_count = sum(1 for r in rows if r["source_type"] == "pdf")
    ocr_count = sum(1 for r in rows if r["ocr_used"] is True)

    summary = {
        "model":        loai,
        "folder":       str(folder_path),
        "total_files":  len(rows),
        "txt_count":    txt_count,
        "pdf_count":    pdf_count,
        "ocr_count":    ocr_count,
        "fraud_count":  fraud_count,
        "normal_count": normal_count,
        "error_count":  error_count,
        "csv_output":   str(csv_path.resolve()),
        "json_output":  str(json_path.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def chay_danh_gia_split(
    loai: str,
    json_path: str = "samples.jsonl",
    seed: int = 42,
) -> dict:
    """Đánh giá model với governance check + train/val/test split đầy đủ."""
    print("=" * 60)
    print(f"DANH GIA MODEL VOI TRAIN/VAL/TEST SPLIT [{loai.upper()}]")
    print("=" * 60)

    governance_report = kiem_tra_governance_dataset(json_path)
    print("[Governance] Step 1-4:")
    for step_name in (
        "step_1_collect_data",
        "step_2_ethics_privacy",
        "step_3_label_quality",
        "step_4_ready_gate",
    ):
        status = governance_report.get(step_name, {}).get("status", "unknown")
        print(f"  - {step_name}: {status}")

    split = _tai_hoac_tao_split(loai, json_path=json_path, seed=seed)
    split_prefix = tao_ten_split_prefix(loai, json_path)
    luu_split_ra_file(split, thu_muc=".", ten_file_prefix=split_prefix)

    model = tao_mo_hinh_theo_loai(loai)
    test_metrics = model.fit_from_split(split, luu_split=False, ten_file_prefix=split_prefix)
    val_metrics = getattr(model, "best_val_metrics", {})
    test_scores = model._predict_scores_batch(split.test_texts)
    bias_report = danh_gia_bias_theo_nhom(
        split.test_texts,
        split.test_labels,
        test_scores,
        threshold=getattr(model, "threshold", 0.5),
    )

    explainability_preview = None
    if la_model_transformer(loai) and split.test_texts and hasattr(model, "explain_all_methods"):
        explainability_preview = {
            "sample_text": split.test_texts[0][:300],
            "methods": model.explain_all_methods(split.test_texts[0], top_k=5),
        }

    result = {
        "model": loai,
        "governance_report": governance_report,
        "split_info": split.summary(),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "bias_report": bias_report,
    }
    if explainability_preview is not None:
        result["explainability_preview"] = explainability_preview

    out = Path(f"split_eval_result_{loai}.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDa luu ket qua danh gia vao {out.resolve()}")
    return result


def chuan_bi_checkpoint_api(
    loai: str,
    json_path: str = "samples.jsonl",
    seed: int = 42,
) -> dict:
    """Train offline và lưu checkpoint để FastAPI chỉ việc inference."""
    print("=" * 60)
    print(f"CHUAN BI CHECKPOINT API [{loai.upper()}]")
    print("=" * 60)

    split = _tai_hoac_tao_split(loai, json_path=json_path, seed=seed)
    split_prefix = tao_ten_split_prefix(loai, json_path)
    luu_split_ra_file(split, thu_muc=".", ten_file_prefix=split_prefix)

    model = tao_mo_hinh_theo_loai(loai)
    test_metrics = model.fit_from_split(
        split,
        luu_split=False,
        ten_file_prefix=split_prefix,
    )
    checkpoint_path = getattr(model, "checkpoint_path", None)
    metadata_path = getattr(model, "metadata_path", None)

    result = {
        "model": loai,
        "dataset": str(Path(json_path)),
        "split_info": split.summary(),
        "test_metrics": test_metrics,
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
        "metadata_path": str(Path(metadata_path).resolve()) if metadata_path else None,
    }
    out = Path(f"prepare_api_result_{loai}.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nDa luu thong tin checkpoint vao {out.resolve()}")
    return result


# ---------------------------------------------------------------------------
# Tham số CLI
# ---------------------------------------------------------------------------

def doc_tham_so() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo phat hien gian lan cho van ban kiem toan tieng Viet."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CHOICES),
        default=MODEL_BASELINE,
        help=(
            "Chon model: 'baseline' (TF-IDF + LR), "
            "'phobert' (PhoBERT fine-tune) hoac "
            "'mfinbert' (MFinBERT fine-tune). Mac dinh: baseline"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="samples.jsonl",
        help="Duong dan toi file JSONL dataset (mac dinh: samples.jsonl)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Tat train/val/test split, dung toan bo data (khong khuyen dung)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed cho split (mac dinh: 42)",
    )
    parser.add_argument("--demo",       action="store_true", help="Chay bo demo danh gia nhanh")
    parser.add_argument("--eval-split", action="store_true", help="Train & danh gia voi governance + train/val/test split day du")
    parser.add_argument(
        "--prepare-api",
        action="store_true",
        help="Train offline va luu checkpoint de FastAPI chi dung cho inference",
    )
    parser.add_argument("--text",       type=str,            help="Phan tich text nhap truc tiep")
    parser.add_argument("--file",       type=str,            help="Phan tich file .txt hoac .pdf (co OCR fallback cho PDF scan)")
    parser.add_argument("--folder",     type=str,            help="Quet toan bo file .txt/.pdf trong thu muc")
    return parser.parse_args()


def main() -> None:
    cau_hinh_console_utf8()
    args      = doc_tham_so()
    loai      = args.model
    dung_split = not args.no_split

    # --- Chế độ đánh giá split đầy đủ ---
    if args.eval_split:
        chay_danh_gia_split(loai=loai, json_path=args.dataset, seed=args.seed)
        return

    # --- Chuan bi checkpoint cho API --- 
    if args.prepare_api:
        chuan_bi_checkpoint_api(loai=loai, json_path=args.dataset, seed=args.seed)
        return

    # --- Demo nhanh ---
    if args.demo:
        chay_demo(loai, dung_split=dung_split, json_path=args.dataset, seed=args.seed)
        return

    # --- Phân tích text trực tiếp ---
    if args.text:
        phan_tich_text(args.text, loai, dung_split=dung_split, json_path=args.dataset, seed=args.seed)
        return

    # --- Phân tích từ file ---
    if args.file:
        phan_tich_tai_lieu(
            args.file,
            loai,
            dung_split=dung_split,
            json_path=args.dataset,
            seed=args.seed,
        )
        return

    # --- Quét thư mục ---
    if args.folder:
        quet_thu_muc(
            args.folder,
            loai,
            dung_split=dung_split,
            json_path=args.dataset,
            seed=args.seed,
        )
        return

    # --- Không truyền tham số → in hướng dẫn ---
    print("Ban chua truyen tham so. Vi du:")
    print()
    print("  # Demo nhanh")
    print("  python main.py --demo")
    print("  python main.py --demo --model phobert")
    print("  python main.py --demo --model mfinbert")
    print()
    print("  # Danh gia baseline/PhoBERT/MFinBERT voi governance + train/val/test split day du")
    print("  python main.py --eval-split --model baseline")
    print("  python main.py --eval-split --model phobert")
    print("  python main.py --eval-split --model mfinbert")
    print("  python main.py --eval-split --model mfinbert --dataset samples.jsonl --seed 42")
    print()
    print("  # Chuan bi checkpoint cho FastAPI (inference-only)")
    print("  python main.py --prepare-api --model baseline")
    print("  python main.py --prepare-api --model phobert")
    print("  python main.py --prepare-api --model mfinbert")
    print()
    print("  # Phan tich text")
    print('  python main.py --text "Doanh thu khong tu hoa don gia." --model baseline')
    print('  python main.py --text "Doanh thu khong tu hoa don gia." --model phobert')
    print('  python main.py --text "Doanh thu khong tu hoa don gia." --model mfinbert')
    print()
    print("  # Phan tich file")
    print("  python main.py --file sample_report.txt --model baseline")
    print("  python main.py --file sample_report.txt --model phobert")
    print("  python main.py --file sample_report.txt --model mfinbert")
    print("  python main.py --file sample_report.pdf --model phobert")
    print()
    print("  # Quet thu muc (.txt + .pdf)")
    print('  python main.py --folder "/duong_dan/thu_muc" --model phobert')
    print('  python main.py --folder "/duong_dan/thu_muc" --model mfinbert')
    print()
    print("  # Tat split, dung toan bo data (khong khuyen dung)")
    print("  python main.py --demo --model phobert --no-split")
    print("  python main.py --demo --model mfinbert --no-split")


if __name__ == "__main__":
    main()
