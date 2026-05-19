#!/usr/bin/env python
"""
train_auditbert.py — Script huấn luyện AuditBERT-VN.

Mô hình tổng hợp cuối cùng: PhoBERT backbone + Feature Fusion 20 chiều.
Chỉ cần 1 lệnh để train và lưu checkpoint.

Cách dùng:
    python train_auditbert.py                          # Train từ samples.jsonl
    python train_auditbert.py --data my_data.jsonl    # Train từ file tùy chọn
    python train_auditbert.py --eval                  # Train + đánh giá trên test set
    python train_auditbert.py --quick                 # Train nhanh từ mini_samples (demo)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình AuditBERT-VN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="samples.jsonl",
        help="Đường dẫn đến file dataset JSONL (mặc định: samples.jsonl)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Đánh giá mô hình trên test set sau khi train",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Dùng mini_samples.jsonl để train demo nhanh trên CPU",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Đường dẫn lưu checkpoint (mặc định: auditbert_fraud_checkpoint.pt)",
    )
    return parser.parse_args()


def kiem_tra_dataset(data_path: Path) -> tuple[int, int]:
    """Kiểm tra dataset và in thống kê nhanh."""
    if not data_path.exists():
        print(f"[LỖI] Không tìm thấy file: {data_path}")
        print("Hãy tạo samples.jsonl trước bằng cách chạy engine_parser.py")
        sys.exit(1)

    samples = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(samples)
    fraud = sum(1 for s in samples if s.get("label") == 1)
    non_fraud = total - fraud

    print(f"\n{'='*60}")
    print(f"  Dataset: {data_path.name}")
    print(f"  Tổng mẫu  : {total:,}")
    print(f"  Gian lận  : {fraud:,} ({fraud/max(total,1)*100:.1f}%)")
    print(f"  Bình thường: {non_fraud:,} ({non_fraud/max(total,1)*100:.1f}%)")
    print(f"{'='*60}\n")

    if total < 10:
        print("[CẢNH BÁO] Dataset rất nhỏ — kết quả chỉ phù hợp để demo.")
    return fraud, non_fraud


def main() -> None:
    args = parse_args()

    # ── Chọn đường dẫn dataset ──────────────────────────────────────────────
    if args.quick:
        data_path = Path("tests/fixtures/mini_samples.jsonl")
        print("[QUICK MODE] Dùng mini_samples.jsonl để demo nhanh trên CPU.")
    else:
        data_path = Path(args.data)

    kiem_tra_dataset(data_path)

    # ── Import model ─────────────────────────────────────────────────────────
    print("[1/4] Khởi tạo AuditBERT-VN...")
    try:
        from engine_auditbert import MoHinhGianLanAuditBERT
    except ImportError as e:
        print(f"[LỖI] Không load được engine_auditbert.py: {e}")
        print("Hãy chắc chắn đã cài: pip install torch transformers sentencepiece")
        sys.exit(1)

    model_kwargs = {}
    if args.checkpoint:
        model_kwargs["checkpoint_path"] = args.checkpoint
    if args.quick:
        # Giảm epochs cho demo nhanh
        model_kwargs["epochs"] = 2
        model_kwargs["patience"] = 1

    model = MoHinhGianLanAuditBERT(**model_kwargs)
    print(f"  ✓ Backbone   : {model.model_name}")
    print(f"  ✓ Checkpoint : {model.checkpoint_path}")
    print(f"  ✓ Max length : {model.max_len} tokens")
    print(f"  ✓ Epochs     : {model.epochs}")
    print(f"  ✓ Device     : {model.device}\n")

    # ── Đánh giá governance ──────────────────────────────────────────────────
    print("[2/4] Kiểm tra governance dataset...")
    try:
        from engine_governance import kiem_tra_governance_dataset
        report = kiem_tra_governance_dataset(str(data_path))
        ready = report.get("ready_for_training", False)
        print(f"  ✓ Governance ready: {ready}")
        if not ready and not args.quick:
            warnings = report.get("warnings", [])
            for w in warnings[:5]:
                print(f"  ⚠ {w}")
    except Exception as e:
        print(f"  ⚠ Bỏ qua governance check: {e}")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[3/4] Bắt đầu huấn luyện AuditBERT-VN...")
    print("  (Nếu đã có checkpoint, bước này sẽ load lại và bỏ qua fine-tune backbone)\n")
    try:
        from engine_common import luu_manifest_split, tao_split_mac_dinh_tu_json
        split, split_manifest = tao_split_mac_dinh_tu_json(
            str(data_path),
            ty_le_train=0.8,
            ty_le_val=0.1,
            seed=42,
        )
        luu_manifest_split(split_manifest, "auditbert_split_manifest.json")
        metrics = model.fit_from_split(split, ten_file_prefix="auditbert_split")
        if args.eval:
            print(f"\n  Kết quả Test set:")
            for k, v in sorted(metrics.items()):
                print(f"    {k:>15}: {v:.4f}")
    except KeyboardInterrupt:
        print("\n[INFO] Huấn luyện bị dừng bởi người dùng.")
        sys.exit(0)
    except Exception as e:
        print(f"[LỖI] Huấn luyện thất bại: {e}")
        raise

    # ── Kết quả ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Hoàn thành!")
    print(f"  ✓ Checkpoint lưu tại: {model.checkpoint_path}")
    print(f"  ✓ Threshold tối ưu  : {model.threshold:.3f}")
    print(f"  ✓ Model source      : {model.model_source}")

    print("\n" + "="*60)
    print("  AuditBERT-VN đã sẵn sàng dùng qua FastAPI!")
    print("  Chạy server: uvicorn app:app --reload")
    print("  Swagger UI : http://127.0.0.1:8000/docs")
    print("  Chọn model : auditbert (⭐ AuditBERT-VN - Final Model)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
