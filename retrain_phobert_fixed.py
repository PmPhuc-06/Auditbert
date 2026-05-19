#!/usr/bin/env python
"""
retrain_phobert_fixed.py — Retrain PhoBERT với pipeline đã sửa triệt để.

NGUYÊN NHÂN COLLAPSE ĐÃ XÁC ĐỊNH VÀ SỬA:
  ❌ Cũ: Fine-tune TẤT CẢ 12 layers cùng lúc → Catastrophic Forgetting → AUC=0.5
  ✅ Mới: Đóng băng 8 layers đầu, chỉ fine-tune 4 layers cuối + classifier head
  ❌ Cũ: LR = 2e-5 đồng đều toàn bộ model → Gradient explosion ở backbone
  ✅ Mới: Backbone LR = 2e-5, Classifier head LR = 2e-4 (differential LR)
  ❌ Cũ: fit_from_split() dùng checkpoint cũ (AUC=0.5) mà không kiểm tra
  ✅ Mới: Tự động detect collapsed checkpoint (AUC < 0.6) → Xóa và retrain

PIPELINE MỚI:
  1. Temporal 80/10/10 split when timestamps are available (fallback: stratified split)
  2. Freeze 8/12 encoder layers (chỉ unfreeze 4 layers cuối + head)
  3. Differential LR: backbone=2e-5, head=2e-4
  4. Linear warm-up 6% steps + Linear decay
  5. Early stopping (patience=3) trên val F2
  6. Kiểm tra AUC > 0.70 — cảnh báo nếu chưa đạt

Cách dùng:
    python retrain_phobert_fixed.py --force-retrain
    python retrain_phobert_fixed.py --data fraud_samples.jsonl --force-retrain
    python retrain_phobert_fixed.py --lr 1e-5 --freeze-layers 6
    python retrain_phobert_fixed.py --freeze-layers 0   # fine-tune tất cả (cẩn thận!)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrain PhoBERT với layer freezing + differential LR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data",     default="samples.jsonl",
                   help="File JSONL dataset")
    p.add_argument("--epochs",   type=int,   default=10,
                   help="Số epochs tối đa (early stopping sẽ dừng sớm)")
    p.add_argument("--lr",       type=float, default=2e-5,
                   help="Learning rate backbone (head LR = lr * 10)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len",  type=int,   default=256)
    p.add_argument("--patience", type=int,   default=3)
    p.add_argument("--gamma",    type=float, default=1.5,
                   help="Focal Loss gamma (1.5 hơi mạnh hơn 1.0)")
    p.add_argument("--freeze-layers", type=int, default=8,
                   help="Số encoder layers đóng băng (0=fine-tune tất cả, 8=Mặc định)")
    p.add_argument("--force-retrain", action="store_true",
                   help="Xóa checkpoint cũ bắt buộc (collapsed checkpoint sẽ tự động bị xóa nếu AUC < 0.6)")
    p.add_argument("--checkpoint", default="phobert_fraud_checkpoint.pt")
    p.add_argument("--with-hybrid", action="store_true",
                   help="Bật hybrid metadata head (chỉ dùng cho ablation, không phải baseline PhoBERT thuần)")
    p.add_argument("--min-auc", type=float, default=0.60,
                   help="Ngưỡng AUC để xác định checkpoint bị collapse (mặc định: 0.60)")
    return p.parse_args()


def kiem_tra_dataset(data_path: Path) -> tuple[int, int]:
    if not data_path.exists():
        print(f"[LỖI] Không tìm thấy: {data_path}")
        sys.exit(1)
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    total = len(samples)
    fraud = sum(1 for s in samples if s.get("label") == 1)
    print(f"\n{'='*60}")
    print(f"  Dataset  : {data_path.name}")
    print(f"  Tổng mẫu : {total:,}")
    print(f"  Gian lận : {fraud:,} ({fraud/max(total,1)*100:.1f}%)")
    print(f"  Bình thường: {total-fraud:,} ({(total-fraud)/max(total,1)*100:.1f}%)")
    print(f"  Tỷ lệ imbalance: 1:{(total-fraud)/max(fraud,1):.1f}")
    print(f"{'='*60}\n")
    if total < 20:
        print("[CẢNH BÁO] Dataset rất nhỏ (<20 mẫu). Kết quả sẽ kém tin cậy.")
    return fraud, total - fraud


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    # ── Kiểm tra dataset ─────────────────────────────────────────────────
    n_fraud, n_non = kiem_tra_dataset(data_path)

    # ── Xóa checkpoint cũ nếu --force-retrain ───────────────────────────
    ckpt = Path(args.checkpoint)
    if args.force_retrain:
        for f in [ckpt, ckpt.with_suffix(".meta.json"), ckpt.with_suffix(".best.pt")]:
            if f.exists():
                f.unlink()
                print(f"[RESET] Đã xóa: {f.name}")
        print()
    elif ckpt.exists():
        print(
            f"[INFO] Checkpoint cũ tồn tại: {ckpt.name}\n"
            f"       → Tự động kiểm tra AUC. Nếu AUC < {args.min_auc}, sẽ tự xóa và retrain.\n"
            f"       → Dùng --force-retrain để ép xóa ngay.\n"
        )

    # ── Khởi tạo model ───────────────────────────────────────────────────
    print("[1/5] Khởi tạo PhoBERT với pipeline mới...")
    try:
        from engine_phobert import MoHinhGianLanPhoBERT
    except ImportError as e:
        print(f"[LỖI] {e}")
        sys.exit(1)

    model = MoHinhGianLanPhoBERT(
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_len=args.max_len,
        patience=args.patience,
        gamma=args.gamma,
        hybrid_metadata_enabled=args.with_hybrid,
        freeze_layers=args.freeze_layers,
    )

    print(f"\n  ✓ Backbone     : {model.model_name}")
    print(f"  ✓ Freeze layers: {args.freeze_layers}/12 (chỉ fine-tune layers {args.freeze_layers}–11 + head)")
    print(f"  ✓ Backbone LR  : {model.learning_rate:.2e}")
    print(f"  ✓ Head LR      : {model.learning_rate * 10:.2e} (10x backbone)")
    print(f"  ✓ Epochs max   : {model.epochs} (early stopping patience={model.patience})")
    print(f"  ✓ Batch size   : {model.batch_size}")
    print(f"  ✓ Max length   : {model.max_len} tokens")
    print(f"  ✓ Focal γ      : {model.gamma}")
    print(f"  ✓ Hybrid meta  : {model.hybrid_metadata_enabled}")
    print(f"  ✓ Device       : {model.device}\n")

    # ── Split dữ liệu ───────────────────────────────────────────────────
    print("[2/5] Load và tách dữ liệu (ưu tiên temporal split 80/10/10, seed=42)...")
    from engine_common import luu_manifest_split, tao_split_mac_dinh_tu_json
    split, split_manifest = tao_split_mac_dinh_tu_json(
        str(data_path),
        ty_le_train=0.80,
        ty_le_val=0.10,
        seed=42,
    )
    luu_manifest_split(split_manifest, "phobert_split_manifest.json")
    n_tr_f  = sum(split.train_labels)
    n_val_f = sum(split.val_labels)
    n_te_f  = sum(split.test_labels)
    print(f"  Train: {len(split.train_labels):,} ({n_tr_f} fraud)")
    print(f"  Val  : {len(split.val_labels):,} ({n_val_f} fraud)")
    print(f"  Test : {len(split.test_labels):,} ({n_te_f} fraud)\n")

    # ── Governance check ────────────────────────────────────────────────
    print("[3/5] Kiểm tra governance...")
    try:
        from engine_governance import kiem_tra_governance_dataset
        report = kiem_tra_governance_dataset(str(data_path))
        print(f"  ✓ Governance ready: {report.get('ready_for_training', '?')}")
    except Exception as e:
        print(f"  ⚠ Bỏ qua: {e}")

    # ── Retrain ─────────────────────────────────────────────────────────
    print(
        "\n[4/5] Bắt đầu retrain PhoBERT...\n"
        f"      Pipeline: freeze {args.freeze_layers} layers → warm-up 6% → "
        f"differential LR → early stopping\n"
    )
    try:
        metrics = model.fit_from_split(
            split,
            luu_split=True,
            ten_file_prefix="phobert_split",
            min_auc_threshold=args.min_auc,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Dừng sớm.")
        sys.exit(0)

    # ── Kết quả ─────────────────────────────────────────────────────────
    print(f"\n[5/5] Kết quả Test set:")
    print(f"{'='*55}")
    for k in ["precision", "recall", "f1", "f2", "auprc", "auc_roc", "mcc"]:
        v = metrics.get(k, float("nan"))
        print(f"  {k:>12}: {v:.4f}")
    print(f"{'='*55}")

    auc = metrics.get("auc_roc", 0.0)
    thr = getattr(model, "threshold", 0.5)
    print(f"\n  Threshold: {thr:.4f}")

    if auc >= 0.75:
        print(f"\n✅ PhoBERT PHỤC HỒI HOÀN TOÀN: AUC-ROC = {auc:.4f} (≥ 0.75) 🎉")
        print("   → Tiếp theo: python eval_test_giangvien.py")
    elif auc >= 0.70:
        print(f"\n✅ PhoBERT đã phục hồi đủ: AUC-ROC = {auc:.4f} (≥ 0.70)")
        print("   → Tiếp theo: python eval_test_giangvien.py")
    else:
        print(f"\n⚠️  PhoBERT vẫn yếu: AUC-ROC = {auc:.4f} (< 0.70)")
        print("   Khuyến nghị:")
        if args.freeze_layers > 4:
            print(f"   1. Giảm freeze_layers: --freeze-layers {args.freeze_layers - 2}")
        print("   2. Dùng dataset lớn hơn: --data fraud_samples.jsonl")
        print("   3. Giảm LR thêm: --lr 5e-6")
        print("   4. Tăng gamma: --gamma 2.0")

    # ── Lưu kết quả ─────────────────────────────────────────────────────
    result_path = Path("retrain_phobert_result.json")
    result_path.write_text(
        json.dumps({
            "model": "phobert",
            "data": str(data_path),
            "hyperparams": {
                "lr_backbone": args.lr,
                "lr_head": args.lr * 10,
                "freeze_layers": args.freeze_layers,
                "epochs_max": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
                "gamma": args.gamma,
            "hybrid_metadata_enabled": args.with_hybrid,
            "split_strategy": split_manifest.get("strategy", "unknown"),
            },
            "test_metrics": metrics,
            "threshold": thr,
            "auc_recovered": auc >= 0.70,
            "split_manifest": split_manifest,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✅ Kết quả → {result_path}")
    print(f"✅ Checkpoint → {model.checkpoint_path}")


if __name__ == "__main__":
    main()
