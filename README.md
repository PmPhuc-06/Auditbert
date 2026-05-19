# AuditBERT-VN

AuditBERT-VN là bộ mã nguồn phục vụ nghiên cứu sàng lọc cảnh báo rủi ro sai lệch trọng yếu trong báo cáo tài chính tiếng Việt. Dự án kế thừa logic cũ về tiền xử lý tiếng Việt, phát hiện red flags, TF-IDF baseline, PhoBERT/MFinBERT và Hybrid Metadata Head, nhưng đã được tổ chức lại theo chuẩn `AuditLLM_Dataset_PerModel.docx`: dữ liệu gốc được tách khỏi dữ liệu đã xử lý, split dùng chung được sinh một lần, sau đó từng mô hình/baseline nhận đúng format riêng.

Lưu ý học thuật quan trọng: mã nguồn và README dùng cách diễn giải "cảnh báo rủi ro" thay vì kết luận pháp lý "gian lận". AuditBERT-VN là kiến trúc phân loại lai dựa trên PhoBERT + metadata late fusion, không phải một pretrained language model mới.

## Điểm cập nhật chính

- Thêm pipeline `scripts/build_dataset_per_model.py` để build đầy đủ Module 1 và cập nhật Module 3 theo file chuẩn `AuditLLM_Dataset_PerModel.docx`.
- Sinh cây `dataset/` gồm `raw/`, `processed/`, `splits/`, `per_model/` và `stats/`.
- Module 1 có đủ proposed + 5 baseline: `bge-m3 + reranker + VinaLLaMA`, rule-based, TF-IDF+LR, BM25+GPT-4o, RAG no-rerank, GPT-4o direct.
- Module 3 có FraudLens format: PhoBERT input, 24 linguistic features, signal lexicon, Benford-only, features-only, PhoBERT-only, GPT-4o, FinBERT EN.
- Sửa `.gitignore` để loại checkpoint/log/temp/split cũ khỏi mã nguồn.
- Điều chỉnh wording trong registry/model để đầu ra ưu tiên "Cảnh báo rủi ro" và giữ `legacy_label` cho tương thích ngược.

## Cài đặt

Yêu cầu Python 3.10+.

```powershell
cd "C:\Users\pmphu\OneDrive - ut.edu.vn\Artical\BaoOngThay"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-web.txt
```

Nếu chỉ build dataset per-model, các thư viện chính cần có là `scikit-learn`, `joblib`, `numpy`, `pandas`, `torch`, `transformers`, `underthesea`.

## Cấu trúc thư mục

```text
BaoOngThay/
  app.py                         # FastAPI demo/predict
  main.py                        # CLI train/eval/predict/scan folder
  train_auditbert.py             # Train AuditBERT-VN
  eval_test_giangvien.py         # Eval các model và xuất bảng cho bài báo
  engine_common.py               # Preprocess, split, metrics, red flags
  engine_metadata.py             # 20 metadata features + schema
  engine_transformer.py          # PhoBERT/MFinBERT/AuditBERT core
  engine_auditbert.py            # AuditBERT-VN wrapper
  engine_baseline.py             # TF-IDF + Logistic Regression
  engine_registry.py             # Registry model và format output
  scripts/
    build_dataset_per_model.py   # Build M1/M3 theo chuẩn AuditLLM
  dataset/
    processed/M1_canonical.jsonl
    splits/M1_train.jsonl
    splits/M1_dev.jsonl
    splits/M1_test.jsonl
    splits/M3_train.jsonl
    splits/M3_dev.jsonl
    splits/M3_test.jsonl
    per_model/M1/proposed/
    per_model/M1/B1_rule_based/
    per_model/M1/B2_tfidf_lr/
    per_model/M1/B3_bm25_gpt4/
    per_model/M1/B4_rag_norerank/
    per_model/M1/B5_gpt4_direct/
    per_model/M3/proposed/
    per_model/M3/B1_benford/
    per_model/M3/B2_features_xgb/
    per_model/M3/B3_phobert_only/
    per_model/M3/B4_gpt4/
    per_model/M3/B5_finbert/
    stats/
  tests/
```

## Build dataset

Theo file chuẩn, mỗi mô hình phải có format riêng và test set phải dùng chung.

```powershell
python scripts\build_dataset_per_model.py --input samples.jsonl --output-dir dataset --seed 42 --version v1.0
```

Mặc định script dùng tỷ lệ 70/15/15 cho Module 1 theo hướng dẫn `AuditVN-500`. Nếu cần tái lập đúng tỷ lệ bài AuditBERT-VN v5 trên 3.203 mẫu, dùng:

```powershell
python scripts\build_dataset_per_model.py --input samples.jsonl --output-dir dataset --seed 42 --version v1.0 --train-ratio 0.80 --dev-ratio 0.10
```

## Train, Eval, Predict

Train AuditBERT-VN:

```powershell
python train_auditbert.py --data samples.jsonl --eval
```

Eval split đầy đủ qua CLI:

```powershell
python main.py --eval-split --model auditbert --dataset samples.jsonl --seed 42
```

Predict một đoạn văn:

```powershell
python main.py --text "Doanh nghiệp có giao dịch bên liên quan lớn và dòng tiền âm kéo dài." --model auditbert
```

Chạy API demo:

```powershell
uvicorn app:app --reload
```

Mở Swagger tại `http://127.0.0.1:8000/docs`.

## Thông tin mô hình theo bài v5

- Backbone: `vinai/phobert-base`.
- Bài toán: phân loại nhị phân rủi ro sai lệch trọng yếu, không kết luận gian lận pháp lý.
- Dữ liệu bài v5: 3.203 mẫu, chia khoảng 80% train, 10% validation, 10% test.
- Test set bài v5: 318 mẫu, ma trận nhầm lẫn 207 TN, 3 FP, 2 FN, 106 TP.
- Max sequence length: 256 token.
- Văn bản dài: recursive chunking, tối đa khoảng 768 ký tự mỗi chunk, overlap 120 ký tự.
- Tổng hợp cấp tài liệu: `0.6 * max(chunk_score) + 0.4 * weighted_mean(chunk_score)`.
- Hybrid vector: 21 chiều gồm raw PhoBERT probability + 20 metadata features.
- Metadata head: Logistic Regression sau chuẩn hóa Z-score.
- Loss: Focal Loss `FL(p_t) = -alpha_t(1 - p_t)^gamma log(p_t)`, code hiện dùng `gamma=1.0`.
- Optimizer: AdamW, batch size 8, learning rate AuditBERT `1e-5`, tối đa 6 epochs, early stopping patience 3 theo F2.
- Diễn giải kết quả: AuditBERT-VN cân bằng tốt trên F1, AUPRC, AUC-ROC và MCC; PhoBERT có thể cao hơn ở Recall/F2; Baseline có thể cao nhất Precision.

## Góp ý đã phản ánh vào code/cấu trúc

- Đổi trọng tâm output sang cảnh báo rủi ro và giữ nhãn cũ chỉ ở `legacy_label`.
- Tách rõ late fusion/meta-classifier, tránh mô tả là pretrained model mới.
- Dataset builder ghi seed, version, policy `test_set_shared`, `tfidf_fit_train_only`, `augmentation_train_only`.
- TF-IDF vectorizer chỉ fit trên train rồi transform dev/test.
- `round_number_ratio` được mô tả là heuristic số tròn, không gọi là kiểm định Benford đầy đủ.
- Module 3 bổ sung Benford-only đúng nghĩa ở baseline riêng, tách khỏi metadata heuristic.

## Danh sách file/thư mục cũ cần xóa

Các nhóm đã được đưa vào danh sách cleanup vì là checkpoint, log, temp, split sinh lại hoặc artefact cũ:

- Checkpoints: `*_fraud_checkpoint.pt`, `*_fraud_checkpoint.best.pt`, `*_fraud_checkpoint.meta.json`, `baseline_fraud_checkpoint.json`, `baseline_fraud_checkpoint.pkl`.
- Split/output cũ ở root: `baseline_split_*`, `phobert_split_*`, `mfinbert_split_*`, `auditbert_split_*`, `split_eval_result_*.json`, `prepare_api_result_*.json`.
- Dataset trung gian lớn: `all_temp.jsonl`, `all_balanced.jsonl`, `dataset_merged.jsonl`, `fraud_samples.jsonl`, `samples_final.jsonl`, `nonfraud_samples.jsonl`.
- Logs/result tạm: `import_error.txt`, `startup_smoke.log`, `train_log.txt`, `drift_log.json`, `ket_qua_eval.json`, `ablation_results.json`, `mcnemar_results.json`.
- Cache/env/temp: `__pycache__/`, `tests/__pycache__/`, `tests/tmp*/`, `poppler_*/`, `.venv/`, `venv310/`.

Giữ lại `samples.jsonl`, `Tap_Du_Lieu_Test_Kiem_Tra.csv`, code engine, tests, và cây `dataset/` mới.
