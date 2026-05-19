# Chi tiết về Mô hình AuditBERT-VN

Tài liệu này đi sâu vào chi tiết cấu trúc, thành phần và cách thức hoạt động của mô hình **AuditBERT-VN** — mô hình tối ưu và cũng là mô hình tinh hoa cuối cùng trong hệ thống cảnh báo rủi ro BCTC BaoOngThay.

---

## 1. AuditBERT-VN là gì?

**AuditBERT-VN** là một mô hình học máy áp dụng cơ chế **Feature Fusion** (Kết hợp đặc trưng) nhằm tổng hợp sức mạnh từ 3 cách tiếp cận khác nhau trong bài toán nhận diện rủi ro BCTC tiếng Việt:

1. **PhoBERT (Deep Learning):** Giải quyết bài toán phân tích ngữ nghĩa, hiểu cấu trúc câu và nhận diện các ngữ cảnh rủi ro phức tạp của tiếng Việt.
2. **Baseline TF-IDF (Machine Learning truyền thống):** Nắm bắt tần suất xuất hiện của các từ khóa về tài chính hay quy luật luật Benford (thông qua số liệu tròn chẵn).
3. **MFinBERT (Tài chính tiếng Anh):** Lấy ưu điểm vượt trội của mô hình chuyên ngành tài chính bằng cách dịch và nhận diện phân bổ từ vựng tài chính tiếng Anh như một đặc trưng (proxy feature).

**Điểm đặc biệt:** Thay vì phải chạy cả 3 mô hình song song làm tiêu tốn rất nhiều tài nguyên RAM của hệ thống, **AuditBERT-VN gộp các thành phần trên lại vào chung quá trình huấn luyện**. Khi dự đoán (inference), hệ thống chỉ cần gọi **1 checkpoint duy nhất** với dung lượng siêu nhẹ khoảng ~540MB, đẩy tốc độ xử lý nhanh hơn gấp nhiều lần.

---

## 2. Kiến trúc Feature Fusion

Cấu trúc của AuditBERT-VN gồm 2 phần chính dính liền và tương tác với nhau:
- **Backbone (Mạng cơ sở):** Kế thừa `vinai/phobert-base` – Mô hình Transformer huấn luyện tốt nhất cho ngôn ngữ tiếng Việt.
- **Hybrid Head (Đầu phân loại lai):** Bên cạnh Layer phân loại cuối cùng của PhoBERT, mô hình nối thêm một bộ lọc đặc trưng (Hybrid Metadata) bao gồm **20 features (đặc trưng)**.

### Chi tiết 20 Hybrid Metadata Features:

**14 Features cơ bản** (sử dụng chung cho các mô hình trước đây như PhoBERT / MFinBERT cũ):
1. **Thống kê văn bản (7 features):** `log_char_len`, `log_token_count`, `avg_token_len`, `ocr_quality`, `uppercase_ratio`, `digit_ratio`, `newline_density`.
2. **Rule-based (2 features):** `red_flag_count` (tổng số lượng cờ đỏ), `suspicious_segment_count` (số đoạn văn bản nghi ngờ).
3. **Tín hiệu rủi ro mở rộng (5 features):** `english_keyword_ratio`, `has_related_party` (bên liên quan), `has_invoice_risk` (rủi ro hóa đơn ảo), `has_cashflow_risk` (rủi ro dòng tiền âm/bất thường), `has_off_balance_risk` (rủi ro ngoại bảng).

**6 Features MỚI mở rộng RẤT QUAN TRỌNG (độc quyền của AuditBERT-VN):**
1. `financial_term_density_vn`: Mật độ từ khóa tài chính tiếng Việt (Proxy Feature thay thế hoàn toàn cho Baseline TF-IDF).
2. `financial_term_density_en`: Mật độ từ khóa tài chính tiếng Anh (Proxy Feature thay thế hoàn toàn cho MFinBERT).
3. `round_number_ratio`: Tỷ lệ số liệu được làm tròn (Dựa trên định luật **Benford's Law** — cốt lõi trong rà soát bất thường kế toán).
4. `accounting_code_count`: Tổng số lượng các mã tài khoản của Chuẩn mực Kế toán VN (VAS) (ví dụ: 111, 131, 511, 331...) dùng để đo lường tính chất sổ sách, hệ thống kế toán.
5. `loss_keyword_count`: Bộ đếm các từ khóa mang sắc thái tiêu cực hoặc rủi ro như "lỗ", "âm", "giảm", "thiếu hụt", "khoản phải thu".
6. `abnormal_profit_signal`: Tín hiệu phát hiện các biến động lợi nhuận quá mức, bất thường đột biến so với ngữ cảnh văn bản.

---

## 3. Luồng hoạt động của AuditBERT-VN cụ thể

### 3.1. Quá trình Huấn luyện (Training - Chạy qua lệnh `python train_auditbert.py`)
1. Dữ liệu từ file BCTC (`samples.jsonl`) đi qua bộ tách từ (Tokenizer của PhoBERT).
2. Mã nguồn gọi thư viện `engine_metadata.py` để trích xuất ngay lập tức 20 tính năng ở đoạn trên dựa theo văn bản (Metadata Features).
3. Cùng thời điểm, thuật toán `FocalLoss` giúp PhoBERT học cách phát hiện rủi ro dựa trên text thuần. Focal Loss vô cùng linh hoạt trong việc cân bằng Data Imbalance (Vì trên thực tế, BCTC rủi ro bao giờ cũng ít hơn BCTC lành mạnh).
4. Khâu quan trọng (Kết nối - Fusion): Cuối huấn luyện, hệ thống chạy hàm `_fit_hybrid_metadata_model` để Train thêm một lớp Logistic Regression bên trên dự đoán của cốt lõi bằng chính 20 features trên.
5. Toàn bộ trọng số PhoBERT và Logistic Regression đó đúc lại thành file **`auditbert_fraud_checkpoint.pt`**.

### 3.2. Quá trình Suy luận (Inference - Khi dùng thử qua API / `app.py`)
Khi Upload một file PDF BCTC chạy phân tích mô hình AuditBERT:
1. Dữ liệu được trích xuất text và truyền vào `MoHinhGianLanAuditBERT.predict(text)`.
2. Backbone tiến tới lấy xác suất rủi ro phần thô (Raw Probability).
3. Hàm `tao_hybrid_feature_vector()` được gọi để bóc tách 20 features của văn bản đó.
4. Quá trình chạy hàm `_apply_hybrid_vector()` tạo ra điểm kết hợp giữa Raw Score và Metadata Score.
5. Điểm số kết hợp được cộng cộng dồn thêm các trọng số của Red Flag (cờ đỏ). Cho ra **Xác suất cuối cùng** (Final Probability).
6. Mô hình chạy kỹ thuật AI giải thích (Integrated Gradients) để tiến hành highlight màu chỉ ra đâu là đoạn văn bản đóng góp làm tăng xác suất cảnh báo. Trả kết quả JSON về UI.

---

## 4. Giải đáp nhanh — Dùng để bảo vệ đồ án/demo giảng viên

Phòng khi giảng viên (hoặc các bên chuyên môn) đưa ra thắc mắc cốt tử: *"Tại sao nhóm bạn không dùng MFinBERT (chuyên về tài chính) hay Baseline luôn đi, mà phải chế ra AuditBERT-VN làm gì?"*

**Khuyến nghị trả lời bảo vệ chuyên sâu:**
> *"Thưa thầy/cô, Báo cáo tài chính Việt Nam chứa rất nhiều số liệu cấu trúc và tiếng Việt đặc thù. Nếu chỉ dùng Baseline thì các thuật toán mất đi khả năng nhận thức cấu trúc câu ngữ nghĩa. Nếu chỉ dùng PhoBERT thì tuy hiểu tiếng Việt nhưng mô hình lại kém trong ngôn ngữ chuyên ngành và đặc biệt không thể theo dõi quy luật độ chẵn lẻ của số liệu (Benford's Law).*
> *Do đó, giải pháp của hệ thống là **Kiến trúc Feature Fusion - AuditBERT-VN**. Bọn em lấy lõi PhoBERT làm trung tâm để hiểu tiếng Việt, sau đó bổ sung Đầu lai (**Hybrid Head**) chứa đặc trưng của MFinBERT và Baseline (Ví dụ: Từ khóa Anh, Việt, Tỷ lệ mã tài khoản, Benford Law). Nhờ vậy, rút gọn mô hình lại chỉ còn 540MB nhưng lại thông minh, bao quát vượt trội, chạy cực nhanh và có khả năng đưa ra cảnh báo rủi ro với căn cứ giải thích rõ ràng."*

