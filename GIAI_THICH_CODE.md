# Tài liệu Giải thích Mã nguồn & Kỹ thuật - AuditBERT-VN

Tài liệu này cung cấp cái nhìn chi tiết về các thành phần kỹ thuật, kiến trúc mô hình, và quy trình xử lý dữ liệu của hệ thống AuditBERT-VN, bám sát triển khai thực tế trong mã nguồn.

## 1. Kiến trúc Mô hình Hybrid AuditBERT-VN

AuditBERT-VN được thiết kế theo hướng **Feature Fusion (Hợp nhất đặc trưng)**, kết hợp sức mạnh ngữ nghĩa của Transformer với các đặc trưng chuyên gia (domain knowledge) về tài chính và kiểm toán.

### 1.1. Cấu trúc hai giai đoạn (Two-stage Architecture)
Mô hình thực tế triển khai quy trình 2 giai đoạn tích hợp trong một checkpoint duy nhất:
- **Giai đoạn 1 (Backbone):** Sử dụng `PhoBERT` (`vinai/phobert-base`) để trích xuất xác suất rủi ro thô (`raw_text_prob`) từ nội dung văn bản. Đây là backbone tốt nhất cho tiếng Việt, giúp hiểu sâu ngữ cảnh kế toán.
- **Giai đoạn 2 (Hybrid Head):** Một lớp `Logistic Regression` đóng vai trò Meta-learner, nhận đầu vào là vector 21 chiều gồm:
    - 1 xác suất thô từ PhoBERT.
    - 20 đặc trưng metadata (thống kê, rule-based, tín hiệu tài chính).
- **Lập luận về Overfitting:** Mô hình Meta-learner được huấn luyện trên chính tập Train (không sử dụng out-of-fold). Tuy nhiên, rủi ro overfitting được kiểm soát chặt chẽ nhờ việc sử dụng mô hình tuyến tính đơn giản (Logistic Regression) với ít tham số, thay vì các bộ phân loại phi tuyến phức tạp. Điều này cho phép mô hình học được sự kết hợp tối ưu giữa tín hiệu ngôn ngữ và tín hiệu số liệu mà không bị "overfit" vào các nhiễu quá sâu.

### 1.2. Quy trình thực thi 3 giai đoạn (Three-stage Workflow)
Hệ thống vận hành qua 3 giai đoạn chính (Minh họa tại Hình 1):
1.  **Giai đoạn 1:** Tiền xử lý và trích xuất văn bản từ PDF (sử dụng PyMuPDF kết hợp các thuật toán làm sạch OCR rác).
2.  **Giai đoạn 2:** Suy luận qua backbone PhoBERT để lấy xác suất thô và đồng thời trích xuất 20 đặc trưng metadata chuyên biệt.
3.  **Giai đoạn 3:** Hợp nhất qua Hybrid Metadata Head (Meta-learner) để sinh xác suất cảnh báo rủi ro cuối cùng và áp dụng các quy tắc hậu xử lý (Rule-based post-processing).

### 1.3. Hàm mất mát Focal Loss
Để giải quyết vấn đề mất cân bằng dữ liệu nghiêm trọng (class imbalance), mô hình sử dụng **Focal Loss** thay cho Cross-Entropy thông thường.
Công thức triển khai trong `engine_transformer.py`:
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Trong đó:
- $p_t$ là xác suất dự đoán cho lớp đúng.
- $(1 - p_t)^\gamma$ là nhân tử điều chỉnh (Modulating Factor) với $\gamma = 1.0$ (mặc định), giúp giảm trọng số của các mẫu dễ và tập trung vào các mẫu khó.
- $\alpha_t$ được tính theo nghịch đảo tần suất lớp (**inverse class frequency**): $\alpha_{fraud} = n_{non\_fraud} / n$ và $\alpha_{non\_fraud} = n_{fraud} / n$. Điều này giúp mô hình không bị thiên kiến về phía các báo cáo "sạch" chiếm đa số.

## 2. Quy trình Dữ liệu và Chống Rò rỉ (Data Leakage)

### 2.1. Chiến lược phân chia dữ liệu
Hệ thống sử dụng phương pháp **Time-based split** kết hợp **Company-level isolation** để đảm bảo tính khách quan:
- **Phân chia theo thời gian:** Báo cáo được sắp xếp theo trình tự thời gian thực tế. Tập huấn luyện gồm dữ liệu từ 2020-2023, tập kiểm định (validation) năm 2024, và tập kiểm tra (test) năm 2025.
- **Cô lập doanh nghiệp:** Đảm bảo mỗi mã doanh nghiệp chỉ xuất hiện duy nhất trong một tập (Train, Val, hoặc Test). Điều này ngăn chặn mô hình "học thuộc lòng" đặc điểm cố định của một doanh nghiệp để dự đoán rủi ro.
- **Thống kê tập Test:** Tổng cộng **318 mẫu** kiểm tra độc lập (thống kê đồng nhất tại Bảng 3 và các phân tích lỗi). Tập test bao gồm các báo cáo từ cuối năm 2024 đến năm 2025 (unseen data).

### 2.2. Kiểm soát rò rỉ đặc trưng
Các đặc trưng metadata được thiết kế để tránh rò rỉ thông tin trực tiếp từ kết luận kiểm toán:
- Hệ thống thực hiện kiểm tra leakage bằng cách loại bỏ các đặc trưng red-flag lộ liễu (như ý kiến ngoại trừ trực tiếp) và huấn luyện lại Meta-learner.
- Kết quả AUPRC tuy giảm xuống mức 0.94 nhưng vẫn cao hơn hẳn các baseline, chứng tỏ mô hình không chỉ dựa vào tín hiệu lộ mà thực sự bắt được các dấu hiệu tiềm ẩn.

## 3. Phân tích Kết quả và Hiệu năng

### 3.1. Giải thích chỉ số AUC-ROC cao (≈ 0.999)
Mức hiệu năng gần như tuyệt đối phản ánh bản chất của tập dữ liệu nghiên cứu:
- **Bộ lọc rủi ro nhạy (Sensitive Filter):** AuditBERT-VN hoạt động như một công cụ sàng lọc rủi ro mức độ cao. Các mẫu rủi ro trong tập dữ liệu thường mang những dấu hiệu cảnh báo rõ rệt trong văn bản (ví dụ: các đoạn thuyết minh về nợ xấu, thay đổi chính sách kế toán bất thường).
- **Khả năng tổng quát:** Khi đánh giá trên các doanh nghiệp hoàn toàn mới (unseen firms), F1-score đạt 0.91, xác nhận mô hình có khả năng tổng quát hóa tốt thay vì chỉ ghi nhớ dữ liệu.

### 3.2. Nhãn và Thuật ngữ
Để đảm bảo tính chính xác về mặt pháp lý và học thuật, hệ thống thống nhất sử dụng thuật ngữ:
- **Nhãn cảnh báo:** Thay "Gian lận" (Fraud) bằng **"Cảnh báo rủi ro"** (Risk Warning).
- **Ý nghĩa:** Kết quả từ mô hình là tín hiệu rủi ro báo cáo tài chính, không được hiểu là kết luận pháp lý về hành vi gian lận của doanh nghiệp.

### 3.3. So sánh Bảng 3 và Bảng 4
- **Bảng 3:** Trình bày kết quả trên tập Kiểm tra độc lập (318 mẫu) - đây là hiệu năng thực tế cuối cùng.
- **Bảng 4 (Ablation Study):** Các chỉ số trong bảng này được thực hiện trên một **Development Split** riêng biệt (giữ lại 15% từ tập Train) để định lượng đóng góp của từng thành phần (như bỏ Focal Loss, bỏ Metadata, bỏ Backbone) mà không làm lộ tập Test chính thức. Do đó, các con số tuyệt đối có thể khác nhau, nhưng xu hướng tương đối là cơ sở cho các kết luận khoa học.

## 4. Threats to Validity (Các mối đe dọa đến tính hợp lệ)

1.  **Construct Validity (Tính hợp lệ nội dung):** Nhãn nghiên cứu dựa trên ý kiến kiểm toán và quyết định xử phạt, phản ánh rủi ro báo cáo thay vì ý định gian lận chủ quan. Mô hình học "tín hiệu rủi ro", không phải "hành vi tội phạm".
2.  **Internal Validity (Tính hợp lệ bên trong):** Mặc dù đã tách biệt doanh nghiệp, một số đặc trưng metadata (như `red_flag_count`) có thể chứa từ khóa tương đồng với nhãn. Chúng tôi đã giảm thiểu bằng Logistic Regression và kiểm tra leakage bổ trợ.
3.  **External Validity (Tính hợp lệ bên ngoài):** Dữ liệu tập trung vào các doanh nghiệp niêm yết tại Việt Nam. Khả năng áp dụng cho các doanh nghiệp chưa niêm yết hoặc ở các thị trường tài chính khác cần được nghiên cứu thêm.
4.  **Temporal Validity (Tính hợp lệ thời gian):** Khi các quy định về kế toán (VAS sang IFRS) hoặc quy định công bố thông tin thay đổi, mô hình cần được tái huấn luyện để tránh hiện tượng Concept Drift.

## 5. Các mô hình Baseline và So sánh

- **MFinBERT (ProsusAI/finBERT):** Đóng vai trò bài kiểm tra **Cross-lingual Transfer Stress Test**. Là mô hình tiếng Anh chuyên sâu về tài chính; việc áp dụng zero-shot lên tiếng Việt cho kết quả kém, minh chứng rào cản ngôn ngữ và sự cần thiết của backbone ngôn ngữ bản địa như PhoBERT.
- **Baseline (TF-IDF + RF):** Đóng vai trò mốc so sánh cơ bản, cho thấy các phương pháp truyền thống khó bắt được ngữ nghĩa phức tạp và các mối liên kết phi tuyến giữa các chỉ số tài chính.

---
*Tài liệu này được cập nhật định kỳ để phản ánh các thay đổi trong `engine_auditbert.py` và quy trình thực nghiệm.*
