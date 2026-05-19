# Hướng Dẫn Dataset Thật — BaoOngThay

Tài liệu này giải thích **toàn bộ quy trình xây dựng dataset** từ Báo cáo Tài chính (BCTC) thật,
đảm bảo tính liêm chính học thuật: **không AI sinh dữ liệu, không câu giả, không số liệu bịa đặt**.

---

## 1. Tại sao phải dùng dữ liệu thật?

### 1.1. Vấn đề với Synthetic Data (AI sinh dữ liệu)

Trong các hệ thống phát hiện gian lận tài chính, việc dùng AI (ChatGPT, Claude...) để sinh câu mẫu gây ra các hậu quả nghiêm trọng:

| Vấn đề | Hậu quả |
|---|---|
| AI không biết số liệu tài chính nào là "gian lận thật" | Model học pattern giả → False positive cao |
| AI sinh câu văn phong "sách giáo khoa" | Không khớp văn phong BCTC thực tế (viết tắt, bảng số, mã khoản) |
| Số liệu sinh ra không tuân theo phân phối Benford's Law | Benford feature vô nghĩa |
| Không có mã khoản kế toán VAS thực tế (111, 131, 511...) | accounting_code_count feature sai |
| Vi phạm nguyên tắc trung thực trong nghiên cứu | Không đủ chuẩn nộp báo cáo học thuật |

### 1.2. Tiêu chuẩn của dự án này

> **Nguyên tắc cốt lõi**: Mọi mẫu trong `samples.jsonl` phải trích xuất từ tài liệu BCTC thật,
> được công bố công khai bởi cơ quan có thẩm quyền, không qua bất kỳ AI generation nào.

---

## 2. Nguồn dữ liệu được phép dùng

### 2.1. Nguồn chính thống — Tải PDF BCTC gốc

| Nguồn | URL trực tiếp | Nội dung | Nhãn phù hợp |
|---|---|---|---|
| **UBCKNN / SSC** — Báo cáo & Công bố | [`congbothongtin.ssc.gov.vn`](https://congbothongtin.ssc.gov.vn/) | BCTC kiểm toán năm, quý toàn bộ công ty đại chúng | `label = 0` (BCTC bình thường) |
| **UBCKNN / SSC** — Xử phạt | [`ssc.gov.vn`](https://www.ssc.gov.vn) → mục *Xử phạt vi phạm hành chính* | Quyết định XPVPHC có tên công ty, mã vi phạm, năm | `label = 1` ✅ |
| **Công bố thông tin SSC** | [`congbothongtin.ssc.gov.vn`](http://congbothongtin.ssc.gov.vn) | Toàn bộ hồ sơ công bố của công ty đại chúng | Cả 2 nhãn |
| **Kiểm toán Nhà nước** | [`sav.gov.vn/pages/bao-cao-kiem-toan`](https://sav.gov.vn) → Báo cáo kiểm toán | Báo cáo kiểm toán ngân sách / doanh nghiệp Nhà nước | `label = 1` nếu có kiến nghị |
| **Bộ Tài chính** | [`mof.gov.vn`](https://www.mof.gov.vn) → Kế toán-Kiểm toán | Chuẩn mực VAS, thông tư kế toán | `label = 0` |

### 2.2. Công cụ hỗ trợ tải hàng loạt (khuyến nghị cho nghiên cứu)

| Công cụ | URL | Hình thức | Chi phí |
|---|---|---|---|
| **vnstock** (Python) | [`github.com/thinh-vu/vnstock`](https://github.com/thinh-vu/vnstock) | Thư viện Python, tải dữ liệu tài chính công chúng tự động | **Miễn phí** |
| **Vietstock Finance** | [`finance.vietstock.vn`](https://finance.vietstock.vn) | Export Excel BCTC theo công ty/ngành | Một phần miễn phí |
| **Vietdata** | [`vietdata.vn`](https://vietdata.vn) | Dashboard BCTC từ 2007, export CSV | Có phí |
| **WiFeed/WiGroup** | [`wifeed.vn`](https://wifeed.vn) | API tài chính doanh nghiệp | Có phí / API key |
| **Kreston Data** | [`data.kreston.vn`](https://data.kreston.vn) | Chuyên sâu ý kiến kiểm toán | Nghiên cứu |
| **CafeF** | [`cafef.vn/chung-khoan/xu-phat`](https://cafef.vn) | Tổng hợp quyết định xử phạt SSC theo năm | Miễn phí |

**Cách dùng vnstock để tải BCTC tự động (không cần PDF):**
```python
pip install vnstock

from vnstock import stock_financial_report

# Tải BCTC của VNM (Vinamilk) — Kết quả kinh doanh
df = stock_financial_report(symbol="VNM", report_type="incomestatement",
                             report_range="annual")
print(df)

# Tải bảng cân đối kế toán
df_bs = stock_financial_report(symbol="FLC", report_type="balancesheet",
                                report_range="annual")
```

> ⚠️ **Lưu ý**: vnstock trả dữ liệu dạng số (float) — cần tự chuyển thành văn bản
> mô tả để dùng với mô hình NLP. Cách chuyển xem phần 7 bên dưới.

### 2.3. Các vụ gian lận BCTC có hồ sơ công khai — Nhãn fraud = 1

| Công ty | Mã CK | Năm vi phạm | Loại vi phạm | Quyết định xử phạt |
|---|---|---|---|---|
| FLC Faros | ROS | 2021–2022 | Khai khống vốn góp, thao túng thị trường | SSC QĐ số 574/QĐ-XPVPHC |
| Louis Holdings | TGG | 2021 | Gian lận BCTC, thao túng cổ phiếu | SSC công bố 2022 |
| Apec Group | APC | 2021 | Thao túng giá, thông tin sai lệch | SSC công bố 2022 |
| Trung Nam | | 2022 | Sai phạm công bố thông tin BCTC | SSC QĐ xử phạt |
| Nhiều CT bị hủy niêm yết | Nhiều | Hàng năm | Lỗ lũy kế ≥ 50% vốn hoặc âm vốn chủ | Cập nhật truy xuất từ SSC |

**Tra toàn bộ danh sách xử phạt:** `ssc.gov.vn` → Tin tức → Xử phạt vi phạm hành chính
hoặc tìm trên CafeF: `cafef.vn` → Tìm kiếm "quyết định xử phạt [năm]"

---

## 3. Quy trình xây dựng Dataset từng bước

### Bước 1 — Thu thập PDF BCTC

```
Đến: https://congbothongtin.ssc.gov.vn/ → Nhập tên công ty/Mã Ck → Lọc "Báo cáo tài chính" → Tải PDF gốc
Đến: ssc.gov.vn → Văn bản pháp luật → Quyết định xử phạt → Tải PDF
```

**Tiêu chí chọn file:**
- File PDF phải có **Text Layer** (không phải ảnh scan) — kiểm tra bằng cách bôi đen text trong PDF reader
- Ưu tiên file từ **kiểm toán viên độc lập** (Big4, VACPA member)
- Mỗi công ty nên lấy ít nhất **2-3 năm** để so sánh xu hướng

**Cấu trúc thư mục khuyến nghị:**
```
data_raw/
├── non_fraud/
│   ├── VNM_BCTC_2022_kiem_toan.pdf   ← Vinamilk, không bị phạt
│   ├── FPT_BCTC_2023_kiem_toan.pdf
│   └── ...
└── fraud/
    ├── FLC_BCTC_2021_sau_xu_phat.pdf  ← Có quyết định xử phạt
    ├── SSC_quyet_dinh_xuphat_xxx.pdf
    └── ...
```

---

### Bước 2 — Kiểm tra PDF có Text Layer không

**Tự động bằng `engine_document_io.py`:**
```python
from engine_document_io import doc_tai_lieu_tu_duong_dan

text = doc_tai_lieu_tu_duong_dan("VNM_BCTC_2022.pdf")
print(len(text))  # Nếu < 200 ký tự → PDF scan → Bỏ qua
```

Hệ thống tự động **từ chối** (HTTP 400) nếu PDF là ảnh scan:
```python
# Trong engine_document_io.py
if len(text.strip()) < 20:
    raise LoiTrichXuatTaiLieu(
        "PDF là ảnh scan — không hỗ trợ OCR (nguyên tắc No-AI)",
        status_code=400
    )
```

---

### Bước 3 — Trích xuất văn bản bằng PyMuPDF

**`engine_parser.py` — Hàm `read_pdf_text_layer()`:**
```python
import fitz  # PyMuPDF

with fitz.open(path) as doc:
    for page in doc:
        # Giữ nguyên layout bảng số
        text = page.get_text("layout")
        # Tự động phát hiện bảng (PyMuPDF >= 1.23)
        tables = page.find_tables()
```

Kết quả: văn bản thô giữ nguyên cấu trúc bảng số, không bị AI "dịch" hay "tóm tắt".

---

### Bước 4 — Làm sạch nhiễu OCR nhẹ

Ngay cả PDF Text Layer đôi khi có nhiễu nhẹ từ phần mềm tạo PDF.
**`engine_parser.py` — Hàm `remove_ocr_noise()`:**

```python
def remove_ocr_noise(text: str) -> str:
    # Xóa ký tự điều khiển (giữ newline)
    text = re.sub(r"[^\S\n]+", " ", text)
    # Xóa gạch ngang kéo dài (vỡ bảng): "___________"
    text = re.sub(r"[_\-\.]{5,}", " ", text)
    # Xóa header/footer lặp lại: "- 5 -", "Trang 1/2"
    text = re.sub(r"^\s*(?:[-–]?\s*\d+\s*[-–]?|Trang\s*\d+)\s*$",
                  "", text, flags=re.MULTILINE)
    return text.strip()
```

---

### Bước 5 — Phát hiện và trích xuất sections BCTC

**`engine_parser.py` — Hàm `detect_sections()`:**

```python
SECTION_PATTERNS = {
    "income_statement": [
        r"k[eế]t qu[aả]\s*ho[aạ]t\s*[dđ][oô]ng",   # Kết quả hoạt động
        r"b[aá]o c[aá]o\s*k[eế]t qu[aả]",
        r"doanh thu",
    ],
    "balance_sheet": [
        r"b[aả]ng\s*c[aâ]n\s*[dđ][oô]i",            # Bảng CĐKT
        r"t[aà]i\s*s[aả]n",
    ],
    "cash_flow": [
        r"l[uư]u\s*chuy[eể]n\s*ti[eề]n",             # Lưu chuyển tiền tệ
        r"cash\s*flow",
    ],
}
```

Mỗi section được trích xuất riêng → ghép thành 1 mẫu văn bản hoàn chỉnh.

---

### Bước 6 — Gán nhãn (Labeling)

**Đây là bước quan trọng nhất — quyết định chất lượng dataset.**

#### Nguyên tắc gán nhãn

| Nhãn | Điều kiện | Bằng chứng cần có |
|---|---|---|
| `label = 1` (Gian lận) | Có quyết định xử phạt từ UBCKNN/Bộ TC **hoặc** kiểm toán viên từ chối/có ý kiến ngoại trừ | File PDF quyết định xử phạt |
| `label = 1` (Gian lận) | Công ty bị huỷ niêm yết do lỗ lũy kế ≥ 50% vốn | Thông báo từ Cổng thông tin SSC |
| `label = 0` (Bình thường) | Kiểm toán viên đưa ra ý kiến chấp nhận toàn phần (Unqualified) | Báo cáo kiểm toán không có ngoại trừ |

#### Quy trình gán nhãn thực tế

```
Bước 6a: Tải quyết định xử phạt từ ssc.gov.vn
           ↓
Bước 6b: Đối chiếu tên công ty với danh sách BCTC đã thu thập
           ↓
Bước 6c: Đánh dấu "fraud = 1" nếu company_name khớp + năm vi phạm khớp
           ↓
Bước 6d: BCTC không có trong danh sách xử phạt → mặc định "label = 0"
           ↓
Bước 6e: Review lần 2 bằng rule-based engine_parser.py:
         - Nếu dòng tiền âm + lỗ lũy kế + âm vốn chủ → nâng label = 1
```

---

### Bước 7 — Sinh file `samples.jsonl`

**`engine_parser.py` — Hàm `process_file()` và `run_pipeline()`:**

```python
# Xử lý 1 file
result = process_file("VNM_BCTC_2022.pdf", label=0)

# Kết quả là dict chuẩn:
{
    "text": "Doanh thu thuần 2022: 59.636 tỷ đồng...\nTổng tài sản: 38.992 tỷ...",
    "label": 0,
    "source": "pdf_text_layer",
    "doc_id": "VNM_BCTC_2022_kiem_toan",
    "fraud_signals": [],
    "fields": {
        "doanh_thu": 59636000000000,
        "loi_nhuan_truoc_thue": 9234000000000
    }
}
```

**Xử lý toàn bộ thư mục:**
```python
from engine_parser import run_pipeline

run_pipeline(
    input_dir="data_raw/",      # Thư mục chứa PDF
    output_path="samples.jsonl" # File output
)
```

Format JSONL (mỗi dòng là 1 mẫu độc lập):
```jsonl
{"text": "Doanh thu thuần...", "label": 0, "doc_id": "VNM_2022", "source": "pdf_text_layer"}
{"text": "Lỗ lũy kế âm vốn chủ sở hữu...", "label": 1, "doc_id": "FLC_2021", "source": "pdf_text_layer"}
```

---

## 4. Kiểm soát chất lượng Dataset

### 4.1 Governance check tự động

```python
from engine_governance import kiem_tra_governance_dataset

report = kiem_tra_governance_dataset("samples.jsonl")
print(report)
# {
#   "total_samples": 850,
#   "fraud_ratio": 0.23,         # 23% gian lận — chấp nhận được
#   "duplicate_rate": 0.002,      # 0.2% trùng lặp — tốt
#   "source_coverage": 0.98,      # 98% có trường source
#   "ready_for_training": True,
#   "warnings": []
# }
```

### 4.2 Kiểm tra thủ công (Manual QA)

Lấy mẫu ngẫu nhiên 5% và kiểm tra:

```python
import json, random

samples = [json.loads(l) for l in open("samples.jsonl")]
random_samples = random.sample(samples, k=int(len(samples)*0.05))

for s in random_samples:
    print(f"Label: {s['label']} | Source: {s['doc_id']}")
    print(s['text'][:200])
    print("---")
```

Checklist kiểm tra thủ công:
- [ ] Text có phải tiếng Việt thật không?
- [ ] Có số liệu thực (tỷ đồng, %) không?
- [ ] Nhãn có khớp với thực tế công ty không?
- [ ] Text không có dấu hiệu AI sinh (quá hoàn chỉnh, không có lỗi ngữ pháp nhỏ)?

### 4.3 Thống kê nhanh để báo cáo

```python
import json
from collections import Counter

samples = [json.loads(l) for l in open("samples.jsonl")]
labels = [s["label"] for s in samples]
sources = [s.get("source", "unknown") for s in samples]

print(f"Tổng mẫu: {len(samples)}")
print(f"Phân bố nhãn: {Counter(labels)}")
print(f"Nguồn dữ liệu: {Counter(sources)}")
print(f"Tỷ lệ gian lận: {sum(labels)/len(labels)*100:.1f}%")
```

---

## 5. Cấu trúc `samples.jsonl` hiện tại

Kiểm tra dataset thực tế của dự án:

```bash
# Đếm số mẫu
wc -l samples.jsonl

# Xem 5 mẫu đầu
head -5 samples.jsonl | python -m json.tool

# Phân bố nhãn
python -c "
import json
s = [json.loads(l) for l in open('samples.jsonl')]
f = sum(1 for x in s if x['label']==1)
print(f'Tổng: {len(s)}, Fraud: {f} ({f/len(s)*100:.1f}%), Non-fraud: {len(s)-f}')
"
```

---

## 6. Phân tách Train / Validation / Test

Hệ thống tự phân tách khi train — **không cần làm thủ công**:

```python
# Trong engine_common.py - tach_du_lieu_train_val_test()
# Mặc định: 80% train / 10% val / 10% test
# Stratified split: giữ tỷ lệ fraud/non-fraud đồng đều ở cả 3 tập
```

Kết quả phân tách lưu tại:
```
baseline_split_train.jsonl    # 80% mẫu
baseline_split_val.jsonl      # 10% mẫu
baseline_split_test.jsonl     # 10% mẫu
```

---

## 7. Luồng dữ liệu trực tiếp trong code

```mermaid
flowchart TD
    A["PDF BCTC thật (Cổng SSC)"] --> B{"Tách Text\nPyMuPDF"}
```

```python
engine_document_io.py
  └─ fitz.open(pdf) → page.get_text("layout")   ← PyMuPDF, không OCR
  └─ Từ chối nếu text < 20 ký tự (PDF scan)
       │
       ▼
engine_parser.py
  └─ remove_ocr_noise()          ← Làm sạch nhiễu nhẹ
  └─ detect_sections()           ← Tách Income/Balance/CashFlow
  └─ extract_financial_fields()  ← Regex lấy Doanh thu, Lợi nhuận, Tổng TS...
  └─ detect_fraud_signals()      ← Rule-based: lỗ lũy kế, âm vốn chủ...
       │
       ▼
samples.jsonl
  └─ {"text": "...", "label": 0/1, "source": "pdf_text_layer", "doc_id": "..."}
       │
       ▼
train_auditbert.py
  └─ kiem_tra_governance_dataset()   ← Kiểm tra chất lượng
  └─ tach_du_lieu_train_val_test()   ← 80/10/10 stratified split
  └─ MoHinhGianLanAuditBERT.fit()   ← Train AuditBERT-VN
       │
       ▼
auditbert_fraud_checkpoint.pt
```

---

## 8. Tuyên bố về tính liêm chính dữ liệu

**Để đưa vào báo cáo học thuật:**

> **Liêm chính Dữ liệu:** Toàn bộ BCTC của dự án đều là dữ liệu thực tế
> công khai từ Cổng Thông tin Điện tử Ủy ban Chứng khoán Nhà nước (https://congbothongtin.ssc.gov.vn/).
> Chúng tôi tuyệt đối không sử dụng công cụ AI (ChatGPT, Claude) để sinh tự động giai đoạn 2018–2024. Nhãn gian lận (label=1)
> được gán dựa trên quyết định xử phạt hành chính công khai của UBCKNN và thông báo hủy niêm
> yết chính thức. Toàn bộ văn bản được trích xuất bằng PyMuPDF từ lớp Text Layer vật lý của file
> PDF gốc — không qua bất kỳ công nghệ OCR hay mô hình AI sinh nào. Quy trình gán nhãn và
> trích xuất được kiểm tra tự động bằng engine_governance.py theo các tiêu chí: tỷ lệ trùng
> lặp, coverage trường metadata, cân bằng class và Cohen's Kappa.

---

## 9. Những gì KHÔNG được làm

| ❌ KHÔNG làm | Lý do |
|---|---|
| Dùng ChatGPT/Claude sinh câu BCTC giả | Vi phạm liêm chính, model học pattern sai |
| Dùng Tesseract OCR để đọc PDF scan | Độ chính xác thấp, nhiễu số liệu |
| Copy/paste từ tóm tắt Wikipedia | Không có cấu trúc BCTC thật |
| Tự bịa số liệu tài chính | Vi phạm nghiêm trọng nguyên tắc nghiên cứu |
| Dùng dataset nước ngoài không phù hợp | Khác chuẩn VAS, khác ngôn ngữ, không có mã khoản VN |
| Oversample bằng SMOTE trên text | Tạo mẫu giả — vi phạm nguyên tắc no-synthetic |
