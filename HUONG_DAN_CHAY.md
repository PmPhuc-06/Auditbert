# Hướng dẫn Cài đặt & Chạy Hệ Thống BaoOngThay

Tài liệu này hướng dẫn chi tiết cách thiết lập môi trường, cài đặt các thư viện cần thiết và khởi chạy dự án BaoOngThay.

## 1. Yêu cầu hệ thống (Prerequisites)
- **Python**: Phiên bản 3.9 đến 3.11 (khuyên dùng Python 3.10)
- **Hệ điều hành**: Windows, macOS, hoặc Linux
- **RAM**: Khuyến nghị từ 8GB trở lên vì các mô hình AI/Transformer (PhoBERT, AuditBERT) chiếm khá nhiều bộ nhớ khi chạy.

---

## 2. Các bước cài đặt

### Bước 2.1: Tạo môi trường ảo (Virtual Environment)
Nhằm tránh xung đột thư viện với các dự án khác, bạn nên tạo môi trường ảo:

```bash
# Mở Terminal / PowerShell trong thư mục dự án (BaoOngThay)
python -m venv venv

# Kích hoạt môi trường ảo (trên Windows):
.\venv\Scripts\activate

# (Hoặc kích hoạt trên macOS/Linux):
source venv/bin/activate
```
*(Nếu kích hoạt thành công, bạn sẽ thấy chữ `(venv)` hiện ở đầu dòng lệnh).*

### Bước 2.2: Cài đặt thư viện
Trong thư mục dự án đã có sẵn file `requirements-web.txt`, chạy lệnh sau để tự động cài các thư viện quan trọng như FastAPI, Transformer, PyTorch, Uvicorn, v.v.

```bash
pip install -r requirements-web.txt
```

Ngoài ra, hệ thống sử dụng thư viện xử lý tài liệu định dạng PDF, đảm bảo bạn cài thêm PyMuPDF (nếu chưa có):
```bash
pip install pymupdf
```

---

## 3. Các thao tác chạy hệ thống

### 3.1. Chạy API Server (Khởi động giao diện chính)
Sau khi cài đặt xong, bạn có thể khởi chạy server bằng Uvicorn:

```bash
uvicorn app:app --reload
```
- Truy cập Swagger UI để test API API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Hoặc mở giao diện web mặc định (nếu có): [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 3.2. Huấn luyện AuditBERT (Tạo checkpoint mô hình)
Lưu ý: Nếu project chưa có checkpoint (`.pt`), bạn không thể chạy suy luận hệ thống. Chạy câu lệnh dưới đây để bắt đầu trình đào tạo mô hình offline tạo checkpoint nhẹ ~540MB:

```bash
python train_auditbert.py
```
*Lưu ý: Ở chế độ demo, bạn có thể chạy `python train_auditbert.py --quick` để train cực nhanh.*

---

## 4. Xử lý lỗi Git Push bị lỗi dung lượng (HTTP 500 / RPC failed)
Gần đây bạn có thể gặp lỗi khi `git push` (VD: `send-pack: unexpected disconnect`). Lỗi này xuất hiện vì các mô hình trọng số (`*.pt`, `*.bin`, `*.jsonl`) có dung lượng RẤT LỚN (trên 500MB đến 1GB) không được phép đưa trực tiếp lên Git do giới hạn của GitHub/GitLab.

**Cách khắc phục:** 

Hủy lưu trữ thư mục mô hình khỏi git (Chỉ xóa trên git, file trong máy bạn vẫn an toàn) bằng cách gõ:
```bash
git rm --cached *.pt
git rm --cached *.jsonl
git commit -m "Xóa theo dõi các file mô hình checkpoint nặng"
git push origin clean_branch
```
(Hãy đảm bảo bạn đã đưa `*.pt` và `*.jsonl` vào file `.gitignore` để sau này không bị lỗi nhé!)
