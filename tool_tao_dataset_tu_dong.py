import os
import json
from pathlib import Path
from engine_parser import process_file

def tao_dataset_sll():
    """
    Script tạo Dataset tự động bằng cách quét thư mục:
    - Thu mục: data_raw/gian_lan (Tự động gán nhãn 1)
    - Thu mục: data_raw/sach (Tự động gán nhãn 0)
    """
    file_output = "dataset_sieu_toc.jsonl"
    
    # 1. Đọc danh sách các file PDF đã xử lý (để chống trùng lặp)
    processed_files = set()
    if os.path.exists(file_output):
        with open(file_output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "file" in data:
                            processed_files.add(data["file"])
                    except json.JSONDecodeError:
                        pass
                        
    print(f"[THÔNG TIN] Đã tìm thấy {len(processed_files)} file đã xử lý trước đó. Sẽ chỉ quét các file mới!")

    # 2. Khai báo thư mục chứa file PDF tải từ Cổng SSC về (Chỉ cần 1 thư mục)
    thu_muc_tong_hop = Path("data_raw/tong_hop")
    
    # Tạo sẵn thư mục nếu chưa có để bạn bỏ PDF vào
    thu_muc_tong_hop.mkdir(parents=True, exist_ok=True)
    
    danh_sach_ket_qua = []
    
    print("\n[AI-PARSER] Bắt đầu quét tự động các file mới trong mục TỔNG HỢP...")
    
    # 3. Quét toàn bộ PDF, để tự Auto-Label
    for file_pdf in thu_muc_tong_hop.glob("*.pdf"):
        if file_pdf.name in processed_files:
            continue # Bỏ qua nếu đã xử lý
            
        print(f"Đang bóc chữ: {file_pdf.name}...")
        ket_qua = process_file(file_pdf)
        if ket_qua and "error" not in ket_qua:
            # AI Parser () tự trả về "label" qua hàm auto_detect_label
            label = ket_qua.get("label", -1)
            loai = "GIAN LẬN" if label == 1 else "SẠCH" if label == 0 else "CHƯA RÕ"
            print(f"  -> AI Tự phân loại: {loai} (Nhãn: {label})")
            
            danh_sach_ket_qua.append(ket_qua)
            processed_files.add(file_pdf.name)
            
    # 5. Xuất ra file Dataset JSONL cho AI học (Ghi nối thêm vào file cũ)
    if danh_sach_ket_qua:
        with open(file_output, "a", encoding="utf-8") as f:
            for dong in danh_sach_ket_qua:
                f.write(json.dumps(dong, ensure_ascii=False) + "\n")
        print(f"\n[THÀNH CÔNG] Đã tạo và NỐI THÊM {len(danh_sach_ket_qua)} báo cáo MỚI vào: {file_output}\n")
    else:
        print("\n[HOÀN TẤT] Không có file PDF nào mới cần xử lý, bộ dữ liệu không có gì thay đổi.\n")

if __name__ == "__main__":
    tao_dataset_sll()
