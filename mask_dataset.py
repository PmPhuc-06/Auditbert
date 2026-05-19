import pandas as pd
import re

# File đầu vào và đầu ra
INPUT_FILE = "Tap_Du_Lieu_Test_Kiem_Tra.csv"
OUTPUT_FILE = "Tap_Du_Lieu_Test_Masked.csv"

# Danh sách các từ khóa lộ nhãn (Label Leakage) cần che giấu
# Dựa theo Phụ lục B: các kết luận của kiểm toán viên và dấu hiệu vi phạm rõ ràng
LEAKAGE_KEYWORDS = [
    r"ý kiến ngoại trừ",
    r"cơ sở của ý kiến ngoại trừ",
    r"ý kiến kiểm toán ngoại trừ",
    r"xử phạt vi phạm hành chính",
    r"từ chối đưa ra ý kiến",
    r"ý kiến trái ngược",
    r"nghi ngờ đáng kể về khả năng hoạt động liên tục",
    r"lưu ý của kiểm toán viên",
    r"vấn đề cần nhấn mạnh"
]

def mask_text(text):
    if not isinstance(text, str):
        return text
    
    masked_text = text
    for keyword in LEAKAGE_KEYWORDS:
        # Thay thế không phân biệt hoa thường (re.IGNORECASE)
        pattern = re.compile(keyword, re.IGNORECASE)
        masked_text = pattern.sub("[MASK]", masked_text)
        
    return masked_text

def main():
    print(f"Đang đọc dữ liệu từ {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    # Giả sử cột chứa văn bản là cột thứ 4 (index 3), tương tự cách eval_test_giangvien.py load
    text_col = df.columns[3]
    print(f"Tiến hành che giấu (mask) dữ liệu trên cột: '{text_col}'")
    
    # Áp dụng mask
    df[text_col] = df[text_col].apply(mask_text)
    
    # Lưu ra file mới
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Đã lưu dữ liệu sau khi mask ra file: {OUTPUT_FILE}")
    print("Bây giờ bạn có thể chạy đánh giá Bảng 4 bằng lệnh:")
    print(f"python eval_test_giangvien.py --csv {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
