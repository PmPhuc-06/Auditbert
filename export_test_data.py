import json
import csv
import sys

def convert_to_csv():
    try:
        with open('baseline_split_test.jsonl', 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
            
        with open('Tap_Du_Lieu_Test_Kiem_Tra.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['ID (File Gốc)', 'Nguồn Trích Xuất', 'Nhãn Gian Lận (1=Có, 0=Không)', 'Nội Dung Văn Bản Test'])
            for s in samples:
                writer.writerow([s.get('doc_id', ''), s.get('source', ''), s.get('label', ''), s.get('text', '')])
        print("Đã tạo thành công Tap_Du_Lieu_Test_Kiem_Tra.csv !")
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == '__main__':
    convert_to_csv()
