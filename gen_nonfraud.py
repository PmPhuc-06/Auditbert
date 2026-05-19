import json
from vnstock3 import Vnstock

symbols = ["VNM","FPT","HPG","MWG","VCB","SSI","TCB","VIC","VHM","GAS"]
output = []
for sym in symbols:
    try:
        stock = Vnstock().stock(symbol=sym, source='TCBS')
        df = stock.finance.income_statement(period='year', to_date='2023-12-31')
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            text = f"Công ty {sym} doanh thu {latest.get('netRevenue',0)} tỷ, lợi nhuận {latest.get('netIncome',0)} tỷ."
            output.append({"text": text, "label": 0, "source": "vnstock3", "doc_id": sym})
    except Exception as e:
        print(f"Lỗi {sym}: {e}")
with open("nonfraud_samples.jsonl", "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Tạo {len(output)} mẫu non-fraud")