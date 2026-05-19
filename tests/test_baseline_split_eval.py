import unittest

from engine_baseline import MoHinhGianLan
from engine_common import DataSplit


class BaselineSplitEvalTest(unittest.TestCase):
    def test_fit_from_split_returns_metrics_and_threshold(self) -> None:
        split = DataSplit(
            train_texts=[
                "Doanh thu tăng hợp lệ nhờ hợp đồng và xác nhận ngân hàng.",
                "Chi phí được ghi nhận đầy đủ, kiểm soát nội bộ tốt.",
                "Hóa đơn giả được dùng để ghi nhận doanh thu khống.",
                "Che giấu nợ phải trả ngoài bảng cân đối.",
            ],
            train_labels=[0, 0, 1, 1],
            val_texts=[
                "Doanh thu khống từ hóa đơn giả cuối kỳ.",
                "Báo cáo minh bạch, không có giao dịch bất thường.",
            ],
            val_labels=[1, 0],
            test_texts=[
                "Thay đổi chính sách kế toán không có giải trình.",
                "Dòng tiền ổn định và chứng từ đầy đủ.",
            ],
            test_labels=[1, 0],
        )

        model = MoHinhGianLan(epochs=200)
        metrics = model.fit_from_split(split, luu_split=False)

        self.assertIn("threshold", metrics)
        self.assertIn("f1", metrics)
        self.assertGreaterEqual(metrics["threshold"], 0.0)
        self.assertLessEqual(metrics["threshold"], 1.0)
        self.assertTrue(model.best_val_metrics)


if __name__ == "__main__":
    unittest.main()
