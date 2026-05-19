import unittest

from engine import PredictionResult, tao_ket_qua_du_doan


class PredictionOutputTest(unittest.TestCase):
    def test_payload_keeps_raw_and_adjusted_scores_clear(self) -> None:
        prediction = PredictionResult(
            label=1,
            muc_do_rui_ro="Cao",
            model_probability_fraud=0.64321,
            model_probability_non_fraud=0.35679,
            probability_fraud=0.74321,
            probability_non_fraud=0.25679,
            threshold_used=0.5,
            chat_luong_van_ban=0.91,
            red_flags=["hoa don gia"],
            explanation=["foo"],
            top_terms=["hoa", "don"],
            van_ban_sach="van ban sach",
            van_ban_embedding="van ban embedding",
            doan_nghi_ngo=[{"text": "x", "reason": "y"}],
            metadata_features={"ocr_quality": 0.91234},
            raw_text_probability_fraud=0.61234,
            raw_text_probability_non_fraud=0.38766,
            bias_flags=["low_ocr_quality"],
            explainability={"available_methods": ["ig"]},
        )

        result = tao_ket_qua_du_doan(
            "phobert",
            prediction,
            model_source="checkpoint:phobert_fraud_checkpoint.pt",
            top_terms_method="attention_hint",
        )

        self.assertEqual(result["label_id"], 1)
        self.assertEqual(result["label"], "Cảnh báo rủi ro")
        self.assertEqual(result["legacy_label"], "Gian lận")
        self.assertEqual(result["model_source"], "checkpoint:phobert_fraud_checkpoint.pt")
        self.assertEqual(result["top_terms_method"], "attention_hint")
        self.assertEqual(result["model_fraud_probability"], 0.6432)
        self.assertEqual(result["fraud_probability"], 0.7432)
        self.assertEqual(result["risk_score"], 0.7432)
        self.assertEqual(
            result["score_semantics"]["model_fraud_probability"],
            "raw_model_probability",
        )
        self.assertEqual(
            result["score_semantics"]["fraud_probability"],
            "after_rule_based_postprocessing",
        )
        self.assertEqual(result["metadata_features"]["ocr_quality"], 0.9123)
        self.assertEqual(result["raw_text_probability_fraud"], 0.6123)
        self.assertEqual(result["raw_text_probability_non_fraud"], 0.3877)
        self.assertEqual(result["bias_flags"], ["low_ocr_quality"])
        self.assertEqual(result["explainability"]["available_methods"], ["ig"])


if __name__ == "__main__":
    unittest.main()
