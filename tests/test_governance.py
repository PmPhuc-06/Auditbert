import json
import unittest
import uuid
from pathlib import Path

from engine_governance import kiem_tra_governance_dataset


class GovernanceDatasetTest(unittest.TestCase):
    def _write_temp_file(self, payload: object) -> Path:
        path = Path(__file__).resolve().parent / f"governance_{uuid.uuid4().hex}.jsonl"
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_report_contains_steps_and_ready_gate(self) -> None:
        path = self._write_temp_file(
            [
                {
                    "text": "Doanh thu được ghi nhận hợp lệ.",
                    "label": 0,
                    "source": "erp",
                    "doc_id": "DOC-001",
                    "version": "v1",
                    "collected_at": "2026-03-27",
                    "anonymized": True,
                    "privacy_reviewed": True,
                },
                {
                    "text": "Phát hiện hóa đơn giả và doanh thu khống.",
                    "label": 1,
                    "source": "audit_note",
                    "doc_id": "DOC-002",
                    "version": "v1",
                    "collected_at": "2026-03-27",
                    "anonymized": True,
                    "privacy_reviewed": True,
                },
            ]
        )

        report = kiem_tra_governance_dataset(str(path))

        self.assertIn("step_1_collect_data", report)
        self.assertIn("step_2_ethics_privacy", report)
        self.assertIn("step_3_label_quality", report)
        self.assertIn("step_4_ready_gate", report)
        self.assertIn("ready_for_training", report["step_4_ready_gate"])


if __name__ == "__main__":
    unittest.main()
