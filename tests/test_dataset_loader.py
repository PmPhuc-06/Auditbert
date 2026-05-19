import json
import unittest
import uuid
from pathlib import Path

from engine_mfinbert import MoHinhGianLanMFinBERT
from engine_phobert import MoHinhGianLanPhoBERT


class LoadDatasetJsonTest(unittest.TestCase):
    def _write_temp_file(self, content: str) -> Path:
        path = Path(__file__).resolve().parent / f"dataset_{uuid.uuid4().hex}.json"
        path.write_text(content, encoding="utf-8")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def _assert_all_transformer_loaders(
        self,
        path: Path,
        expected_texts: list[str],
        expected_labels: list[int],
    ) -> None:
        for model_cls in (MoHinhGianLanPhoBERT, MoHinhGianLanMFinBERT):
            with self.subTest(model=model_cls.__name__):
                texts, labels = model_cls.load_dataset_json(str(path))
                self.assertEqual(texts, expected_texts)
                self.assertEqual(labels, expected_labels)

    def test_supports_pretty_printed_json_object(self) -> None:
        path = self._write_temp_file(
            json.dumps(
                {"texts": ["van ban 1", "van ban 2"], "labels": [0, 1]},
                indent=2,
                ensure_ascii=False,
            )
        )

        self._assert_all_transformer_loaders(path, ["van ban 1", "van ban 2"], [0, 1])

    def test_supports_json_array(self) -> None:
        path = self._write_temp_file(
            json.dumps(
                [
                    {"text": "van ban 1", "label": 0},
                    {"text": "van ban 2", "label": 1},
                ],
                ensure_ascii=False,
            )
        )

        self._assert_all_transformer_loaders(path, ["van ban 1", "van ban 2"], [0, 1])

    def test_supports_jsonl(self) -> None:
        path = self._write_temp_file(
            "\n".join(
                [
                    json.dumps({"text": "van ban 1", "label": 0}, ensure_ascii=False),
                    json.dumps({"text": "van ban 2", "label": 1}, ensure_ascii=False),
                ]
            )
        )

        self._assert_all_transformer_loaders(path, ["van ban 1", "van ban 2"], [0, 1])


if __name__ == "__main__":
    unittest.main()
