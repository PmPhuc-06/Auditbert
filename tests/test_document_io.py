import unittest
import os
from pathlib import Path
from unittest import mock

from engine_document_io import LoiTrichXuatTaiLieu, _tim_poppler_path, doc_tai_lieu_tu_bytes


class DocumentIOTest(unittest.TestCase):
    def test_reads_txt_bytes_and_marks_metadata(self) -> None:
        result = doc_tai_lieu_tu_bytes(
            "Doanh thu tăng nhưng dòng tiền âm.".encode("utf-8"),
            "bao_cao.txt",
        )

        self.assertEqual(result.source_type, "txt")
        self.assertEqual(result.extraction_method, "plain_text")
        self.assertFalse(result.ocr_used)
        self.assertIn("Doanh thu tăng", result.text)

    def test_rejects_unsupported_extension(self) -> None:
        with self.assertRaises(LoiTrichXuatTaiLieu):
            doc_tai_lieu_tu_bytes(b"abc", "bao_cao.docx")

    def test_detects_portable_poppler_from_path_entry_pointing_to_root_bin(self) -> None:
        portable_root = Path("tests/fixtures/poppler_portable")
        library_bin = portable_root / "Library" / "bin"

        fake_env = {
            "PATH": str(portable_root / "bin"),
            "POPPLER_PATH": "",
            "POPPLER_BIN": "",
            "PDF2IMAGE_POPPLER_PATH": "",
        }
        with mock.patch.dict(os.environ, fake_env, clear=False):
            with mock.patch("engine_document_io.shutil.which", return_value=None):
                detected = _tim_poppler_path()

        self.assertEqual(Path(detected), library_bin)

    def test_detects_poppler_when_env_points_to_portable_root(self) -> None:
        portable_root = Path("tests/fixtures/poppler_portable")
        library_bin = portable_root / "Library" / "bin"

        fake_env = {
            "POPPLER_PATH": str(portable_root),
            "POPPLER_BIN": "",
            "PDF2IMAGE_POPPLER_PATH": "",
        }
        with mock.patch.dict(os.environ, fake_env, clear=False):
            with mock.patch("engine_document_io.shutil.which", return_value=None):
                detected = _tim_poppler_path()

        self.assertEqual(Path(detected), library_bin)


if __name__ == "__main__":
    unittest.main()
