from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import shutil

class LoiTrichXuatTaiLieu(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class KetQuaTrichXuatTaiLieu:
    text: str
    source_type: str
    extraction_method: str
    ocr_used: bool


def _ung_vien_poppler(path: str | Path) -> list[Path]:
    base = Path(path)
    return [
        base,
        base / "bin",
        base / "Library" / "bin",
        base.parent / "Library" / "bin",
    ]


def _la_thu_muc_poppler(path: Path) -> bool:
    return (path / "pdftoppm.exe").exists() or (path / "pdftoppm").exists()


def _tim_poppler_path() -> str | None:
    for env_name in ("POPPLER_PATH", "POPPLER_BIN", "PDF2IMAGE_POPPLER_PATH"):
        value = os.environ.get(env_name, "").strip()
        if not value:
            continue
        for candidate in _ung_vien_poppler(value):
            if _la_thu_muc_poppler(candidate):
                return str(candidate)

    found = shutil.which("pdftoppm")
    if found:
        return str(Path(found).parent)

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        for candidate in _ung_vien_poppler(entry):
            if _la_thu_muc_poppler(candidate):
                return str(candidate)
    return None


def doc_pdf_tu_bytes(raw: bytes) -> KetQuaTrichXuatTaiLieu:
    try:
        import fitz
    except ImportError:
        raise LoiTrichXuatTaiLieu(
            "Thiếu thư viện xử lý PDF. Vui lòng cài đặt: pip install pymupdf",
            status_code=500,
        )

    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text("layout") or "")
        text = "\n\n".join(parts).strip()
        doc.close()
    except Exception as exc:
        raise LoiTrichXuatTaiLieu(
            f"Lỗi phân tích PDF bằng PyMuPDF: {exc}",
            status_code=400,
        )

    if not text or len(text) < 20:
        raise LoiTrichXuatTaiLieu(
            "Yêu cầu từ chối: File PDF là ảnh Scan (không có Text-layer). "
            "Hệ thống đã tắt OCR theo cấu hình 'Không AI scan data' để đảm bảo tính liêm chính dữ liệu.",
            status_code=400,
        )

    return KetQuaTrichXuatTaiLieu(
        text=text,
        source_type="pdf",
        extraction_method="pymupdf_text_layer",
        ocr_used=False,
    )


def doc_txt_tu_bytes(raw: bytes) -> KetQuaTrichXuatTaiLieu:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="ignore")
    text = text.strip()
    if not text:
        raise LoiTrichXuatTaiLieu("File không có nội dung hợp lệ.", status_code=400)
    return KetQuaTrichXuatTaiLieu(
        text=text,
        source_type="txt",
        extraction_method="plain_text",
        ocr_used=False,
    )


def doc_tai_lieu_tu_bytes(raw: bytes, file_name: str) -> KetQuaTrichXuatTaiLieu:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".txt":
        return doc_txt_tu_bytes(raw)
    if suffix == ".pdf":
        return doc_pdf_tu_bytes(raw)
    raise LoiTrichXuatTaiLieu(
        "Chỉ hỗ trợ file .txt hoặc .pdf",
        status_code=400,
    )


def doc_tai_lieu_tu_duong_dan(file_path: str) -> KetQuaTrichXuatTaiLieu:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    raw = path.read_bytes()
    return doc_tai_lieu_tu_bytes(raw, path.name)
