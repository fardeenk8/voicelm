"""Local OCR for images and scanned PDF pages.

Runs on this machine (RapidOCR / ONNX). No pixels leave the library. Results feed the
same `PageSpan` map every other format uses — page numbers for PDF pages, page 1 for a
standalone image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from voicelm.domain.models import PageSpan
from voicelm.ingestion.pdf import assemble_pages

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"})

# ~150 DPI is enough for body text; higher is slower without much gain for RAG chunks.
DEFAULT_PDF_RENDER_SCALE = float(os.environ.get("VOICELM_OCR_PDF_SCALE", "2.0"))


class OcrError(Exception):
    """Base for every reason OCR did not produce usable text."""


class OcrUnavailable(OcrError):
    """The OCR engine or a required native dependency is not installed."""


class EmptyOcrResult(OcrError):
    """OCR ran, but found no readable text."""


class CorruptImage(OcrError):
    """The image file could not be opened."""


@dataclass(frozen=True)
class OcrPage:
    """Text recovered from one rendered page or one image."""

    text: str


class OcrEngine(Protocol):
    def read_image(self, path: Path) -> str: ...

    def read_pil(self, image: object) -> str: ...


class RapidOcrEngine:
    """Lazy RapidOCR wrapper — model weights download on first use."""

    def __init__(self) -> None:
        self._engine = None

    def read_image(self, path: Path) -> str:
        try:
            from PIL import Image
        except ImportError as error:
            raise OcrUnavailable("Pillow is required for OCR") from error
        try:
            with Image.open(path) as image:
                return self.read_pil(image.convert("RGB"))
        except OSError as error:
            raise CorruptImage(f"{path.name} could not be opened as an image") from error

    def read_pil(self, image: object) -> str:
        engine = self._load()
        import numpy as np

        array = np.asarray(image)
        try:
            result, _ = engine(array)
        except Exception as error:
            raise OcrError(f"OCR failed: {error}") from error
        if not result:
            return ""
        # Each row is [box, text, confidence].
        lines = [row[1] for row in result if len(row) > 1 and row[1]]
        return "\n".join(lines)

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as error:
            raise OcrUnavailable(
                "rapidocr-onnxruntime is not installed; cannot OCR images or scans"
            ) from error
        self._engine = RapidOCR()
        return self._engine


_default_ocr: RapidOcrEngine | None = None


def default_ocr() -> RapidOcrEngine:
    global _default_ocr
    if _default_ocr is None:
        _default_ocr = RapidOcrEngine()
    return _default_ocr


def ocr_image(path: Path, *, engine: OcrEngine | None = None) -> tuple[str, tuple[PageSpan, ...]]:
    """OCR a single image into cleaned text with one page span."""
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise CorruptImage(f"{path.name} is not a supported image type")
    ocr = engine or default_ocr()
    raw = ocr.read_image(path)
    text, pages = assemble_pages([raw])
    if not text:
        raise EmptyOcrResult(f"{path.name} has no readable text")
    return text, pages


def render_pdf_page(path: Path, page_index: int, *, scale: float = DEFAULT_PDF_RENDER_SCALE):
    """Render one 0-based PDF page to a PIL Image."""
    try:
        import pypdfium2 as pdfium
    except ImportError as error:
        raise OcrUnavailable("pypdfium2 is required to OCR scanned PDFs") from error

    document = pdfium.PdfDocument(str(path))
    try:
        if page_index < 0 or page_index >= len(document):
            raise OcrError(f"{path.name} has no page {page_index + 1}")
        page = document[page_index]
        bitmap = page.render(scale=scale)
        return bitmap.to_pil()
    finally:
        document.close()


def ocr_pdf_pages(
    path: Path,
    *,
    engine: OcrEngine | None = None,
    existing: list[str] | None = None,
    scale: float = DEFAULT_PDF_RENDER_SCALE,
) -> list[str]:
    """OCR blank PDF pages (or every page when `existing` is all empty).

    Pages that already have a text layer keep that text. Blank pages are rendered and
    run through OCR so a mixed born-digital / scanned PDF still cites correctly.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as error:
        raise OcrUnavailable("pypdfium2 is required to OCR scanned PDFs") from error

    ocr = engine or default_ocr()
    document = pdfium.PdfDocument(str(path))
    try:
        count = len(document)
        pages = list(existing) if existing is not None else [""] * count
        if len(pages) != count:
            pages = (pages + [""] * count)[:count]

        for index in range(count):
            if pages[index].strip():
                continue
            page = document[index]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            pages[index] = ocr.read_pil(image)
    finally:
        document.close()

    if not any(page.strip() for page in pages):
        raise EmptyOcrResult(f"{path.name} has no extractable text, and OCR found none either")
    return pages
