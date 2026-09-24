"""Reading documents off disk into a `Source`."""

import hashlib
import uuid
from pathlib import Path

from voicelm.domain.models import PageSpan, Source
from voicelm.ingestion.cleaning import clean_text
from voicelm.ingestion.media import MEDIA_SUFFIXES, load_media
from voicelm.ingestion.ocr import IMAGE_SUFFIXES, ocr_image
from voicelm.ingestion.office import load_docx, load_pptx
from voicelm.ingestion.pdf import assemble_pages, extract_pages

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".ytt"})
PDF_SUFFIXES = frozenset({".pdf"})
DOCX_SUFFIXES = frozenset({".docx"})
PPTX_SUFFIXES = frozenset({".pptx"})
SUPPORTED_SUFFIXES = (
    TEXT_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES | PPTX_SUFFIXES | MEDIA_SUFFIXES | IMAGE_SUFFIXES
)


class UnsupportedFileType(Exception):
    """Raised for a file extension this milestone cannot handle."""


class UndecodableFile(Exception):
    """Raised when a file is not valid UTF-8."""


def hash_content(text: str) -> str:
    """SHA-256 of the cleaned text, used to skip re-embedding unchanged files.

    This is *not* the source identity. Two files with the same hash are still two sources
    if they live at different paths (ADR-0019).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_source(path: Path, *, ocr: object | None = None, use_ocr: bool = True) -> Source:
    """Read and clean a document, whatever supported format it is in.

    `id` is a fresh UUID every call. Persistence looks the file up by *path* and reuses
    the existing id when the file was ingested before. The loader cannot do that lookup:
    it does not know whether a store exists yet.

    Everything after this point is format-blind. Chunking, embedding, retrieval, and
    generation see a `Source` and never ask where its text came from; the only trace of the
    original format is whether `pages` is populated and what `location_kind_for` returns.

    `ocr` is forwarded to PDF extraction and image OCR so tests can inject a fake engine.
    """
    text: str
    pages: tuple[PageSpan, ...]
    content_hash: str

    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        text, pages = assemble_pages(extract_pages(path, ocr=ocr, use_ocr=use_ocr))
        content_hash = hash_content(text)
    elif suffix in DOCX_SUFFIXES:
        text, pages = load_docx(path)
        content_hash = hash_content(text)
    elif suffix in PPTX_SUFFIXES:
        text, pages = load_pptx(path)
        content_hash = hash_content(text)
    elif suffix in IMAGE_SUFFIXES:
        text, pages = ocr_image(path, engine=ocr)  # type: ignore[arg-type]
        content_hash = hash_content(text)
    elif suffix in MEDIA_SUFFIXES:
        # Hash is of file bytes so an unchanged recording skips Whisper (ADR-0032).
        text, pages, content_hash = load_media(path)
    elif suffix in TEXT_SUFFIXES:
        text, pages = _read_text_file(path), ()
        content_hash = hash_content(text)
    else:
        raise UnsupportedFileType(f"{path.name}: expected one of {sorted(SUPPORTED_SUFFIXES)}")

    return Source(
        id=str(uuid.uuid4()),
        path=path,
        title=path.stem,
        text=text,
        content_hash=content_hash,
        pages=pages,
    )


def _read_text_file(path: Path) -> str:
    """Read a plain-text document.

    Only UTF-8 is accepted. Legacy encodings such as cp1252 are common in the wild, but
    guessing an encoding can silently corrupt text, and corrupted text produces confident
    nonsense in citations. Failing loudly is the better default; encoding detection is
    worth adding later, explicitly.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise UndecodableFile(f"{path.name} is not valid UTF-8") from error

    return clean_text(raw)
