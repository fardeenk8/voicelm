"""Turning a PDF into text plus a map back to its pages.

A PDF does not contain text. It contains instructions for placing glyphs at coordinates:
"draw 'T' at x=72, y=700 in 11pt Helvetica". There are no paragraphs, no reading order,
and often no space characters — spacing can be nothing but position. Extraction is
therefore *reconstruction by heuristic*, which is why two libraries disagree about the
same file and why the output needs the same cleaning pass as any other input.

Two functions, split so the offset arithmetic can be tested without touching a disk:
`extract_pages` does the I/O and the guessing, `assemble_pages` does the bookkeeping.
"""

import logging
import unicodedata
import warnings
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import (
    DependencyError,
    FileNotDecryptedError,
    PyPdfError,
    WrongPasswordError,
)

from voicelm.domain.models import PageSpan
from voicelm.ingestion.cleaning import clean_text

# Pages are joined with a blank line, which chunking already treats as its preferred split
# point. So chunks line up with page boundaries wherever the budget allows, without page
# breaks being *forced* cuts that would slice sentences in half (ADR-0023).
PAGE_SEPARATOR = "\n\n"


class PdfExtractionError(Exception):
    """Base for every reason a PDF yielded no usable text."""


class CorruptPdf(PdfExtractionError):
    """The file is not a PDF we can parse at all."""


class EncryptedPdf(PdfExtractionError):
    """The file is password protected and we were not given the password."""


class NoTextLayer(PdfExtractionError):
    """Parsed fine, but every page was empty — almost always a scan.

    Worth its own error because the alternative is ingesting a document with no content:
    it would list in `sources`, match nothing, and look like a retrieval bug rather than
    an unsupported file. OCR would fix it and is deliberately out of scope for Phase 1.
    """


def extract_pages(path: Path) -> list[str]:
    """Return the raw text of each page of `path`, in document order.

    Pages that yield nothing come back as empty strings rather than being dropped, so the
    caller can still number the pages that follow them correctly.
    """
    try:
        pages = _read_pdf_pages(path)
    # Narrower than PyPdfError and therefore listed first: these mean "encrypted", and
    # reporting them as corruption would send you looking for the wrong problem.
    except (DependencyError, FileNotDecryptedError, WrongPasswordError) as error:
        raise EncryptedPdf(f"{path.name} is encrypted and could not be opened") from error
    except PyPdfError as error:
        raise CorruptPdf(f"{path.name} could not be read as a PDF: {error}") from error

    if not any(page.strip() for page in pages):
        raise NoTextLayer(
            f"{path.name} has no extractable text. Scanned PDFs need OCR, "
            "which VoiceLM does not do yet."
        )

    return pages


def _read_pdf_pages(path: Path) -> list[str]:
    """Parse `path` with pypdf, muting the fontTools warnings that are not our problem."""
    logger = logging.getLogger("pypdf")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="fontTools is required")
            reader = PdfReader(path)

            # Plenty of real PDFs are "encrypted" with an empty user password purely to set
            # permission flags such as no-printing. Those open fine, so try before refusing.
            if reader.is_encrypted and not reader.decrypt(""):
                raise EncryptedPdf(f"{path.name} is password protected")

            return [page.extract_text() or "" for page in reader.pages]
    finally:
        logger.setLevel(previous)


def assemble_pages(pages: list[str]) -> tuple[str, tuple[PageSpan, ...]]:
    """Clean each page, join them, and record where each one landed.

    Cleaning happens per page and the result is *not* cleaned again, because a second pass
    could shift characters and invalidate the offsets we just recorded. That is safe here:
    `clean_text` collapses blank-line runs to exactly one and strips the ends, so pages
    joined by a single blank line are already in the form it produces.

    Pages that clean away to nothing get no span. Nothing can be cited from a blank page,
    and `PageSpan.number` carries the page number explicitly, so skipping one does not
    renumber the rest.
    """
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0

    for number, raw in enumerate(pages, start=1):
        # PDFs commonly emit presentation forms (ﬁ, ﬂ) as a single glyph. NFKC turns
        # those into the letters a searcher would type; we keep this PDF-only so Markdown
        # indentation and superscripts are not rewritten (ADR-0025).
        cleaned = clean_text(unicodedata.normalize("NFKC", raw))
        if not cleaned:
            continue

        if parts:
            cursor += len(PAGE_SEPARATOR)
        spans.append(PageSpan(number=number, start_char=cursor, end_char=cursor + len(cleaned)))
        parts.append(cleaned)
        cursor += len(cleaned)

    return PAGE_SEPARATOR.join(parts), tuple(spans)
