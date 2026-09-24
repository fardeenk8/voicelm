"""DOCX and PPTX → text plus the same unit index PDFs use.

Both formats land in `Source.text` through `assemble_pages`. A PowerPoint slide is the
viewer unit (open the deck, go to slide N). A Word page is not stable across printers,
so we cite the paragraph the user can search for (ADR-0029).
"""

from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pptx import Presentation
from pptx.exc import PackageNotFoundError as PptxPackageNotFoundError

from voicelm.domain.models import PageSpan
from voicelm.ingestion.pdf import assemble_pages

_OPEN_ERRORS = (
    PackageNotFoundError,
    PptxPackageNotFoundError,
    BadZipFile,
    ValueError,
    OSError,
    KeyError,
)


class OfficeExtractionError(Exception):
    """Base for every reason an Office file yielded no usable text."""


class CorruptOfficeFile(OfficeExtractionError):
    """The file is not a DOCX/PPTX we can open."""


class EmptyOfficeDocument(OfficeExtractionError):
    """Parsed fine, but every unit was empty — nothing to embed or cite."""


def load_docx(path: Path) -> tuple[str, tuple[PageSpan, ...]]:
    """Read a Word file into cleaned text and paragraph spans."""
    return assemble_pages(extract_docx_paragraphs(path))


def load_pptx(path: Path) -> tuple[str, tuple[PageSpan, ...]]:
    """Read a PowerPoint file into cleaned text and slide spans."""
    return assemble_pages(extract_pptx_slides(path))


def extract_docx_paragraphs(path: Path) -> list[str]:
    """Return body paragraphs and table cells, in document order.

    Empty strings are kept so later non-empty units keep their physical numbers, matching
    how blank PDF pages are handled.
    """
    try:
        document = Document(path)
    except _OPEN_ERRORS as error:
        raise CorruptOfficeFile(f"{path.name} could not be read as a Word document") from error

    units: list[str] = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                units.append(cell.text)

    if not any(unit.strip() for unit in units):
        raise EmptyOfficeDocument(f"{path.name} has no extractable text")
    return units


def extract_pptx_slides(path: Path) -> list[str]:
    """Return the text of each slide, in deck order.

    Empty slides stay as empty strings so slide numbers after them stay correct.
    """
    try:
        presentation = Presentation(path)
    except _OPEN_ERRORS as error:
        raise CorruptOfficeFile(f"{path.name} could not be read as a PowerPoint file") from error

    slides: list[str] = []
    for slide in presentation.slides:
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        parts.append(cell.text)
        slides.append("\n".join(part for part in parts if part.strip()))

    if not any(slide.strip() for slide in slides):
        raise EmptyOfficeDocument(f"{path.name} has no extractable text")
    return slides
