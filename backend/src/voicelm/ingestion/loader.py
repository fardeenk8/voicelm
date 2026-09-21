"""Reading documents off disk into a `Source`."""

import hashlib
from pathlib import Path

from voicelm.domain.models import Source
from voicelm.ingestion.cleaning import clean_text

SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown"})


class UnsupportedFileType(Exception):
    """Raised for a file extension this milestone cannot handle."""


class UndecodableFile(Exception):
    """Raised when a file is not valid UTF-8."""


def load_source(path: Path) -> Source:
    """Read and clean a text document.

    Only UTF-8 is accepted. Legacy encodings such as cp1252 are common in the wild, but
    guessing an encoding can silently corrupt text, and corrupted text produces confident
    nonsense in citations. Failing loudly is the better default; encoding detection is
    worth adding later, explicitly.
    """
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileType(f"{path.name}: expected one of {sorted(SUPPORTED_SUFFIXES)}")

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise UndecodableFile(f"{path.name} is not valid UTF-8") from error

    text = clean_text(raw)

    return Source(id=_content_id(text), path=path, title=path.stem, text=text)


def _content_id(text: str) -> str:
    """Derive a stable id from the content itself.

    Because the id is a hash of the text rather than a random value, re-ingesting an
    unchanged file produces the same id and the same chunk ids. That makes ingestion
    repeatable and gives us de-duplication for free.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
