"""Core data types.

These are plain values with no I/O and no dependencies on the rest of the package.
Everything else depends on them; they depend on nothing.

All types are frozen (immutable). Once provenance has been attached to a chunk, later
code cannot quietly change it — an attempt raises instead of corrupting a citation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# What `PageSpan.number` means for this source. Same table, different word in the UI.
LocationKind = Literal["page", "slide", "paragraph", "timestamp", "line"]


@dataclass(frozen=True)
class PageSpan:
    """Which part of `Source.text` came from one unit of the original document.

    The unit is a PDF page, a PowerPoint slide, a Word paragraph, a whole second of
    audio/video, or a source line — see `location_kind_for`. Formats with no units
    (plain `.txt` / `.md` outside GitHub) produce none of these. `number` is 1-based
    position in the file for pages/slides/paragraphs/lines (ADR-0024); for timestamps
    it is seconds from the start of the media (ADR-0032).
    """

    number: int
    start_char: int
    end_char: int


def location_kind_for(path: Path) -> LocationKind | None:
    """How to label `pages` numbers for this file. None when there are no units."""
    # Owned GitHub tree copies cite by line (ADR-0034).
    if "github" in path.parts:
        return "line"
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "page"
    if suffix == ".pptx":
        return "slide"
    if suffix == ".docx":
        return "paragraph"
    # `.ytt` is our owned YouTube transcript copy. Media suffixes are recordings.
    if suffix == ".ytt" or suffix in {
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".flac",
        ".aac",
        ".wma",
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
        ".mpeg",
        ".mpg",
        ".m4v",
    }:
        return "timestamp"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}:
        return "page"
    return None


def pages_covering(start_char: int, end_char: int, pages: tuple[PageSpan, ...]) -> tuple[int, ...]:
    """Unit numbers whose spans overlap `[start_char, end_char)`.

    A chunk that straddles a break reports both units rather than pretending it lives
    on one. Empty when the source has no units (plain text, Markdown).
    """
    return tuple(
        span.number for span in pages if span.start_char < end_char and span.end_char > start_char
    )


@dataclass(frozen=True)
class Source:
    """A document that has been ingested.

    `text` is the *cleaned* text and is the canonical content of the document: it is what
    we chunk, what we quote back in citations, and what chunk offsets index into.

    `id` is a UUID. Persistence looks files up by path and reuses the existing id when
    the same file is ingested again. `content_hash` is SHA-256 of `text` and decides
    whether re-embedding can be skipped (ADR-0019).

    `pages` maps regions of `text` back to units of the original file (PDF page, slide,
    or paragraph), so a citation can say where to look. Empty for formats that have none.

    `origin_url` is set when the source was fetched from the web (ADR-0030). File-based
    sources leave it None.
    """

    id: str
    path: Path
    title: str
    text: str
    content_hash: str
    pages: tuple[PageSpan, ...] = ()
    origin_url: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A slice of a source, small enough to embed as a single coherent idea.

    `start_char` and `end_char` are offsets into `Source.text`, so this always holds:

        source.text[chunk.start_char : chunk.end_char] == chunk.text

    That invariant is what makes a citation verifiable rather than a claim.
    """

    id: str
    source_id: str
    text: str
    start_char: int
    end_char: int
    ordinal: int


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk together with the vector representing its meaning.

    `model` is recorded because vectors from different embedding models are not
    comparable — mixing them silently returns nonsense rather than failing. Carrying the
    model name lets us detect that (ADR-0008).
    """

    chunk: Chunk
    vector: tuple[float, ...]
    model: str

    @property
    def dimensions(self) -> int:
        return len(self.vector)


@dataclass(frozen=True)
class Citation:
    """Where one claim in an answer came from.

    `marker` is the number the model wrote in the answer text, so `[2]` in the prose and
    the citation labelled 2 refer to the same excerpt. Everything else is recorded from our
    own ingestion records, never from the model, so a citation cannot be hallucinated.
    """

    marker: int
    chunk_id: str
    source_id: str
    source_title: str
    start_char: int
    end_char: int
    quote: str
    # 1-based units this quote overlaps (pages, slides, or paragraphs). Empty when none.
    pages: tuple[int, ...] = ()
    location_kind: LocationKind | None = None
    origin_url: str | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    model: str
    # Markers the model wrote that matched no excerpt. Removed from `text`, but recorded
    # because they are direct evidence the answer is less grounded than it appears.
    unsupported_markers: tuple[int, ...] = ()
