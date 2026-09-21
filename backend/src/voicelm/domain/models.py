"""Core data types.

These are plain values with no I/O and no dependencies on the rest of the package.
Everything else depends on them; they depend on nothing.

All types are frozen (immutable). Once provenance has been attached to a chunk, later
code cannot quietly change it — an attempt raises instead of corrupting a citation.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PageSpan:
    """Which part of `Source.text` came from one page of the original document.

    Formats with no pages (`.txt`, `.md`) produce none of these, which is the honest
    representation: there is no page to cite. `number` is the 1-based physical position in
    the file, matching the "page 4 of 20" a PDF viewer shows, not the printed page label —
    front matter numbered in roman numerals would disagree with it (ADR-0024).
    """

    number: int
    start_char: int
    end_char: int


def pages_covering(start_char: int, end_char: int, pages: tuple[PageSpan, ...]) -> tuple[int, ...]:
    """Page numbers whose spans overlap `[start_char, end_char)`.

    A chunk that straddles a page break reports both pages rather than pretending it
    lives on one. Empty when the source has no pages (plain text, Markdown).
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

    `pages` maps regions of `text` back to pages of the original file, so a citation can
    say where to look. Empty for formats that have no pages.
    """

    id: str
    path: Path
    title: str
    text: str
    content_hash: str
    pages: tuple[PageSpan, ...] = ()


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
    # 1-based physical pages this quote overlaps. Empty for formats that have no pages.
    pages: tuple[int, ...] = ()


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    model: str
    # Markers the model wrote that matched no excerpt. Removed from `text`, but recorded
    # because they are direct evidence the answer is less grounded than it appears.
    unsupported_markers: tuple[int, ...] = ()
