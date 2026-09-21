"""Core data types.

These are plain values with no I/O and no dependencies on the rest of the package.
Everything else depends on them; they depend on nothing.

All types are frozen (immutable). Once provenance has been attached to a chunk, later
code cannot quietly change it — an attempt raises instead of corrupting a citation.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    """A document that has been ingested.

    `text` is the *cleaned* text and is the canonical content of the document: it is what
    we chunk, what we quote back in citations, and what chunk offsets index into.

    `id` is a UUID. Persistence looks files up by path and reuses the existing id when
    the same file is ingested again. `content_hash` is SHA-256 of `text` and decides
    whether re-embedding can be skipped (ADR-0019).
    """

    id: str
    path: Path
    title: str
    text: str
    content_hash: str


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


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    model: str
    # Markers the model wrote that matched no excerpt. Removed from `text`, but recorded
    # because they are direct evidence the answer is less grounded than it appears.
    unsupported_markers: tuple[int, ...] = ()
