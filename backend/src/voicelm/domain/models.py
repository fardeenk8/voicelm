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
    """

    id: str
    path: Path
    title: str
    text: str


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
