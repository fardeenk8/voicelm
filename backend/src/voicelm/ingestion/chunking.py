"""Splitting a document into chunks small enough to embed as a single idea.

The whole module works in *character spans* — `(start, end)` pairs indexing into the
source text — and only slices out strings at the very end. That is what keeps the
round-trip invariant true by construction:

    source.text[chunk.start_char : chunk.end_char] == chunk.text

Strategy, in order of preference for where to split:

1. Paragraph breaks. A paragraph is usually one idea, so this is the natural seam.
2. Sentence ends, for a paragraph that is too long on its own.
3. A hard cut at `max_chars`, for a single sentence that is still too long.
"""

import re
from dataclasses import dataclass

from voicelm.domain.models import Chunk, Source

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# A sentence ends at ".", "!" or "?" followed by whitespace. The lookbehind keeps the
# punctuation attached to the sentence it belongs to.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

Span = tuple[int, int]


@dataclass(frozen=True)
class ChunkingConfig:
    """Tuning knobs for chunking.

    `max_chars` bounds how much *new* text a chunk may contain. Overlap is added by
    extending a chunk's start backwards, so a chunk's final length can reach
    `max_chars + overlap_chars`.
    """

    max_chars: int = 1000
    overlap_chars: int = 150

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")


def chunk_document(source: Source, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Split `source.text` into chunks, preserving provenance for each one."""
    config = config or ChunkingConfig()
    text = source.text

    if not text.strip():
        return []

    spans = _paragraph_spans(text)
    spans = _split_oversized(text, spans, config.max_chars)
    spans = _pack(spans, config.max_chars)
    spans = _add_overlap(spans, config.overlap_chars)

    return [
        Chunk(
            id=f"{source.id}:{ordinal}",
            source_id=source.id,
            text=text[start:end],
            start_char=start,
            end_char=end,
            ordinal=ordinal,
        )
        for ordinal, (start, end) in enumerate(spans)
    ]


def _paragraph_spans(text: str) -> list[Span]:
    """Spans of the paragraphs in `text`, excluding the blank lines between them."""
    spans: list[Span] = []
    position = 0

    for separator in _PARAGRAPH_BREAK.finditer(text):
        if separator.start() > position:
            spans.append((position, separator.start()))
        position = separator.end()

    if position < len(text):
        spans.append((position, len(text)))

    return spans


def _sentence_spans(text: str, start: int, end: int) -> list[Span]:
    """Spans of the sentences within `text[start:end]`."""
    spans: list[Span] = []
    position = start

    for separator in _SENTENCE_END.finditer(text, start, end):
        if separator.start() > position:
            spans.append((position, separator.start()))
        position = separator.end()

    if position < end:
        spans.append((position, end))

    return spans


def _split_oversized(text: str, spans: list[Span], max_chars: int) -> list[Span]:
    """Break any span longer than `max_chars` at sentence ends, then by brute force."""
    result: list[Span] = []

    for start, end in spans:
        if end - start <= max_chars:
            result.append((start, end))
            continue

        for sentence_start, sentence_end in _pack(_sentence_spans(text, start, end), max_chars):
            if sentence_end - sentence_start <= max_chars:
                result.append((sentence_start, sentence_end))
            else:
                result.extend(_hard_split(sentence_start, sentence_end, max_chars))

    return result


def _pack(spans: list[Span], max_chars: int) -> list[Span]:
    """Greedily merge consecutive spans while they fit within `max_chars`.

    Merging by span rather than by string means the text between two merged spans (the
    blank line separating two paragraphs) stays part of the chunk, so the result is still
    one contiguous slice of the document.
    """
    packed: list[Span] = []
    current: Span | None = None

    for start, end in spans:
        if current is None:
            current = (start, end)
        elif end - current[0] <= max_chars:
            current = (current[0], end)
        else:
            packed.append(current)
            current = (start, end)

    if current is not None:
        packed.append(current)

    return packed


def _hard_split(start: int, end: int, max_chars: int) -> list[Span]:
    """Last resort: cut a span at fixed intervals, mid-word if necessary."""
    return [(cut, min(cut + max_chars, end)) for cut in range(start, end, max_chars)]


def _add_overlap(spans: list[Span], overlap_chars: int) -> list[Span]:
    """Extend each chunk backwards so it repeats the tail of its predecessor.

    Without this, a fact sitting on a chunk boundary is cut in half and neither chunk
    contains it whole. Extending the *start* rather than concatenating text keeps each
    chunk a contiguous slice, so offsets stay meaningful.
    """
    if overlap_chars == 0:
        return spans

    result: list[Span] = [spans[0]] if spans else []

    for index in range(1, len(spans)):
        start, end = spans[index]
        previous_start = spans[index - 1][0]
        # Never reach past the start of the previous chunk, which would make this chunk a
        # superset of it.
        result.append((max(start - overlap_chars, previous_start), end))

    return result
