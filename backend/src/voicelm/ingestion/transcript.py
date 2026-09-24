"""Timed speech (captions or STT) → the same PageSpan map PDFs use.

`PageSpan.number` is seconds from the start of the media (0, 15, 92, …). The UI
formats that as `0:00`, `0:15`, `1:32`. Chunking stays format-blind.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from voicelm.domain.models import PageSpan
from voicelm.ingestion.cleaning import clean_text
from voicelm.ingestion.pdf import PAGE_SEPARATOR


@dataclass(frozen=True)
class TimedSegment:
    """One caption cue or Whisper segment."""

    start: float
    text: str


def format_timestamp(seconds: int) -> str:
    """Render a whole-second offset as `m:ss` or `h:mm:ss`."""
    if seconds < 0:
        seconds = 0
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def assemble_timed_segments(
    segments: Sequence[TimedSegment],
) -> tuple[str, tuple[PageSpan, ...]]:
    """Clean each cue, join them, and record where each one landed.

    Empty cues get no span. `PageSpan.number` keeps the wall-clock second so a later
    blank does not renumber anything.
    """
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0

    for segment in segments:
        cleaned = clean_text(segment.text)
        if not cleaned:
            continue

        if parts:
            cursor += len(PAGE_SEPARATOR)
        number = max(0, int(segment.start))
        spans.append(PageSpan(number=number, start_char=cursor, end_char=cursor + len(cleaned)))
        parts.append(cleaned)
        cursor += len(cleaned)

    return PAGE_SEPARATOR.join(parts), tuple(spans)
