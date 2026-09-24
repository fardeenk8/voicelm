"""JSON shapes the HTTP API speaks.

These are the contract a Flutter client will generate code against. They are serialised
views of domain objects, not a second source of truth.
"""

from typing import Literal

from pydantic import BaseModel, Field

from voicelm.domain.models import Answer, Citation, location_kind_for
from voicelm.knowledge import IngestResult
from voicelm.storage.sqlite import SourceRecord


class SourceOut(BaseModel):
    id: str
    title: str
    path: str
    chunk_count: int
    # Count of location units (PDF pages, slides, or paragraphs). 0 for plain text / web.
    page_count: int
    location_kind: Literal["page", "slide", "paragraph", "timestamp", "line"] | None = None
    origin_url: str | None = None


class IngestOut(SourceOut):
    status: Literal["ingested", "updated", "skipped"]


class CitationOut(BaseModel):
    marker: int
    source_id: str
    source_title: str
    start_char: int
    end_char: int
    pages: list[int]
    quote: str
    location_kind: Literal["page", "slide", "paragraph", "timestamp", "line"] | None = None
    origin_url: str | None = None


class AskIn(BaseModel):
    question: str
    top_k: int = Field(default=10, ge=1, le=20)


class UrlIn(BaseModel):
    url: str


class AskOut(BaseModel):
    text: str
    model: str
    citations: list[CitationOut]
    unsupported_markers: list[int]


def source_out(record: SourceRecord, chunk_count: int) -> SourceOut:
    return SourceOut(
        id=record.source.id,
        title=record.source.title,
        path=str(record.source.path),
        chunk_count=chunk_count,
        page_count=len(record.source.pages),
        location_kind=location_kind_for(record.source.path),
        origin_url=record.source.origin_url,
    )


def ingest_out(result: IngestResult) -> IngestOut:
    return IngestOut(
        id=result.source.id,
        title=result.source.title,
        path=str(result.source.path),
        chunk_count=result.chunk_count,
        page_count=len(result.source.pages),
        status=result.status,
        location_kind=location_kind_for(result.source.path),
        origin_url=result.source.origin_url,
    )


def citation_out(citation: Citation) -> CitationOut:
    return CitationOut(
        marker=citation.marker,
        source_id=citation.source_id,
        source_title=citation.source_title,
        start_char=citation.start_char,
        end_char=citation.end_char,
        pages=list(citation.pages),
        quote=citation.quote,
        location_kind=citation.location_kind,
        origin_url=citation.origin_url,
    )


def ask_out(answer: Answer) -> AskOut:
    return AskOut(
        text=answer.text,
        model=answer.model,
        citations=[citation_out(citation) for citation in answer.citations],
        unsupported_markers=list(answer.unsupported_markers),
    )
