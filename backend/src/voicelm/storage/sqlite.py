"""SQLite catalog of sources and chunks.

Qdrant will store vectors; this module stores everything a citation needs: the cleaned
text, character offsets, title, and path. SQLite is a library, not a server — there is no
port to bind and nothing to start. Python opens one file.

Foreign keys are off in SQLite unless you turn them on *per connection*. WAL mode lets a
reader search while a writer is ingesting.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from voicelm.domain.models import Chunk, PageSpan, Source

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_max_chars INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    ordinal INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_source_id ON chunks(source_id);

CREATE TABLE IF NOT EXISTS pages (
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    number INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    PRIMARY KEY (source_id, number)
);
"""


class DuplicatePathError(Exception):
    """Raised when saving a source whose path already belongs to a different id."""


@dataclass(frozen=True)
class IngestMeta:
    embedding_model: str
    chunk_max_chars: int
    chunk_overlap: int
    ingested_at: str | None = None


@dataclass(frozen=True)
class SourceRecord:
    """A source plus the processing parameters used when it was ingested."""

    source: Source
    embedding_model: str
    chunk_max_chars: int
    chunk_overlap: int
    ingested_at: str


class SqliteMetadataStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs `def` routes in a thread pool. The connection is opened during
        # lifespan on the main thread, then used on a worker — SQLite forbids that
        # unless we opt in. This process is single-user; we are not sharing the
        # connection across concurrent writers.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def save(self, source: Source, chunks: Sequence[Chunk], meta: IngestMeta) -> None:
        """Insert or replace a source and its chunks in one transaction.

        Same id: update in place. Same path, different id: refuse, so two documents
        cannot silently claim one file.
        """
        for chunk in chunks:
            if chunk.source_id != source.id:
                raise ValueError(f"chunk {chunk.id} belongs to {chunk.source_id}, not {source.id}")

        existing = self.get_by_path(source.path)
        if existing is not None and existing.source.id != source.id:
            raise DuplicatePathError(f"{source.path} is already ingested as {existing.source.id}")

        ingested_at = meta.ingested_at or datetime.now(UTC).isoformat()
        path = str(source.path.resolve())

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO sources (
                    id, path, title, text, content_hash,
                    embedding_model, chunk_max_chars, chunk_overlap, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    title = excluded.title,
                    text = excluded.text,
                    content_hash = excluded.content_hash,
                    embedding_model = excluded.embedding_model,
                    chunk_max_chars = excluded.chunk_max_chars,
                    chunk_overlap = excluded.chunk_overlap,
                    ingested_at = excluded.ingested_at
                """,
                (
                    source.id,
                    path,
                    source.title,
                    source.text,
                    source.content_hash,
                    meta.embedding_model,
                    meta.chunk_max_chars,
                    meta.chunk_overlap,
                    ingested_at,
                ),
            )
            self._conn.execute("DELETE FROM chunks WHERE source_id = ?", (source.id,))
            self._conn.execute("DELETE FROM pages WHERE source_id = ?", (source.id,))
            self._conn.executemany(
                """
                INSERT INTO chunks (id, source_id, text, start_char, end_char, ordinal)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.source_id,
                        chunk.text,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.ordinal,
                    )
                    for chunk in chunks
                ],
            )
            self._conn.executemany(
                """
                INSERT INTO pages (source_id, number, start_char, end_char)
                VALUES (?, ?, ?, ?)
                """,
                [(source.id, span.number, span.start_char, span.end_char) for span in source.pages],
            )

    def get(self, source_id: str) -> SourceRecord | None:
        row = self._conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        return self._record_from_row(row) if row is not None else None

    def get_by_path(self, path: Path) -> SourceRecord | None:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE path = ?", (str(path.resolve()),)
        ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def list_sources(self) -> list[SourceRecord]:
        rows = self._conn.execute("SELECT * FROM sources ORDER BY ingested_at DESC").fetchall()
        return [self._record_from_row(row) for row in rows]

    def delete(self, source_id: str) -> bool:
        """Delete a source and its chunks. Returns False if the id was unknown."""
        with self._conn:
            cursor = self._conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        return cursor.rowcount > 0

    def get_chunks(self, source_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE source_id = ? ORDER BY ordinal", (source_id,)
        ).fetchall()
        return [_chunk_from_row(row) for row in rows]

    def get_chunks_by_ids(self, ids: Sequence[str]) -> list[Chunk]:
        """Load chunks, returning them in the same order as `ids`.

        Search ranks by similarity, so order is part of the contract. Missing ids mean
        Qdrant and SQLite have drifted — raise rather than silently dropping a hit.
        """
        if not ids:
            return []

        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", tuple(ids)
        ).fetchall()
        by_id = {row["id"]: _chunk_from_row(row) for row in rows}

        missing = [chunk_id for chunk_id in ids if chunk_id not in by_id]
        if missing:
            raise KeyError(f"chunks not in catalog: {missing}")

        return [by_id[chunk_id] for chunk_id in ids]

    def _record_from_row(self, row: sqlite3.Row) -> SourceRecord:
        source = Source(
            id=row["id"],
            path=Path(row["path"]),
            title=row["title"],
            text=row["text"],
            content_hash=row["content_hash"],
            pages=self._pages_for(row["id"]),
        )
        return SourceRecord(
            source=source,
            embedding_model=row["embedding_model"],
            chunk_max_chars=row["chunk_max_chars"],
            chunk_overlap=row["chunk_overlap"],
            ingested_at=row["ingested_at"],
        )

    def _pages_for(self, source_id: str) -> tuple[PageSpan, ...]:
        rows = self._conn.execute(
            "SELECT number, start_char, end_char FROM pages WHERE source_id = ? ORDER BY number",
            (source_id,),
        ).fetchall()
        return tuple(
            PageSpan(number=row["number"], start_char=row["start_char"], end_char=row["end_char"])
            for row in rows
        )


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        source_id=row["source_id"],
        text=row["text"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        ordinal=row["ordinal"],
    )
