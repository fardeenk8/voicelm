"""The join between the catalog (SQLite) and the vector index (Qdrant).

Ingestion, skip-if-unchanged, search, and answering all go through here so neither
store is written from a route handler or a command function.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from voicelm.domain.models import Answer, Chunk, EmbeddedChunk, Source
from voicelm.generation.answering import answer_question
from voicelm.generation.ollama import OllamaChatModel
from voicelm.ingestion.chunking import ChunkingConfig, chunk_document
from voicelm.ingestion.loader import load_source
from voicelm.retrieval.store import SearchResult
from voicelm.storage.qdrant import QdrantVectorIndex
from voicelm.storage.sqlite import IngestMeta, SourceRecord, SqliteMetadataStore

IngestStatus = Literal["ingested", "updated", "skipped"]


class Embedder(Protocol):
    """Anything that turns text into vectors. Tests pass a fake; production uses Ollama."""

    model: str

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class IngestResult:
    source: Source
    chunk_count: int
    status: IngestStatus


class KnowledgeBase:
    def __init__(
        self,
        data_dir: Path,
        embedder: Embedder,
        chunking: ChunkingConfig | None = None,
        catalog: SqliteMetadataStore | None = None,
        index: QdrantVectorIndex | None = None,
    ) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._chunking = chunking or ChunkingConfig()
        self._catalog = catalog or SqliteMetadataStore(data_dir / "voicelm.db")
        self._index = index or QdrantVectorIndex(data_dir / "qdrant")

    def close(self) -> None:
        self._catalog.close()
        self._index.close()

    def ingest(self, path: Path) -> IngestResult:
        """Add or refresh one document.

        Same path, same hash, complete index: do nothing.
        Same path, same hash, missing vectors: re-embed (repair after a crashed write).
        Same path, different hash: keep the id, replace text, chunks, and vectors.
        New path: mint the loader's UUID as the id.
        """
        loaded = load_source(path)
        existing = self._catalog.get_by_path(path)

        if (
            existing is not None
            and existing.source.content_hash == loaded.content_hash
            and self._index_is_complete(existing.source.id)
        ):
            chunks = self._catalog.get_chunks(existing.source.id)
            return IngestResult(existing.source, len(chunks), "skipped")

        if existing is None:
            source_id = loaded.id
            status: IngestStatus = "ingested"
        else:
            source_id = existing.source.id
            status = "updated"

        source = Source(
            id=source_id,
            path=path,
            title=loaded.title,
            text=loaded.text,
            content_hash=loaded.content_hash,
        )
        chunks = chunk_document(source, self._chunking)
        embedded = self._embedder.embed_chunks(chunks) if chunks else []
        meta = IngestMeta(
            embedding_model=self._embedder.model,
            chunk_max_chars=self._chunking.max_chars,
            chunk_overlap=self._chunking.overlap_chars,
        )

        was_new = existing is None
        try:
            # Catalog first (source of truth), then the index. Embeddings are already in
            # memory, so a failed Ollama call never leaves a half-written document.
            self._catalog.save(source, chunks, meta)
            self._index.delete_by_source(source.id)
            if embedded:
                self._index.add(embedded)
        except Exception:
            self._index.delete_by_source(source.id)
            if was_new:
                self._catalog.delete(source.id)
            raise

        return IngestResult(source, len(chunks), status)

    def search(self, question: str, top_k: int = 5) -> tuple[list[SearchResult], dict[str, Source]]:
        """Qdrant finds ids; SQLite loads the passages. That join is this method."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        hits = self._index.search(self._embedder.embed_query(question), top_k=top_k)
        if not hits:
            return [], {}

        chunks = self._catalog.get_chunks_by_ids([hit.chunk_id for hit in hits])
        results = [
            SearchResult(chunk=chunk, score=hit.score)
            for chunk, hit in zip(chunks, hits, strict=True)
        ]

        sources: dict[str, Source] = {}
        for result in results:
            source_id = result.chunk.source_id
            if source_id in sources:
                continue
            record = self._catalog.get(source_id)
            if record is None:
                raise KeyError(f"chunk {result.chunk.id} refers to missing source {source_id}")
            sources[source_id] = record.source

        return results, sources

    def ask(self, question: str, model: OllamaChatModel, top_k: int = 5) -> Answer:
        results, sources = self.search(question, top_k=top_k)
        return answer_question(question, results, sources, model)

    def list_sources(self) -> list[SourceRecord]:
        return self._catalog.list_sources()

    def chunk_count(self, source_id: str) -> int:
        return len(self._catalog.get_chunks(source_id))

    def _index_is_complete(self, source_id: str) -> bool:
        return self._index.count_by_source(source_id) == len(self._catalog.get_chunks(source_id))
