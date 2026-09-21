"""Persistent vector index using Qdrant in local mode (ADR-0007).

Qdrant stores vectors and answers "which chunk ids are nearest this query?". It does
*not* store chunk text — that stays in SQLite so there is one canonical copy.

Point ids in Qdrant must be UUIDs or integers. Our chunk ids are strings like
`{source_uuid}:0`, so we derive a deterministic UUID from each chunk id and keep the
real id in the payload.
"""

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from voicelm.domain.models import EmbeddedChunk

COLLECTION = "chunks"


@dataclass(frozen=True)
class VectorHit:
    """A search hit: which chunk, from which source, how similar.

    Unlike `SearchResult`, this does not carry the chunk text. The catalog loads that.
    """

    chunk_id: str
    source_id: str
    score: float


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorIndex:
    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        # Local mode: the engine runs in this process against `path`. No server, no Docker.
        self._client = QdrantClient(path=str(path))

    def close(self) -> None:
        self._client.close()

    def __len__(self) -> int:
        if not self._client.collection_exists(COLLECTION):
            return 0
        return int(self._client.count(COLLECTION).count)

    @property
    def dimensions(self) -> int | None:
        if not self._client.collection_exists(COLLECTION):
            return None
        vectors = self._client.get_collection(COLLECTION).config.params.vectors
        return int(vectors.size)

    @property
    def model(self) -> str | None:
        if not self._client.collection_exists(COLLECTION) or len(self) == 0:
            return None
        records, _offset = self._client.scroll(COLLECTION, limit=1, with_payload=True)
        return str(records[0].payload["embedding_model"])

    def add(self, embedded: Iterable[EmbeddedChunk]) -> None:
        entries = list(embedded)
        if not entries:
            return

        model = entries[0].model
        dimensions = entries[0].dimensions

        for entry in entries:
            if entry.model != model:
                raise ValueError(
                    f"batch mixes models {model!r} and {entry.model!r}; "
                    "vectors from different models are not comparable"
                )
            if entry.dimensions != dimensions:
                raise ValueError(
                    f"batch mixes {dimensions}- and {entry.dimensions}-dimensional vectors"
                )

        stored_model = self.model
        if stored_model is not None and stored_model != model:
            raise ValueError(
                f"index holds vectors from {stored_model!r} but got {model!r}; "
                "vectors from different models are not comparable"
            )

        self._ensure_collection(dimensions)

        points = [
            PointStruct(
                id=_point_id(entry.chunk.id),
                vector=list(entry.vector),
                payload={
                    "chunk_id": entry.chunk.id,
                    "source_id": entry.chunk.source_id,
                    "embedding_model": entry.model,
                    "dimensions": entry.dimensions,
                },
            )
            for entry in entries
        ]
        self._client.upsert(COLLECTION, points)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[VectorHit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self._client.collection_exists(COLLECTION) or len(self) == 0:
            return []
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"query has {len(query_vector)} dimensions but the index holds "
                f"{self.dimensions}; was it embedded with a different model?"
            )

        response = self._client.query_points(
            COLLECTION, query=list(query_vector), limit=top_k, with_payload=True
        )
        return [
            VectorHit(
                chunk_id=str(point.payload["chunk_id"]),
                source_id=str(point.payload["source_id"]),
                score=float(point.score),
            )
            for point in response.points
        ]

    def delete_by_source(self, source_id: str) -> None:
        """Remove every vector belonging to a source. Used when that file is re-ingested."""
        if not self._client.collection_exists(COLLECTION):
            return
        self._client.delete(
            COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
        )

    def count_by_source(self, source_id: str) -> int:
        """How many vectors belong to this source. Used to detect a half-finished ingest."""
        if not self._client.collection_exists(COLLECTION):
            return 0
        result = self._client.count(
            COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
        )
        return int(result.count)

    def _ensure_collection(self, dimensions: int) -> None:
        if not self._client.collection_exists(COLLECTION):
            self._client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
            )
            return

        stored = self.dimensions
        if stored != dimensions:
            raise ValueError(f"index holds {stored}-dimensional vectors but got {dimensions}")
