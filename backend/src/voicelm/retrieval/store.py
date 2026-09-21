"""A brute-force vector store.

Every search compares the query against every stored vector. That is deliberate
(ADR-0017): it is fifteen lines, it is fast enough for hundreds of chunks, and it makes
plain what a vector database actually does before we depend on one. When Qdrant arrives,
this stays as the reference implementation its results are checked against.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from voicelm.domain.models import Chunk, EmbeddedChunk
from voicelm.retrieval.similarity import cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: list[EmbeddedChunk] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def dimensions(self) -> int | None:
        """Vector length this store holds, or None while it is empty."""
        return self._entries[0].dimensions if self._entries else None

    @property
    def model(self) -> str | None:
        return self._entries[0].model if self._entries else None

    def add(self, embedded: Iterable[EmbeddedChunk]) -> None:
        """Store embedded chunks, rejecting anything inconsistent with what is here.

        The two checks below are ADR-0008 doing its job. Vectors from different models
        occupy different spaces, so comparing them produces confident nonsense rather than
        an error. Refusing the write is the only way to catch it.
        """
        for entry in embedded:
            if self._entries:
                if entry.model != self._entries[0].model:
                    raise ValueError(
                        f"store holds vectors from {self._entries[0].model!r} but got "
                        f"{entry.model!r}; vectors from different models are not comparable"
                    )
                if entry.dimensions != self._entries[0].dimensions:
                    raise ValueError(
                        f"store holds {self._entries[0].dimensions}-dimensional vectors "
                        f"but got {entry.dimensions}"
                    )
            self._entries.append(entry)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        """Return the `top_k` most similar chunks, most similar first."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self._entries:
            return []
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"query has {len(query_vector)} dimensions but the store holds "
                f"{self.dimensions}; was it embedded with a different model?"
            )

        results = [
            SearchResult(chunk=entry.chunk, score=cosine_similarity(query_vector, entry.vector))
            for entry in self._entries
        ]

        # Python's sort is stable, so equal scores keep insertion order and results are
        # reproducible rather than arbitrary.
        results.sort(key=lambda result: result.score, reverse=True)

        return results[:top_k]
