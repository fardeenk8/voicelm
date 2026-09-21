"""Turning text into vectors with a local Ollama model.

Ollama runs as an HTTP server on the machine, so this module is an ordinary HTTP client.
The endpoint accepts a list of strings and returns one vector per string:

    POST /api/embed  {"model": "nomic-embed-text", "input": ["a", "b"]}
    -> {"model": ..., "embeddings": [[...768 floats...], [...]], ...}
"""

from collections.abc import Sequence

import httpx2

from voicelm.domain.models import Chunk, EmbeddedChunk

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT_SECONDS = 120.0


class EmbeddingError(Exception):
    """Raised when Ollama cannot produce usable embeddings."""


class OllamaEmbedder:
    """Embeds text using a model served by a local Ollama instance."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx2.Client | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model = model
        self.batch_size = batch_size
        # An injectable client keeps the tests offline: they pass one wired to a mock
        # transport instead of a real socket.
        self._client = client or httpx2.Client(base_url=base_url, timeout=timeout)

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed each string, preserving order."""
        if not texts:
            return []

        for position, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"text at position {position} is empty")

        vectors: list[tuple[float, ...]] = []
        # Sent in batches so that a large document does not become one enormous request
        # that times out halfway through.
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))

        distinct_dimensions = {len(vector) for vector in vectors}
        if len(distinct_dimensions) > 1:
            raise EmbeddingError(
                f"model {self.model} returned vectors of differing lengths: "
                f"{sorted(distinct_dimensions)}"
            )

        return vectors

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.embed_texts([text])[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        vectors = self.embed_texts([chunk.text for chunk in chunks])

        return [
            EmbeddedChunk(chunk=chunk, vector=vector, model=self.model)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def _embed_batch(self, batch: Sequence[str]) -> list[tuple[float, ...]]:
        try:
            response = self._client.post(
                "/api/embed", json={"model": self.model, "input": list(batch)}
            )
            response.raise_for_status()
        except httpx2.HTTPStatusError as error:
            raise EmbeddingError(
                f"Ollama returned {error.response.status_code}: {error.response.text}"
            ) from error
        except httpx2.RequestError as error:
            raise EmbeddingError(
                f"could not reach Ollama ({error}). Is it running? "
                "Start it with: brew services start ollama"
            ) from error

        payload = response.json()
        embeddings = payload.get("embeddings")

        if not isinstance(embeddings, list):
            raise EmbeddingError(f"unexpected response from Ollama; keys were {sorted(payload)}")

        # Guards against silent truncation: a missing vector would otherwise be paired
        # with the wrong chunk and every citation downstream would point at the wrong text.
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"asked for {len(batch)} embeddings but received {len(embeddings)}"
            )

        return [tuple(float(value) for value in vector) for vector in embeddings]
