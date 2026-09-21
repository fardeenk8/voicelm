"""Offline tests for the Ollama client.

These never touch the network. `httpx2.MockTransport` intercepts requests and lets us
return whatever response we want, including malformed ones that a real server would be
hard to coax into producing.
"""

import httpx2
import pytest

from voicelm.domain.models import Chunk
from voicelm.embeddings.ollama import EmbeddingError, OllamaEmbedder


def embedder_returning(
    payload: dict | list, status_code: int = 200
) -> tuple[OllamaEmbedder, list[dict]]:
    """Build an embedder whose server always answers with `payload`.

    Also returns a list that records each request body, so tests can assert on what was
    actually sent — for example that batching produced more than one request.
    """
    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        import json

        sent.append(json.loads(request.content))
        return httpx2.Response(status_code, json=payload)

    client = httpx2.Client(base_url="http://testserver", transport=httpx2.MockTransport(handler))
    return OllamaEmbedder(client=client), sent


def test_returns_one_vector_per_text() -> None:
    embedder, sent = embedder_returning({"embeddings": [[1.0, 2.0], [3.0, 4.0]]})

    vectors = embedder.embed_texts(["first", "second"])

    assert vectors == [(1.0, 2.0), (3.0, 4.0)]
    assert sent == [{"model": "nomic-embed-text", "input": ["first", "second"]}]


def test_embed_query_returns_a_single_vector() -> None:
    embedder, _ = embedder_returning({"embeddings": [[0.5, 0.5]]})

    assert embedder.embed_query("question") == (0.5, 0.5)


def test_no_request_is_made_for_an_empty_list() -> None:
    embedder, sent = embedder_returning({"embeddings": []})

    assert embedder.embed_texts([]) == []
    assert sent == []


def test_blank_text_is_rejected_before_calling_ollama() -> None:
    embedder, sent = embedder_returning({"embeddings": [[1.0]]})

    with pytest.raises(ValueError, match="position 1"):
        embedder.embed_texts(["fine", "   "])

    assert sent == []


def test_large_input_is_split_into_batches() -> None:
    embedder, sent = embedder_returning({"embeddings": [[1.0]]})
    embedder.batch_size = 1

    embedder.embed_texts(["a", "b", "c"])

    assert len(sent) == 3
    assert [body["input"] for body in sent] == [["a"], ["b"], ["c"]]


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        OllamaEmbedder(batch_size=0)


def test_embed_chunks_attaches_the_model_and_the_chunk() -> None:
    embedder, _ = embedder_returning({"embeddings": [[1.0, 2.0]]})
    chunk = Chunk(id="src:0", source_id="src", text="body", start_char=0, end_char=4, ordinal=0)

    [result] = embedder.embed_chunks([chunk])

    assert result.chunk is chunk
    assert result.vector == (1.0, 2.0)
    assert result.model == "nomic-embed-text"
    assert result.dimensions == 2


# --- failure modes ----------------------------------------------------------------


def test_missing_vector_is_an_error_not_a_silent_mismatch() -> None:
    # If this were tolerated, the second text's chunk would be paired with no vector or
    # the wrong one, and every citation downstream would point at the wrong text.
    embedder, _ = embedder_returning({"embeddings": [[1.0, 2.0]]})

    with pytest.raises(EmbeddingError, match="asked for 2 embeddings"):
        embedder.embed_texts(["first", "second"])


def test_inconsistent_dimensions_are_an_error() -> None:
    embedder, _ = embedder_returning({"embeddings": [[1.0, 2.0], [3.0]]})

    with pytest.raises(EmbeddingError, match="differing lengths"):
        embedder.embed_texts(["first", "second"])


def test_unexpected_response_shape_is_an_error() -> None:
    embedder, _ = embedder_returning({"error": "model not found"})

    with pytest.raises(EmbeddingError, match="unexpected response"):
        embedder.embed_texts(["text"])


def test_http_error_status_is_wrapped() -> None:
    embedder, _ = embedder_returning({"error": "no such model"}, status_code=404)

    with pytest.raises(EmbeddingError, match="404"):
        embedder.embed_texts(["text"])


def test_connection_failure_explains_how_to_fix_it() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    client = httpx2.Client(base_url="http://testserver", transport=httpx2.MockTransport(handler))
    embedder = OllamaEmbedder(client=client)

    with pytest.raises(EmbeddingError, match="brew services start ollama"):
        embedder.embed_texts(["text"])
