from pathlib import Path

import pytest

from voicelm.domain.models import Chunk, EmbeddedChunk
from voicelm.retrieval.store import InMemoryVectorStore
from voicelm.storage.qdrant import QdrantVectorIndex


def make_chunk(chunk_id: str, source_id: str = "src") -> Chunk:
    return Chunk(
        id=chunk_id,
        source_id=source_id,
        text=chunk_id,
        start_char=0,
        end_char=len(chunk_id),
        ordinal=0,
    )


def embedded(
    chunk_id: str,
    vector: tuple[float, ...],
    model: str = "nomic-embed-text",
    source_id: str = "src",
) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=make_chunk(chunk_id, source_id), vector=vector, model=model)


@pytest.fixture
def index(tmp_path: Path):
    idx = QdrantVectorIndex(tmp_path / "qdrant")
    yield idx
    idx.close()


def test_empty_index_returns_nothing(index: QdrantVectorIndex) -> None:
    assert len(index) == 0
    assert index.dimensions is None
    assert index.search([1.0, 0.0]) == []


def test_search_ranks_by_similarity(index: QdrantVectorIndex) -> None:
    index.add(
        [
            embedded("far", (0.0, 1.0)),
            embedded("near", (1.0, 0.0)),
            embedded("middling", (0.9, 0.4)),
        ]
    )

    hits = index.search([1.0, 0.0], top_k=3)

    assert [hit.chunk_id for hit in hits] == ["near", "middling", "far"]
    assert hits[0].score > hits[1].score > hits[2].score
    assert hits[0].score == pytest.approx(1.0)


def test_top_k_limits_results(index: QdrantVectorIndex) -> None:
    index.add([embedded(str(i), (1.0, float(i))) for i in range(10)])

    assert len(index.search([1.0, 0.0], top_k=3)) == 3


def test_rejects_a_different_model(index: QdrantVectorIndex) -> None:
    index.add([embedded("a", (1.0, 0.0), model="nomic-embed-text")])

    with pytest.raises(ValueError, match="not comparable"):
        index.add([embedded("b", (0.0, 1.0), model="bge-small-en-v1.5")])


def test_rejects_a_different_length(index: QdrantVectorIndex) -> None:
    index.add([embedded("a", (1.0, 0.0))])

    with pytest.raises(ValueError, match="dimensional"):
        index.add([embedded("b", (0.0, 1.0, 0.0))])


def test_rejects_a_query_of_the_wrong_length(index: QdrantVectorIndex) -> None:
    index.add([embedded("a", (1.0, 0.0))])

    with pytest.raises(ValueError, match="different model"):
        index.search([1.0, 0.0, 0.0])


def test_delete_by_source_removes_only_that_source(index: QdrantVectorIndex) -> None:
    index.add(
        [
            embedded("keep", (1.0, 0.0), source_id="keep"),
            embedded("drop", (0.0, 1.0), source_id="drop"),
        ]
    )
    index.delete_by_source("drop")

    hits = index.search([1.0, 0.0], top_k=5)

    assert [hit.chunk_id for hit in hits] == ["keep"]
    assert len(index) == 1


def test_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "qdrant"
    first = QdrantVectorIndex(path)
    first.add([embedded("near", (1.0, 0.0))])
    first.close()

    second = QdrantVectorIndex(path)
    hits = second.search([1.0, 0.0], top_k=1)
    second.close()

    assert len(hits) == 1
    assert hits[0].chunk_id == "near"
    assert hits[0].score == pytest.approx(1.0)


def test_does_not_store_chunk_text(index: QdrantVectorIndex) -> None:
    # The payload must not grow a copy of the document. SQLite is the canonical text.
    index.add([embedded("near", (1.0, 0.0))])
    records, _ = index._client.scroll("chunks", limit=1, with_payload=True)

    assert "text" not in records[0].payload
    assert records[0].payload["chunk_id"] == "near"


def test_ranking_matches_the_brute_force_store(index: QdrantVectorIndex) -> None:
    """The reason we kept InMemoryVectorStore (ADR-0017): Qdrant must agree with it."""
    entries = [
        embedded("a", (1.0, 0.0, 0.0)),
        embedded("b", (0.8, 0.6, 0.0)),
        embedded("c", (0.0, 1.0, 0.0)),
        embedded("d", (0.0, 0.0, 1.0)),
        embedded("e", (0.5, 0.5, 0.7)),
    ]
    query = (0.9, 0.2, 0.1)

    memory = InMemoryVectorStore()
    memory.add(entries)
    index.add(entries)

    expected = [result.chunk.id for result in memory.search(query, top_k=5)]
    actual = [hit.chunk_id for hit in index.search(query, top_k=5)]

    assert actual == expected
