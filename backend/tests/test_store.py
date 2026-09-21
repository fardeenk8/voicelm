from pathlib import Path

import pytest

from voicelm.domain.models import Chunk, EmbeddedChunk, Source
from voicelm.ingestion.chunking import chunk_document
from voicelm.retrieval.store import InMemoryVectorStore


def make_chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(
        id=chunk_id,
        source_id="src",
        text=text,
        start_char=0,
        end_char=len(text),
        ordinal=0,
    )


def embedded(
    chunk_id: str,
    vector: tuple[float, ...],
    model: str = "nomic-embed-text",
) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=make_chunk(chunk_id), vector=vector, model=model)


def test_empty_store_returns_nothing() -> None:
    store = InMemoryVectorStore()

    assert len(store) == 0
    assert store.dimensions is None
    assert store.search([1.0, 0.0]) == []


def test_search_ranks_by_similarity() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            embedded("far", (0.0, 1.0)),
            embedded("near", (1.0, 0.1)),
            embedded("middling", (1.0, 1.0)),
        ]
    )

    results = store.search([1.0, 0.0], top_k=3)

    assert [result.chunk.id for result in results] == ["near", "middling", "far"]
    assert results[0].score > results[1].score > results[2].score


def test_top_k_limits_the_number_of_results() -> None:
    store = InMemoryVectorStore()
    store.add([embedded(str(index), (1.0, float(index))) for index in range(10)])

    assert len(store.search([1.0, 0.0], top_k=3)) == 3


def test_top_k_larger_than_the_store_returns_everything() -> None:
    store = InMemoryVectorStore()
    store.add([embedded("a", (1.0, 0.0)), embedded("b", (0.0, 1.0))])

    assert len(store.search([1.0, 0.0], top_k=50)) == 2


def test_top_k_must_be_positive() -> None:
    store = InMemoryVectorStore()
    store.add([embedded("a", (1.0, 0.0))])

    with pytest.raises(ValueError, match="at least 1"):
        store.search([1.0, 0.0], top_k=0)


def test_equal_scores_keep_insertion_order() -> None:
    store = InMemoryVectorStore()
    store.add([embedded("first", (1.0, 0.0)), embedded("second", (2.0, 0.0))])

    # Same direction, different magnitude, so cosine similarity is identical.
    results = store.search([1.0, 0.0], top_k=2)

    assert [result.chunk.id for result in results] == ["first", "second"]


# --- the ADR-0008 guards ----------------------------------------------------------


def test_rejects_vectors_from_a_different_model() -> None:
    store = InMemoryVectorStore()
    store.add([embedded("a", (1.0, 0.0), model="nomic-embed-text")])

    with pytest.raises(ValueError, match="not comparable"):
        store.add([embedded("b", (0.0, 1.0), model="bge-small-en-v1.5")])


def test_rejects_vectors_of_a_different_length() -> None:
    store = InMemoryVectorStore()
    store.add([embedded("a", (1.0, 0.0))])

    with pytest.raises(ValueError, match="dimensional"):
        store.add([embedded("b", (0.0, 1.0, 0.0))])


def test_rejects_a_query_of_the_wrong_length() -> None:
    store = InMemoryVectorStore()
    store.add([embedded("a", (1.0, 0.0))])

    with pytest.raises(ValueError, match="different model"):
        store.search([1.0, 0.0, 0.0])


# --- integration with real chunks --------------------------------------------------


def test_results_carry_provenance_back_to_the_source() -> None:
    text = "Alpha paragraph about caching.\n\nBeta paragraph about networking."
    source = Source(id="src9", path=Path("notes.md"), title="notes", text=text)
    chunks = chunk_document(source)

    store = InMemoryVectorStore()
    store.add(
        [
            EmbeddedChunk(chunk=chunk, vector=(1.0, float(index)), model="fake")
            for index, chunk in enumerate(chunks)
        ]
    )

    result = store.search([1.0, 0.0], top_k=1)[0]

    # The offsets must still slice back out of the original document, which is what makes
    # a citation verifiable rather than a claim.
    assert source.text[result.chunk.start_char : result.chunk.end_char] == result.chunk.text
    assert result.chunk.source_id == "src9"
