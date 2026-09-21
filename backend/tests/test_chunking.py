from dataclasses import FrozenInstanceError
from itertools import pairwise
from pathlib import Path

import pytest

from voicelm.domain.models import Source
from voicelm.ingestion.chunking import ChunkingConfig, chunk_document


def make_source(text: str, source_id: str = "src1") -> Source:
    return Source(id=source_id, path=Path("memo.md"), title="memo", text=text, content_hash="h")


def paragraphs(count: int, length: int = 300) -> str:
    return "\n\n".join(f"Paragraph {index}. " + "x" * length for index in range(count))


# --- the invariant that makes citations trustworthy -------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "short document",
        paragraphs(2),
        paragraphs(10),
        "One sentence. " * 400,  # forces sentence-level splitting
        "y" * 5000,  # forces hard splitting
    ],
)
def test_offsets_always_slice_back_to_the_chunk_text(text: str) -> None:
    source = make_source(text)

    for chunk in chunk_document(source):
        assert source.text[chunk.start_char : chunk.end_char] == chunk.text


# --- coverage and ordering ---------------------------------------------------------


def test_chunks_lose_no_content() -> None:
    config = ChunkingConfig(max_chars=1000, overlap_chars=0)
    source = make_source(paragraphs(12))
    chunks = chunk_document(source, config)

    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(source.text)

    # Chunks do not tile the document exactly: the blank line between two paragraphs is
    # dropped, so that chunk text never begins or ends with a decorative gap. What must
    # hold is that nothing *meaningful* falls between two chunks.
    for previous, current in pairwise(chunks):
        assert current.end_char > previous.end_char
        assert source.text[previous.end_char : current.start_char].strip() == ""


def test_ordinals_are_sequential_from_zero() -> None:
    chunks = chunk_document(make_source(paragraphs(8)))

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique_and_derived_from_the_source() -> None:
    chunks = chunk_document(make_source(paragraphs(8), source_id="abc123"))
    ids = [chunk.id for chunk in chunks]

    assert len(ids) == len(set(ids))
    assert all(chunk.id.startswith("abc123:") for chunk in chunks)
    assert all(chunk.source_id == "abc123" for chunk in chunks)


# --- size limits ------------------------------------------------------------------


def test_chunks_respect_the_size_budget() -> None:
    config = ChunkingConfig(max_chars=500, overlap_chars=50)
    chunks = chunk_document(make_source(paragraphs(20)), config)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= config.max_chars + config.overlap_chars


def test_a_document_smaller_than_the_limit_becomes_one_chunk() -> None:
    source = make_source("A single short note.\n\nWith two paragraphs.")
    chunks = chunk_document(source)

    assert len(chunks) == 1
    assert chunks[0].text == source.text


# --- split points -----------------------------------------------------------------


def test_prefers_paragraph_boundaries() -> None:
    # Three 300-char paragraphs with a 700-char budget: the first two fit together,
    # the third starts a new chunk.
    config = ChunkingConfig(max_chars=700, overlap_chars=0)
    chunks = chunk_document(make_source(paragraphs(3, length=280)), config)

    assert len(chunks) == 2
    assert chunks[1].text.startswith("Paragraph 2.")


def test_splits_an_oversized_paragraph_at_sentence_ends() -> None:
    long_paragraph = "This is a sentence of moderate length. " * 60
    config = ChunkingConfig(max_chars=400, overlap_chars=0)

    chunks = chunk_document(make_source(long_paragraph.strip()), config)

    assert len(chunks) > 1
    # Splitting at sentence ends means chunks finish on punctuation, not mid-word.
    assert all(chunk.text.rstrip().endswith(".") for chunk in chunks)


def test_hard_splits_a_single_sentence_that_is_too_long() -> None:
    config = ChunkingConfig(max_chars=100, overlap_chars=0)
    chunks = chunk_document(make_source("z" * 450), config)

    assert len(chunks) == 5
    assert all(len(chunk.text) <= 100 for chunk in chunks)


# --- overlap ----------------------------------------------------------------------


def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    config = ChunkingConfig(max_chars=400, overlap_chars=100)
    source = make_source(paragraphs(10, length=350))
    chunks = chunk_document(source, config)

    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        assert current.start_char < previous.end_char
        repeated = source.text[current.start_char : previous.end_char]
        assert previous.text.endswith(repeated)
        assert current.text.startswith(repeated)


def test_zero_overlap_repeats_nothing() -> None:
    config = ChunkingConfig(max_chars=400, overlap_chars=0)
    chunks = chunk_document(make_source(paragraphs(10, length=350)), config)

    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        assert current.start_char >= previous.end_char


# --- edge cases and validation ----------------------------------------------------


def test_empty_document_produces_no_chunks() -> None:
    assert chunk_document(make_source("")) == []
    assert chunk_document(make_source("   \n\n ")) == []


def test_no_chunk_is_empty() -> None:
    chunks = chunk_document(make_source(paragraphs(15)))

    assert all(chunk.text.strip() for chunk in chunks)


def test_chunking_is_deterministic() -> None:
    source = make_source(paragraphs(10))

    assert chunk_document(source) == chunk_document(source)


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [(0, 0), (-1, 0), (100, -5), (100, 100), (100, 200)],
)
def test_invalid_configuration_is_rejected(max_chars: int, overlap_chars: int) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(max_chars=max_chars, overlap_chars=overlap_chars)


def test_chunks_are_immutable() -> None:
    chunk = chunk_document(make_source("some text"))[0]

    with pytest.raises(FrozenInstanceError):
        chunk.text = "tampered"  # type: ignore[misc]
