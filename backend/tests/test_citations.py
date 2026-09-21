from pathlib import Path

from voicelm.domain.models import Chunk, Source
from voicelm.generation.citations import (
    extract_citations,
    invalid_markers,
    strip_unsupported_markers,
)
from voicelm.retrieval.store import SearchResult

SOURCES = {
    "src": Source(id="src", path=Path("notes.md"), title="notes", text="", content_hash="h"),
    "other": Source(id="other", path=Path("spec.txt"), title="spec", text="", content_hash="h"),
}


def make_result(chunk_id: str, text: str, source_id: str = "src") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            source_id=source_id,
            text=text,
            start_char=10,
            end_char=10 + len(text),
            ordinal=0,
        ),
        score=0.5,
    )


RESULTS = [
    make_result("src:0", "first excerpt"),
    make_result("src:1", "second excerpt"),
    make_result("other:0", "third excerpt", source_id="other"),
]


def test_extracts_a_citation_per_marker() -> None:
    citations = extract_citations("Caching helps [1] and so does batching [2].", RESULTS, SOURCES)

    assert [citation.marker for citation in citations] == [1, 2]
    assert [citation.chunk_id for citation in citations] == ["src:0", "src:1"]


def test_citation_details_come_from_our_records_not_the_model() -> None:
    [citation] = extract_citations("Answer [1].", RESULTS, SOURCES)

    assert citation.source_title == "notes"
    assert citation.quote == "first excerpt"
    assert citation.start_char == 10
    assert citation.end_char == 10 + len("first excerpt")


def test_citations_appear_in_order_of_first_mention() -> None:
    citations = extract_citations("Start [2], then [1].", RESULTS, SOURCES)

    assert [citation.marker for citation in citations] == [2, 1]


def test_repeated_markers_produce_one_citation() -> None:
    citations = extract_citations("Both [1] and again [1].", RESULTS, SOURCES)

    assert len(citations) == 1


def test_markers_can_span_multiple_sources() -> None:
    citations = extract_citations("From [1] and from [3].", RESULTS, SOURCES)

    assert [citation.source_title for citation in citations] == ["notes", "spec"]


def test_uncited_answer_yields_no_citations() -> None:
    assert extract_citations("No markers at all.", RESULTS, SOURCES) == ()


# --- invented markers -------------------------------------------------------------


def test_invented_markers_are_not_turned_into_citations() -> None:
    # Only three excerpts were provided, so [9] refers to nothing.
    citations = extract_citations("Real [1] and invented [9].", RESULTS, SOURCES)

    assert [citation.marker for citation in citations] == [1]


def test_invented_markers_are_reported() -> None:
    assert invalid_markers("Claim [1] and [7] and [12].", RESULTS) == (7, 12)


def test_zero_is_not_a_valid_marker() -> None:
    # Numbering starts at 1, so [0] would silently index the last excerpt in Python.
    assert extract_citations("Answer [0].", RESULTS, SOURCES) == ()
    assert invalid_markers("Answer [0].", RESULTS) == (0,)


def test_no_invalid_markers_when_all_are_real() -> None:
    assert invalid_markers("Claim [1] and [3].", RESULTS) == ()


# --- removing markers that point at nothing ----------------------------------------


def test_unsupported_markers_are_removed_from_the_text() -> None:
    # Observed with llama3.1:8b given a single excerpt: it writes "[2]" regardless. Leaving
    # it in shows the user a reference to a source that is not listed anywhere.
    assert strip_unsupported_markers("Tuesdays [2].", RESULTS[:1]) == "Tuesdays."


def test_valid_markers_are_left_alone() -> None:
    assert strip_unsupported_markers("Claim [1] and [2].", RESULTS) == "Claim [1] and [2]."


def test_only_the_unsupported_marker_is_removed() -> None:
    assert strip_unsupported_markers("Real [1], fake [8].", RESULTS) == "Real [1], fake."


def test_text_without_markers_is_unchanged() -> None:
    assert strip_unsupported_markers("No markers.", RESULTS) == "No markers."
