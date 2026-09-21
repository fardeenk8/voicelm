from pathlib import Path

import pytest

from voicelm.domain.models import Chunk, Source
from voicelm.generation.prompt import (
    REFUSAL_TEXT,
    SYSTEM_INSTRUCTION,
    build_user_message,
    select_within_budget,
)
from voicelm.retrieval.store import SearchResult


def make_result(text: str, score: float = 0.5, source_id: str = "src") -> SearchResult:
    chunk = Chunk(
        id=f"{source_id}:{text[:4]}",
        source_id=source_id,
        text=text,
        start_char=0,
        end_char=len(text),
        ordinal=0,
    )
    return SearchResult(chunk=chunk, score=score)


SOURCES = {"src": Source(id="src", path=Path("notes.md"), title="notes", text="")}


# --- the system instruction -------------------------------------------------------


def test_system_instruction_forbids_outside_knowledge() -> None:
    assert "only information found in the excerpts" in SYSTEM_INSTRUCTION
    assert "Never add outside knowledge" in SYSTEM_INSTRUCTION


def test_system_instruction_asks_for_citations_and_offers_a_way_out() -> None:
    assert "square brackets" in SYSTEM_INSTRUCTION
    # A stated refusal sentence matters: without one the model has no sanctioned option
    # except to invent an answer.
    assert REFUSAL_TEXT in SYSTEM_INSTRUCTION


# --- rendering --------------------------------------------------------------------


def test_excerpts_are_numbered_from_one() -> None:
    message = build_user_message("Q?", [make_result("alpha"), make_result("beta")], SOURCES)

    assert '[1] from "notes":\nalpha' in message
    assert '[2] from "notes":\nbeta' in message


def test_question_appears_after_the_excerpts() -> None:
    message = build_user_message("What is cached?", [make_result("alpha")], SOURCES)

    assert message.index("alpha") < message.index("What is cached?")


def test_blank_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        build_user_message("   ", [make_result("alpha")], SOURCES)


def test_no_results_still_produces_a_prompt() -> None:
    # The model should refuse rather than be asked nothing at all.
    message = build_user_message("Q?", [], SOURCES)

    assert "(none were found)" in message


def test_unknown_source_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        build_user_message("Q?", [make_result("alpha", source_id="missing")], SOURCES)


# --- the context budget -----------------------------------------------------------


def test_budget_keeps_everything_that_fits() -> None:
    results = [make_result("x" * 100) for _ in range(5)]

    assert len(select_within_budget(results, max_chars=1000)) == 5


def test_budget_stops_at_the_first_excerpt_that_does_not_fit() -> None:
    results = [make_result("x" * 100) for _ in range(10)]

    # Room for three, so the ranking is truncated to its first three entries.
    assert len(select_within_budget(results, max_chars=350)) == 3


def test_budget_always_keeps_the_top_result() -> None:
    # Answering from the best available excerpt beats answering from nothing.
    results = [make_result("x" * 5000), make_result("y" * 10)]

    kept = select_within_budget(results, max_chars=100)

    assert len(kept) == 1
    assert kept[0].chunk.text.startswith("x")


def test_budget_preserves_ranking_order() -> None:
    results = [make_result("a" * 50, score=0.9), make_result("b" * 50, score=0.1)]

    kept = select_within_budget(results, max_chars=1000)

    assert [result.score for result in kept] == [0.9, 0.1]


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        select_within_budget([make_result("a")], max_chars=0)
