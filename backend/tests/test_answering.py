"""Tests for the assembly step, using a stub model so no network is involved."""

import json
from pathlib import Path

import httpx2
import pytest

from voicelm.domain.models import Chunk, Source
from voicelm.generation.answering import answer_question
from voicelm.generation.ollama import OllamaChatModel
from voicelm.retrieval.store import SearchResult

SOURCES = {"src": Source(id="src", path=Path("notes.md"), title="notes", text="", content_hash="h")}


def make_result(text: str, chunk_id: str = "src:0") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            source_id="src",
            text=text,
            start_char=0,
            end_char=len(text),
            ordinal=0,
        ),
        score=0.8,
    )


def model_replying(content: str) -> tuple[OllamaChatModel, list[dict]]:
    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "message": {"role": "assistant", "content": content},
                "prompt_eval_count": 50,
                "eval_count": 10,
            },
        )

    client = httpx2.Client(base_url="http://testserver", transport=httpx2.MockTransport(handler))
    return OllamaChatModel(client=client), sent


def test_answer_carries_text_citations_and_model() -> None:
    model, _ = model_replying("Caching is used [1].")

    answer = answer_question("What is used?", [make_result("We cache responses.")], SOURCES, model)

    assert answer.text == "Caching is used [1]."
    assert [citation.marker for citation in answer.citations] == [1]
    assert answer.citations[0].quote == "We cache responses."
    assert answer.model == "llama3.1:8b"


def test_the_excerpts_reach_the_model() -> None:
    model, sent = model_replying("ok")

    answer_question("Q?", [make_result("the retrieved text")], SOURCES, model)

    user_message = sent[0]["messages"][1]["content"]
    assert "the retrieved text" in user_message
    assert "Q?" in user_message


def test_a_refusal_produces_no_citations() -> None:
    model, _ = model_replying("The provided sources do not answer this question.")

    answer = answer_question("Unrelated?", [make_result("Something else.")], SOURCES, model)

    assert answer.citations == ()


def test_citations_only_reference_excerpts_actually_sent() -> None:
    # The budget trims the ranking to one excerpt, so a marker of [2] refers to text the
    # model never saw and must not become a citation.
    model, _ = model_replying("Claims [1] and [2].")

    answer = answer_question(
        "Q?",
        [make_result("a" * 60, "src:0"), make_result("b" * 60, "src:1")],
        SOURCES,
        model,
        max_excerpt_chars=100,
    )

    assert [citation.marker for citation in answer.citations] == [1]


def test_blank_question_is_rejected() -> None:
    model, _ = model_replying("ok")

    with pytest.raises(ValueError, match="cannot be empty"):
        answer_question("  ", [make_result("text")], SOURCES, model)
