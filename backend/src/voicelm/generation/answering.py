"""Tying retrieval and generation together into a cited answer."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from voicelm.domain.models import Answer, Source
from voicelm.generation.citations import (
    extract_citations,
    invalid_markers,
    strip_unsupported_markers,
)
from voicelm.generation.ollama import ChatModel
from voicelm.generation.prompt import (
    DEFAULT_MAX_EXCERPT_CHARS,
    SYSTEM_INSTRUCTION,
    build_user_message,
    select_within_budget,
)
from voicelm.retrieval.store import SearchResult


@dataclass(frozen=True)
class AnswerToken:
    """One piece of model text. Citations are not known until the stream ends."""

    text: str


def answer_question(
    question: str,
    results: Sequence[SearchResult],
    sources: Mapping[str, Source],
    model: ChatModel,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> Answer:
    """Answer `question` using only `results`, returning citations for what was used.

    Citations are extracted against the excerpts that were actually sent, not against
    everything retrieved, so a marker can only ever point at text the model really saw.
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    selected = select_within_budget(results, max_excerpt_chars)
    user_message = build_user_message(question, selected, sources)
    result = model.chat(SYSTEM_INSTRUCTION, user_message)

    return _cited_answer(result.text, selected, sources, model.model)


def answer_question_stream(
    question: str,
    results: Sequence[SearchResult],
    sources: Mapping[str, Source],
    model: ChatModel,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> Iterator[AnswerToken | Answer]:
    """Same grounding as `answer_question`, but yield tokens before the final Answer.

    The UI can print tokens as they arrive. Markers are only stripped, and citations
    only attached, on the final `Answer` — a `[2]` in flight might still become valid
    later, or be removed when the model stops.
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    selected = select_within_budget(results, max_excerpt_chars)
    user_message = build_user_message(question, selected, sources)
    pieces: list[str] = []
    for token in model.chat_stream(SYSTEM_INSTRUCTION, user_message):
        pieces.append(token)
        yield AnswerToken(token)

    yield _cited_answer("".join(pieces).strip(), selected, sources, model.model)


def _cited_answer(
    raw_text: str,
    selected: Sequence[SearchResult],
    sources: Mapping[str, Source],
    model_name: str,
) -> Answer:
    return Answer(
        text=strip_unsupported_markers(raw_text, selected),
        citations=extract_citations(raw_text, selected, sources),
        model=model_name,
        unsupported_markers=invalid_markers(raw_text, selected),
    )
