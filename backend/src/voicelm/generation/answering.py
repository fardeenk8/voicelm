"""Tying retrieval and generation together into a cited answer."""

from collections.abc import Mapping, Sequence

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

    return Answer(
        text=strip_unsupported_markers(result.text, selected),
        citations=extract_citations(result.text, selected, sources),
        model=model.model,
        unsupported_markers=invalid_markers(result.text, selected),
    )
