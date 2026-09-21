"""Assembling the prompt that keeps answers grounded.

The model is given the retrieved excerpts and told to answer only from them. That
instruction is the entire mechanism preventing it from blending its training data with the
user's documents, which is the failure mode we care most about: not a refusal, but a
fluent answer where you cannot tell which half came from where.
"""

from collections.abc import Mapping, Sequence

from voicelm.domain.models import Chunk, Source
from voicelm.retrieval.store import SearchResult

REFUSAL_TEXT = "The provided sources do not answer this question."

SYSTEM_INSTRUCTION = f"""You answer questions using only the excerpts provided by the user.

Rules:
- Use only information found in the excerpts. Never add outside knowledge, even if you are
  confident it is correct.
- After each claim, cite the excerpt it came from using its number in square brackets,
  like [1]. Cite every claim.
- Only use excerpt numbers that actually appear above the question.
- If the excerpts do not answer the question, reply with exactly this sentence and nothing
  else: {REFUSAL_TEXT}
- Be concise and do not repeat the question."""

# A budget for the excerpts, in characters. At roughly four characters per token this is
# about 3,000 tokens, which leaves comfortable room inside the 8,192-token context window
# for the instructions above and the generated answer.
DEFAULT_MAX_EXCERPT_CHARS = 12_000


def select_within_budget(
    results: Sequence[SearchResult], max_chars: int = DEFAULT_MAX_EXCERPT_CHARS
) -> list[SearchResult]:
    """Keep the highest-scoring excerpts that fit within `max_chars`.

    Stops at the first excerpt that does not fit rather than skipping it to squeeze in a
    lower-scoring one, so the result is always a prefix of the ranking and therefore easy
    to reason about.

    The top result is always kept even if it alone exceeds the budget: answering from the
    best available excerpt beats answering from nothing. Chunking bounds excerpt length, so
    in practice this does not arise.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")

    kept: list[SearchResult] = []
    used = 0

    for result in results:
        length = len(result.chunk.text)
        if kept and used + length > max_chars:
            break
        kept.append(result)
        used += length

    return kept


def build_user_message(
    question: str,
    results: Sequence[SearchResult],
    sources: Mapping[str, Source],
) -> str:
    """Render the excerpts and the question into a single message.

    Excerpts are numbered from 1 so the model has something concrete to cite, and each is
    labelled with its document title so it can distinguish between sources.
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    if not results:
        return f"Excerpts:\n\n(none were found)\n\nQuestion: {question.strip()}"

    blocks = [
        f'[{marker}] from "{title_for(result.chunk, sources)}":\n{result.chunk.text}'
        for marker, result in enumerate(results, start=1)
    ]

    return "Excerpts:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question.strip()}"


def title_for(chunk: Chunk, sources: Mapping[str, Source]) -> str:
    try:
        return sources[chunk.source_id].title
    except KeyError:
        raise ValueError(f"chunk {chunk.id} refers to unknown source {chunk.source_id!r}") from None
