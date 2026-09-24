"""Turning the `[n]` markers a model writes into verifiable citations.

Nothing here trusts the model beyond the number it wrote. The source, title, and character
offsets all come from our own ingestion records, so a citation points at text we can slice
back out of the document. A model cannot fabricate a citation; the most it can do is cite
the wrong excerpt, or cite a number that does not exist — which we detect.
"""

import re
from collections.abc import Mapping, Sequence

from voicelm.domain.models import Citation, Source, location_kind_for, pages_covering
from voicelm.generation.prompt import title_for
from voicelm.retrieval.store import SearchResult

_MARKER = re.compile(r"\[(\d+)\]")


def extract_citations(
    answer_text: str,
    results: Sequence[SearchResult],
    sources: Mapping[str, Source],
) -> tuple[Citation, ...]:
    """Return a citation for each valid marker, in order of first appearance."""
    citations: list[Citation] = []
    seen: set[int] = set()

    for match in _MARKER.finditer(answer_text):
        marker = int(match.group(1))

        if marker in seen or not 1 <= marker <= len(results):
            continue

        seen.add(marker)
        chunk = results[marker - 1].chunk
        source = sources.get(chunk.source_id)
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.id,
                source_id=chunk.source_id,
                source_title=title_for(chunk, sources),
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                quote=chunk.text,
                pages=pages_covering(chunk.start_char, chunk.end_char, source.pages)
                if source is not None
                else (),
                location_kind=location_kind_for(source.path) if source is not None else None,
                origin_url=source.origin_url if source is not None else None,
            )
        )

    return tuple(citations)


def invalid_markers(answer_text: str, results: Sequence[SearchResult]) -> tuple[int, ...]:
    """Markers the model wrote that do not correspond to any excerpt it was given.

    A non-empty result means the model invented a reference. Worth surfacing rather than
    quietly dropping, because it is direct evidence the answer is less grounded than it
    looks.
    """
    markers = {int(match.group(1)) for match in _MARKER.finditer(answer_text)}

    return tuple(sorted(marker for marker in markers if not 1 <= marker <= len(results)))


def strip_unsupported_markers(answer_text: str, results: Sequence[SearchResult]) -> str:
    """Remove markers that do not correspond to an excerpt.

    Observed with llama3.1:8b: given a single excerpt it still sometimes writes "[2]".
    Leaving that in place shows the user a reference to a source that is not listed, which
    reads like our bug rather than the model's. We cannot guess which excerpt was meant, so
    the honest move is to drop the marker and keep the claim it was attached to.

    The leading whitespace is consumed too, so "Tuesdays [2]." becomes "Tuesdays."
    """
    unsupported = invalid_markers(answer_text, results)

    if not unsupported:
        return answer_text

    alternatives = "|".join(str(marker) for marker in unsupported)

    return re.sub(rf"\s*\[(?:{alternatives})\]", "", answer_text)
