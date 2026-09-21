"""The whole loop, against real Ollama and a real document.

This is the test that says whether VoiceLM works. Everything else checks a part.
Skipped automatically when Ollama is not running.
"""

import re
from pathlib import Path

import httpx2
import pytest

from voicelm.embeddings.ollama import DEFAULT_BASE_URL, OllamaEmbedder
from voicelm.generation.answering import answer_question
from voicelm.generation.ollama import OllamaChatModel
from voicelm.generation.prompt import REFUSAL_TEXT
from voicelm.ingestion.chunking import ChunkingConfig, chunk_document
from voicelm.ingestion.loader import load_source
from voicelm.retrieval.store import InMemoryVectorStore

DOCUMENT = """# Engineering notes

## Caching

We cache embeddings on disk keyed by a content hash rather than by file path, because
re-embedding unchanged documents dominated our ingestion time.

## Deployment

Releases go out on Tuesdays. We avoid Fridays so that nobody spends a weekend on a
rollback.

## Logging

Logs are retained for ninety days and then deleted automatically.
"""


def ollama_is_running() -> bool:
    try:
        return httpx2.get(f"{DEFAULT_BASE_URL}/api/version", timeout=2.0).status_code == 200
    except httpx2.RequestError:
        return False


pytestmark = pytest.mark.skipif(
    not ollama_is_running(), reason="Ollama is not running on 127.0.0.1:11434"
)


@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory):
    """Ingest the document once and reuse it, since embedding costs real time."""
    path: Path = tmp_path_factory.mktemp("docs") / "engineering.md"
    path.write_text(DOCUMENT, encoding="utf-8")

    source = load_source(path)
    # A deliberately small chunk size so this short document yields one chunk per section.
    # At the default 1000 characters the whole document is a single excerpt, which is not
    # the situation citations are for.
    chunks = chunk_document(source, ChunkingConfig(max_chars=160, overlap_chars=30))

    embedder = OllamaEmbedder()
    store = InMemoryVectorStore()
    store.add(embedder.embed_chunks(chunks))

    return embedder, store, {source.id: source}, source


def ask(indexed, question: str, top_k: int = 3):
    embedder, store, sources, _ = indexed
    results = store.search(embedder.embed_query(question), top_k=top_k)

    return answer_question(question, results, sources, OllamaChatModel())


def test_answers_a_question_from_the_document(indexed) -> None:
    answer = ask(indexed, "Why are embeddings cached by content hash?")

    assert REFUSAL_TEXT not in answer.text
    # The reason given in the document, not a plausible-sounding invention.
    assert "re-embedding" in answer.text.lower() or "unchanged" in answer.text.lower()


def test_the_answer_is_cited(indexed) -> None:
    answer = ask(indexed, "When do releases go out?")

    assert answer.citations, f"expected at least one citation, got: {answer.text!r}"


def test_every_citation_slices_back_out_of_the_source(indexed) -> None:
    """The point of the whole provenance chain: a citation is checkable, not a promise."""
    *_, source = indexed
    answer = ask(indexed, "How long are logs kept?")

    assert answer.citations
    for citation in answer.citations:
        assert source.text[citation.start_char : citation.end_char] == citation.quote
        assert citation.source_title == "engineering"


def test_retrieval_finds_the_relevant_section(indexed) -> None:
    embedder, store, _, _ = indexed

    results = store.search(embedder.embed_query("what day do we ship releases?"), top_k=1)

    assert "Tuesdays" in results[0].chunk.text


def test_refuses_a_question_the_document_cannot_answer(indexed) -> None:
    """Refusal is a feature. Confident fabrication is the failure mode we design against."""
    answer = ask(indexed, "What is the CEO's home address?")

    assert REFUSAL_TEXT in answer.text
    assert answer.citations == ()


def test_the_document_produces_several_excerpts(indexed) -> None:
    # Guards the fixture itself: if the document collapsed into one chunk, the citation
    # tests above would be checking a situation that never occurs in practice.
    _, store, _, _ = indexed

    assert len(store) >= 3


def test_an_answer_never_references_a_missing_source(indexed) -> None:
    """Whatever the model writes, displayed markers and listed citations must agree."""
    answer = ask(indexed, "What day do releases happen and how long are logs kept?")

    cited = {citation.marker for citation in answer.citations}
    mentioned = {int(marker) for marker in re.findall(r"\[(\d+)\]", answer.text)}

    assert mentioned == cited
