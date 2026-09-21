import json
from pathlib import Path

import httpx2
import pytest

from voicelm.domain.models import Chunk, EmbeddedChunk
from voicelm.generation.ollama import OllamaChatModel
from voicelm.ingestion.chunking import ChunkingConfig
from voicelm.knowledge import KnowledgeBase
from voicelm.storage.sqlite import SqliteMetadataStore

TWO_PARAGRAPHS = "Alpha lives in the first paragraph.\n\nBeta lives in the second paragraph."


class FakeEmbedder:
    """Deterministic stand-in for Ollama so these tests never touch the network."""

    model = "fake-embed"

    def __init__(self) -> None:
        self.chunk_calls = 0
        self.query_calls = 0

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        self.chunk_calls += 1
        return [
            EmbeddedChunk(
                chunk=chunk,
                vector=(1.0, 0.0) if index == 0 else (0.0, 1.0),
                model=self.model,
            )
            for index, chunk in enumerate(chunks)
        ]

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.query_calls += 1
        if "alpha" in text.lower():
            return (1.0, 0.0)
        return (0.0, 1.0)


class BoomIndex:
    def count_by_source(self, source_id: str) -> int:
        return 0

    def delete_by_source(self, source_id: str) -> None:
        return None

    def add(self, embedded: object) -> None:
        raise RuntimeError("qdrant unavailable")

    def search(self, query_vector: object, top_k: int = 5) -> list:
        return []

    def close(self) -> None:
        return None


def write_doc(folder: Path, name: str, content: str) -> Path:
    path = folder / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path):
    embedder = FakeEmbedder()
    base = KnowledgeBase(
        tmp_path / "data",
        embedder,
        chunking=ChunkingConfig(max_chars=40, overlap_chars=10),
    )
    yield tmp_path, base, embedder
    base.close()


def test_ingest_then_search_joins_ids_to_passages(workspace) -> None:
    folder, base, _embedder = workspace
    write_doc(folder, "notes.md", TWO_PARAGRAPHS)
    result = base.ingest(folder / "notes.md")

    assert result.status == "ingested"
    assert result.chunk_count == 2

    hits, sources = base.search("Tell me about alpha", top_k=1)

    assert len(hits) == 1
    assert "Alpha" in hits[0].chunk.text
    assert hits[0].chunk.source_id in sources
    assert sources[hits[0].chunk.source_id].title == "notes"


def test_search_ranks_the_relevant_paragraph_first(workspace) -> None:
    folder, base, _embedder = workspace
    write_doc(folder, "notes.md", TWO_PARAGRAPHS)
    base.ingest(folder / "notes.md")

    alpha, _ = base.search("alpha", top_k=2)
    beta, _ = base.search("beta", top_k=2)

    assert "Alpha" in alpha[0].chunk.text
    assert "Beta" in beta[0].chunk.text


def test_unchanged_file_is_not_re_embedded(workspace) -> None:
    folder, base, embedder = workspace
    path = write_doc(folder, "notes.md", TWO_PARAGRAPHS)
    first = base.ingest(path)
    second = base.ingest(path)

    assert first.status == "ingested"
    assert second.status == "skipped"
    assert second.source.id == first.source.id
    assert embedder.chunk_calls == 1


def test_changed_file_keeps_the_same_id(workspace) -> None:
    folder, base, embedder = workspace
    path = write_doc(folder, "notes.md", TWO_PARAGRAPHS)
    first = base.ingest(path)
    path.write_text("Alpha changed.\n\nBeta changed too.", encoding="utf-8")
    second = base.ingest(path)

    assert second.status == "updated"
    assert second.source.id == first.source.id
    assert second.source.content_hash != first.source.content_hash
    assert embedder.chunk_calls == 2


def test_identical_text_at_two_paths_is_two_sources(workspace) -> None:
    folder, base, _embedder = workspace
    write_doc(folder, "draft.md", TWO_PARAGRAPHS)
    write_doc(folder, "final.md", TWO_PARAGRAPHS)
    first = base.ingest(folder / "draft.md")
    second = base.ingest(folder / "final.md")

    assert first.source.id != second.source.id
    assert first.source.content_hash == second.source.content_hash
    assert len(base.list_sources()) == 2


def test_missing_vectors_are_repaired_instead_of_skipped(workspace) -> None:
    folder, base, embedder = workspace
    path = write_doc(folder, "notes.md", TWO_PARAGRAPHS)
    first = base.ingest(path)
    base._index.delete_by_source(first.source.id)

    repaired = base.ingest(path)

    assert repaired.status == "updated"
    assert repaired.source.id == first.source.id
    assert embedder.chunk_calls == 2
    hits, _ = base.search("alpha", top_k=1)
    assert hits


def test_qdrant_failure_on_a_new_file_leaves_no_catalog_row(tmp_path: Path) -> None:
    catalog = SqliteMetadataStore(tmp_path / "voicelm.db")
    embedder = FakeEmbedder()
    base = KnowledgeBase(
        tmp_path,
        embedder,
        catalog=catalog,
        index=BoomIndex(),  # type: ignore[arg-type]
    )
    path = write_doc(tmp_path, "notes.md", TWO_PARAGRAPHS)

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        base.ingest(path)

    assert catalog.list_sources() == []
    catalog.close()


def test_ask_uses_retrieved_passages(workspace) -> None:
    folder, base, _embedder = workspace
    write_doc(folder, "notes.md", TWO_PARAGRAPHS)
    base.ingest(folder / "notes.md")

    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Alpha is first [1]."},
                "prompt_eval_count": 40,
                "eval_count": 8,
            },
        )

    model = OllamaChatModel(
        client=httpx2.Client(base_url="http://testserver", transport=httpx2.MockTransport(handler))
    )
    answer = base.ask("What about alpha?", model, top_k=1)

    user_message = sent[0]["messages"][1]["content"]
    assert "Alpha lives" in user_message
    assert answer.citations[0].source_title == "notes"


def test_blank_question_is_rejected(workspace) -> None:
    _folder, base, _embedder = workspace

    with pytest.raises(ValueError, match="cannot be empty"):
        base.search("   ")
