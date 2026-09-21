"""CLI tests.

These use a real KnowledgeBase (real SQLite, real Qdrant local mode) over a tmp
directory and fake only the two things that would hit the network: the embedder and the
chat model. So a passing test here means the wiring really works end to end.
"""

from pathlib import Path

import pytest
from pdf_fixtures import make_pdf

from voicelm.cli import build_parser, default_data_dir, main
from voicelm.domain.models import Chunk, EmbeddedChunk
from voicelm.generation.ollama import ChatResult
from voicelm.knowledge import KnowledgeBase


class FakeEmbedder:
    model = "fake-embed"

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [EmbeddedChunk(chunk=chunk, vector=(1.0, 0.0), model=self.model) for chunk in chunks]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


class FakeChatModel:
    """Stands in for OllamaChatModel. `main` only needs `.model` and `.chat`."""

    def __init__(self, reply: str) -> None:
        self.model = "fake-chat"
        self.reply = reply
        self.calls: list[str] = []

    def chat(self, system: str, user: str) -> ChatResult:
        self.calls.append(user)
        return ChatResult(text=self.reply, prompt_tokens=0, completion_tokens=0)


@pytest.fixture
def doc(tmp_path: Path) -> Path:
    path = tmp_path / "physics.md"
    path.write_text("Light travels at about 300,000 kilometres per second.", encoding="utf-8")
    return path


@pytest.fixture
def base(tmp_path: Path):
    knowledge_base = KnowledgeBase(tmp_path / "data", FakeEmbedder())
    yield knowledge_base
    knowledge_base.close()


def test_ingest_reports_chunks_and_lists_source(base, doc, capsys):
    assert main(["ingest", "--source", str(doc)], knowledge_base=base) == 0
    assert "1 chunk" in capsys.readouterr().err

    assert main(["sources"], knowledge_base=base) == 0
    listing = capsys.readouterr().out
    assert "physics" in listing
    assert "1 chunk" in listing


def test_second_ingest_of_unchanged_file_is_skipped(base, doc, capsys):
    main(["ingest", "--source", str(doc)], knowledge_base=base)
    capsys.readouterr()

    main(["ingest", "--source", str(doc)], knowledge_base=base)
    assert "skipped" in capsys.readouterr().err


def test_ask_prints_answer_and_citations(base, doc, capsys):
    main(["ingest", "--source", str(doc)], knowledge_base=base)
    capsys.readouterr()

    model = FakeChatModel("Light moves at roughly 300,000 km/s [1].")
    assert main(["ask", "how fast is light?"], knowledge_base=base, chat_model=model) == 0

    captured = capsys.readouterr()
    assert "300,000 km/s [1]" in captured.out
    assert "Sources:" in captured.out
    assert "[1] physics" in captured.out


def test_ask_can_ingest_first(base, doc, capsys):
    model = FakeChatModel("Yes [1].")
    exit_code = main(
        ["ask", "how fast is light?", "--source", str(doc)],
        knowledge_base=base,
        chat_model=model,
    )

    assert exit_code == 0
    assert "physics.md: ingested" in capsys.readouterr().err


def test_ask_with_empty_library_explains_what_to_do(base, capsys):
    exit_code = main(["ask", "anything?"], knowledge_base=base, chat_model=FakeChatModel("x"))

    assert exit_code == 1
    assert "voicelm ingest" in capsys.readouterr().err


def test_unsupported_marker_is_reported_and_removed(base, doc, capsys):
    main(["ingest", "--source", str(doc)], knowledge_base=base)
    capsys.readouterr()

    model = FakeChatModel("Light is fast [1] and so is sound [2].")
    main(["ask", "how fast is light?"], knowledge_base=base, chat_model=model)

    captured = capsys.readouterr()
    assert "[2]" not in captured.out
    assert "[2]" in captured.err


def test_sources_on_empty_library(base, capsys):
    assert main(["sources"], knowledge_base=base) == 0
    assert "library is empty" in capsys.readouterr().out


def test_missing_file_is_a_friendly_error(base, tmp_path, capsys):
    exit_code = main(["ingest", "--source", str(tmp_path / "nope.txt")], knowledge_base=base)

    assert exit_code == 1
    assert "error reading source" in capsys.readouterr().err


def test_unsupported_extension_is_a_friendly_error(base, tmp_path, capsys):
    path = tmp_path / "slides.pptx"
    path.write_bytes(b"not a document we handle")

    assert main(["ingest", "--source", str(path)], knowledge_base=base) == 1
    assert "error reading source" in capsys.readouterr().err


def test_a_scanned_pdf_reports_rather_than_crashing(base, tmp_path, capsys):
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf(["", ""]))

    assert main(["ingest", "--source", str(path)], knowledge_base=base) == 1
    assert "OCR" in capsys.readouterr().err


def test_a_pdf_is_ingested_and_listed(base, tmp_path, capsys):
    path = tmp_path / "bio.pdf"
    path.write_bytes(make_pdf(["Mitochondria make ATP.", "Ribosomes build proteins."]))

    assert main(["ingest", "--source", str(path)], knowledge_base=base) == 0
    assert main(["sources"], knowledge_base=base) == 0
    assert "bio" in capsys.readouterr().out


def test_a_pdf_citation_prints_the_page_number(base, tmp_path, capsys):
    path = tmp_path / "bio.pdf"
    path.write_bytes(make_pdf(["Mitochondria make ATP.", "Ribosomes build proteins."]))
    main(["ingest", "--source", str(path)], knowledge_base=base)
    capsys.readouterr()

    model = FakeChatModel("Mitochondria make ATP [1].")
    main(["ask", "what do mitochondria do?"], knowledge_base=base, chat_model=model)

    captured = capsys.readouterr().out
    assert "page" in captured
    assert "bio" in captured


def test_data_dir_defaults_to_cwd_then_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VOICELM_DATA_DIR", raising=False)
    assert default_data_dir() == tmp_path / "data"

    monkeypatch.setenv("VOICELM_DATA_DIR", "/tmp/elsewhere")
    assert default_data_dir() == Path("/tmp/elsewhere")


def test_chunk_options_are_parsed(doc):
    arguments = build_parser().parse_args(
        ["ingest", "--source", str(doc), "--chunk-chars", "400", "--chunk-overlap", "40"]
    )

    assert (arguments.chunk_chars, arguments.chunk_overlap) == (400, 40)


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])
