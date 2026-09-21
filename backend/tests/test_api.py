"""HTTP API tests.

A real KnowledgeBase over a tmp directory, fakes only for Ollama. TestClient runs the
FastAPI lifespan so these hit the same wiring uvicorn would.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import make_pdf

from voicelm import __version__
from voicelm.api.app import create_app
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
    def __init__(self, reply: str) -> None:
        self.model = "fake-chat"
        self.reply = reply

    def chat(self, system: str, user: str) -> ChatResult:
        return ChatResult(text=self.reply, prompt_tokens=0, completion_tokens=0)


@pytest.fixture
def api(tmp_path: Path):
    base = KnowledgeBase(tmp_path / "data", FakeEmbedder())
    app = create_app(knowledge_base=base, chat_model=FakeChatModel("Light is fast [1]."))
    with TestClient(app) as client:
        yield client, base, tmp_path
    base.close()


def test_health_reports_ok(api) -> None:
    client, _base, _tmp = api

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_unknown_route_is_404(api) -> None:
    client, _base, _tmp = api

    assert client.get("/does-not-exist").status_code == 404


def test_empty_library_lists_nothing(api) -> None:
    client, _base, _tmp = api

    assert client.get("/sources").json() == []


def test_upload_ingest_list_and_ask(api) -> None:
    client, _base, _tmp = api

    uploaded = client.post(
        "/sources",
        files={"file": ("physics.md", b"Light travels at 300,000 km/s.", "text/markdown")},
    )
    assert uploaded.status_code == 200
    body = uploaded.json()
    assert body["status"] == "ingested"
    assert body["title"] == "physics"
    assert body["chunk_count"] == 1
    assert body["path"].endswith("files/physics.md")

    listing = client.get("/sources").json()
    assert len(listing) == 1
    assert listing[0]["id"] == body["id"]

    answer = client.post("/ask", json={"question": "how fast is light?"})
    assert answer.status_code == 200
    payload = answer.json()
    assert "fast [1]" in payload["text"]
    assert payload["citations"][0]["source_title"] == "physics"
    assert payload["citations"][0]["pages"] == []


def test_pdf_upload_cites_pages(api) -> None:
    client, _base, _tmp = api
    pdf = make_pdf(["Mitochondria make ATP.", "Ribosomes build proteins."])

    uploaded = client.post("/sources", files={"file": ("bio.pdf", pdf, "application/pdf")})
    assert uploaded.json()["page_count"] == 2

    answer = client.post("/ask", json={"question": "what do mitochondria do?"})
    assert answer.json()["citations"][0]["pages"]


def test_second_upload_of_the_same_name_is_skipped(api) -> None:
    client, _base, _tmp = api

    first = client.post("/sources", files={"file": ("notes.md", b"same body", "text/markdown")})
    second = client.post("/sources", files={"file": ("notes.md", b"same body", "text/markdown")})

    assert first.json()["status"] == "ingested"
    assert second.json()["status"] == "skipped"
    assert first.json()["id"] == second.json()["id"]


def test_delete_removes_the_source_and_the_owned_copy(api) -> None:
    client, base, tmp_path = api
    uploaded = client.post(
        "/sources",
        files={"file": ("notes.md", b"forget me", "text/markdown")},
    )
    source_id = uploaded.json()["id"]
    owned = tmp_path / "data" / "files" / "notes.md"
    assert owned.is_file()

    deleted = client.delete(f"/sources/{source_id}")
    assert deleted.status_code == 204
    assert client.get("/sources").json() == []
    assert not owned.exists()
    assert base.remove(source_id) is False


def test_delete_unknown_id_is_404(api) -> None:
    client, _base, _tmp = api

    response = client.delete("/sources/not-a-real-id")

    assert response.status_code == 404


def test_ask_on_empty_library_is_409(api) -> None:
    client, _base, _tmp = api

    response = client.post("/ask", json={"question": "anything?"})

    assert response.status_code == 409
    assert "empty" in response.json()["detail"]


def test_scanned_pdf_is_400(api) -> None:
    client, _base, _tmp = api
    pdf = make_pdf(["", ""])

    response = client.post("/sources", files={"file": ("scan.pdf", pdf, "application/pdf")})

    assert response.status_code == 400
    assert "OCR" in response.json()["detail"]


def test_unsupported_type_is_400(api) -> None:
    client, _base, _tmp = api

    response = client.post(
        "/sources",
        files={"file": ("slides.pptx", b"not a document we handle", "application/octet-stream")},
    )

    assert response.status_code == 400


def test_path_traversal_filename_is_stored_under_files(api) -> None:
    client, _base, tmp_path = api

    response = client.post(
        "/sources",
        files={"file": ("../escape.md", b"should not land outside files/", "text/markdown")},
    )

    assert response.status_code == 200
    dest = Path(response.json()["path"])
    assert dest.parent == (tmp_path / "data" / "files").resolve()
    assert not (tmp_path / "escape.md").exists()


def test_upload_over_the_size_limit_is_413(api, monkeypatch) -> None:
    client, _base, _tmp = api
    monkeypatch.setattr("voicelm.api.app.MAX_UPLOAD_BYTES", 8)

    response = client.post(
        "/sources",
        files={"file": ("notes.md", b"longer than eight bytes", "text/markdown")},
    )

    assert response.status_code == 413
