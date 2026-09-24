"""GitHub repository ingest (Phase 2E) — offline tests with a fake zipball."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx2
import pytest

from voicelm.domain.models import Chunk, EmbeddedChunk, location_kind_for
from voicelm.ingestion.chunking import ChunkingConfig
from voicelm.ingestion.github import (
    EmptyGitHubRepo,
    NotGitHubUrl,
    assemble_lines,
    fetch_github_repo,
    is_github_url,
    parse_github_url,
)
from voicelm.knowledge import KnowledgeBase


class FakeEmbedder:
    model = "fake-embed"

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(chunk=chunk, vector=(1.0, 0.0), model=self.model) for chunk in chunks
        ]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


def _zipball(*entries: tuple[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, text in entries:
            archive.writestr(f"acme-demo-abc1234/{path}", text)
    return buffer.getvalue()


def test_parse_github_urls() -> None:
    root = parse_github_url("https://github.com/acme/demo")
    assert root.owner == "acme"
    assert root.repo == "demo"
    assert root.ref is None
    assert root.subpath == ""
    assert not root.is_blob

    tree = parse_github_url("https://github.com/acme/demo/tree/main/src")
    assert tree.ref == "main"
    assert tree.subpath == "src"
    assert not tree.is_blob

    blob = parse_github_url("https://GitHub.com/acme/demo/blob/main/src/app.py")
    assert blob.is_blob
    assert blob.subpath == "src/app.py"

    assert is_github_url("https://github.com/acme/demo")
    assert not is_github_url("https://gitlab.com/acme/demo")
    with pytest.raises(NotGitHubUrl):
        parse_github_url("https://github.com/acme/demo/issues/1")


def test_assemble_lines_numbers_each_row() -> None:
    text, pages = assemble_lines("alpha\nbeta\n")
    assert text == "alpha\nbeta"
    assert [span.number for span in pages] == [1, 2]
    assert text[pages[0].start_char : pages[0].end_char] == "alpha"
    assert text[pages[1].start_char : pages[1].end_char] == "beta"


def test_fetch_github_repo_from_zipball(monkeypatch: pytest.MonkeyPatch) -> None:
    zip_bytes = _zipball(
        ("README.md", "# Demo\n\nHello from the readme.\n"),
        ("src/app.py", "def greet():\n    return 'hi'\n"),
        ("node_modules/leftpad/index.js", "module.exports = x => x\n"),
        ("dist/bundle.js", "console.log('skip')\n"),
        ("assets/logo.png", "not-really-png"),
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if url.endswith("/repos/acme/demo"):
            return httpx2.Response(200, json={"default_branch": "main"})
        if "/zipball/" in url:
            return httpx2.Response(
                200,
                content=zip_bytes,
                headers={
                    "content-disposition": 'attachment; filename="acme-demo-abc1234.zip"',
                },
            )
        return httpx2.Response(404, json={"message": "not found"})

    transport = httpx2.MockTransport(handler)
    with httpx2.Client(transport=transport) as client:
        snapshot = fetch_github_repo("https://github.com/acme/demo", client=client)

    paths = {item.relative_path for item in snapshot.files}
    assert paths == {"README.md", "src/app.py"}
    assert snapshot.ref == "main"
    assert snapshot.commit_sha == "abc1234"
    assert all(item.pages for item in snapshot.files)


def test_blob_url_keeps_only_that_file() -> None:
    zip_bytes = _zipball(
        ("README.md", "readme\n"),
        ("src/app.py", "print(1)\n"),
        ("src/util.py", "print(2)\n"),
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        if "/zipball/" in str(request.url):
            return httpx2.Response(
                200,
                content=zip_bytes,
                headers={"content-disposition": 'attachment; filename="acme-demo-deadbeef.zip"'},
            )
        return httpx2.Response(404)

    transport = httpx2.MockTransport(handler)
    with httpx2.Client(transport=transport) as client:
        snapshot = fetch_github_repo(
            "https://github.com/acme/demo/blob/main/src/app.py",
            client=client,
        )

    assert [item.relative_path for item in snapshot.files] == ["src/app.py"]


def test_empty_after_filters_raises() -> None:
    zip_bytes = _zipball(("node_modules/x.js", "x\n"), ("logo.png", "nope"))

    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url).endswith("/repos/acme/demo"):
            return httpx2.Response(200, json={"default_branch": "main"})
        if "/zipball/" in str(request.url):
            return httpx2.Response(
                200,
                content=zip_bytes,
                headers={"content-disposition": 'attachment; filename="acme-demo-abc.zip"'},
            )
        return httpx2.Response(404)

    transport = httpx2.MockTransport(handler)
    with httpx2.Client(transport=transport) as client:
        with pytest.raises(EmptyGitHubRepo):
            fetch_github_repo("https://github.com/acme/demo", client=client)


def test_knowledge_base_ingests_github_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_bytes = _zipball(
        ("README.md", "# Demo\n\nUseful overview.\n"),
        ("src/math.py", "def add(a, b):\n    return a + b\n"),
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url).endswith("/repos/acme/demo"):
            return httpx2.Response(200, json={"default_branch": "main"})
        if "/zipball/" in str(request.url):
            return httpx2.Response(
                200,
                content=zip_bytes,
                headers={"content-disposition": 'attachment; filename="acme-demo-abc1234.zip"'},
            )
        return httpx2.Response(404)

    transport = httpx2.MockTransport(handler)
    client = httpx2.Client(transport=transport)

    def fake_fetch(url: str, **kwargs):
        return fetch_github_repo(url, client=client)

    monkeypatch.setattr("voicelm.knowledge.fetch_github_repo", fake_fetch)

    base = KnowledgeBase(
        tmp_path / "data",
        FakeEmbedder(),
        chunking=ChunkingConfig(max_chars=80, overlap_chars=10),
    )
    try:
        first = base.ingest_url("https://github.com/acme/demo")
        assert first.status == "ingested"
        assert first.chunk_count >= 1
        sources = base.list_sources()
        assert len(sources) == 2
        assert all("github" in str(record.source.path) for record in sources)
        assert all(location_kind_for(record.source.path) == "line" for record in sources)
        assert all(record.source.origin_url and "/blob/" in record.source.origin_url for record in sources)

        second = base.ingest_url("https://github.com/acme/demo")
        assert second.status == "skipped"
    finally:
        base.close()
        client.close()
