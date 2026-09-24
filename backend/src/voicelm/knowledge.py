"""The join between the catalog (SQLite) and the vector index (Qdrant).

Ingestion, skip-if-unchanged, search, and answering all go through here so neither
store is written from a route handler or a command function.
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from voicelm.domain.models import Answer, Chunk, EmbeddedChunk, Source
from voicelm.generation.answering import AnswerToken, answer_question, answer_question_stream
from voicelm.generation.ollama import ChatModel
from voicelm.ingestion.chunking import ChunkingConfig, chunk_document
from voicelm.ingestion.github import fetch_github_repo, is_github_url
from voicelm.ingestion.loader import (
    SUPPORTED_SUFFIXES,
    UnsupportedFileType,
    hash_content,
    load_source,
)
from voicelm.ingestion.media import (
    MEDIA_SUFFIXES,
    Transcriber,
    default_transcriber,
    hash_media_file,
    load_media,
)
from voicelm.ingestion.ocr import IMAGE_SUFFIXES, OcrEngine, default_ocr, ocr_image
from voicelm.ingestion.web import fetch_web_page, url_storage_name
from voicelm.ingestion.youtube import (
    fetch_youtube_video,
    is_youtube_url,
    youtube_storage_name,
)
from voicelm.paths import owned_files_dir
from voicelm.retrieval.store import SearchResult
from voicelm.storage.qdrant import QdrantVectorIndex
from voicelm.storage.sqlite import IngestMeta, SourceRecord, SqliteMetadataStore

IngestStatus = Literal["ingested", "updated", "skipped"]

# Default how many vector hits to request. Five was too few for long web articles
# (a 10-chunk Medium post listing five books only surfaced part of the list).
DEFAULT_TOP_K = 10
# After vector search, also load this many chunks on either side of each hit so a
# numbered list split across consecutive chunks is not half-missing from the prompt.
NEIGHBOR_RADIUS = 1


class Embedder(Protocol):
    """Anything that turns text into vectors. Tests pass a fake; production uses Ollama."""

    model: str

    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class IngestResult:
    source: Source
    chunk_count: int
    status: IngestStatus


class KnowledgeBase:
    def __init__(
        self,
        data_dir: Path,
        embedder: Embedder,
        chunking: ChunkingConfig | None = None,
        catalog: SqliteMetadataStore | None = None,
        index: QdrantVectorIndex | None = None,
        transcriber: Transcriber | None = None,
        ocr: OcrEngine | None = None,
    ) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir
        self._embedder = embedder
        self._chunking = chunking or ChunkingConfig()
        self._catalog = catalog or SqliteMetadataStore(data_dir / "voicelm.db")
        self._index = index or QdrantVectorIndex(data_dir / "qdrant")
        # Lazy default: Whisper / OCR model download only happens when media needs it.
        self._transcriber = transcriber
        self._ocr = ocr

    def close(self) -> None:
        self._catalog.close()
        self._index.close()

    def _get_transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = default_transcriber()
        return self._transcriber

    def _get_ocr(self) -> OcrEngine:
        if self._ocr is None:
            self._ocr = default_ocr()
        return self._ocr

    def ingest(self, path: Path) -> IngestResult:
        """Add or refresh one document.

        Same path, same hash, complete index: do nothing.
        Same path, same hash, missing vectors: re-embed (repair after a crashed write).
        Same path, different hash: keep the id, replace text, chunks, and vectors.
        New path: mint the loader's UUID as the id.

        Audio/video hashes the file bytes before Whisper so an unchanged recording is not
        re-transcribed (ADR-0032).
        """
        if path.suffix.lower() in MEDIA_SUFFIXES:
            return self._ingest_media(path)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return self._ingest_image(path)

        loaded = load_source(path, ocr=self._ocr)
        existing = self._catalog.get_by_path(path)

        if (
            existing is not None
            and existing.source.content_hash == loaded.content_hash
            and self._index_is_complete(existing.source.id)
        ):
            chunks = self._catalog.get_chunks(existing.source.id)
            return IngestResult(existing.source, len(chunks), "skipped")

        if existing is None:
            source_id = loaded.id
            status: IngestStatus = "ingested"
        else:
            source_id = existing.source.id
            status = "updated"

        source = Source(
            id=source_id,
            path=path,
            title=loaded.title,
            text=loaded.text,
            content_hash=loaded.content_hash,
            pages=loaded.pages,
            origin_url=existing.source.origin_url if existing is not None else None,
        )
        return self._commit_source(source, status, was_new=existing is None)

    def _ingest_media(self, path: Path) -> IngestResult:
        content_hash = hash_media_file(path)
        existing = self._catalog.get_by_path(path)

        if (
            existing is not None
            and existing.source.content_hash == content_hash
            and self._index_is_complete(existing.source.id)
        ):
            chunks = self._catalog.get_chunks(existing.source.id)
            return IngestResult(existing.source, len(chunks), "skipped")

        text, pages, content_hash = load_media(path, transcriber=self._get_transcriber())

        if existing is None:
            source_id = str(uuid4())
            status: IngestStatus = "ingested"
        else:
            source_id = existing.source.id
            status = "updated"

        source = Source(
            id=source_id,
            path=path,
            title=path.stem,
            text=text,
            content_hash=content_hash,
            pages=pages,
        )
        return self._commit_source(source, status, was_new=existing is None)

    def _ingest_image(self, path: Path) -> IngestResult:
        """OCR an image; hash file bytes so an unchanged screenshot skips re-OCR."""
        content_hash = hash_media_file(path)
        existing = self._catalog.get_by_path(path)

        if (
            existing is not None
            and existing.source.content_hash == content_hash
            and self._index_is_complete(existing.source.id)
        ):
            chunks = self._catalog.get_chunks(existing.source.id)
            return IngestResult(existing.source, len(chunks), "skipped")

        text, pages = ocr_image(path, engine=self._get_ocr())

        if existing is None:
            source_id = str(uuid4())
            status: IngestStatus = "ingested"
        else:
            source_id = existing.source.id
            status = "updated"

        source = Source(
            id=source_id,
            path=path,
            title=path.stem,
            text=text,
            content_hash=content_hash,
            pages=pages,
        )
        return self._commit_source(source, status, was_new=existing is None)

    def ingest_upload(self, filename: str, data: bytes) -> IngestResult:
        """Save `data` under the library's files directory and ingest that copy.

        The API cannot assume a shared filesystem with the client, so it keeps its own
        copy. The filename's last component is the path we catalog by: uploading
        `notes.md` twice is an update of the same source, not a second document.
        """
        dest = self._destination_for(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        try:
            return self.ingest(dest)
        except Exception:
            # Leave the copy if this path is already in the catalog (an update that
            # failed after the previous version existed). Delete it if we just created
            # a new file that never made it into the catalog.
            if self._catalog.get_by_path(dest) is None:
                dest.unlink(missing_ok=True)
            raise

    def ingest_url(self, url: str) -> IngestResult:
        """Fetch a web page, YouTube transcript, or GitHub repo and ingest owned copies.

        YouTube links prefer on-platform captions; if those fail, local Whisper runs on
        audio downloaded with yt-dlp (ADR-0032). GitHub links download a zipball and
        index each text file as its own source with line citations (ADR-0034). Other
        http(s) URLs use the web extractor. Catalog keys live under `files/web/`,
        `files/youtube/`, or `files/github/`, so the same URL twice is an update (or
        skip), not a duplicate.
        """
        if is_youtube_url(url):
            return self._ingest_youtube(url)
        if is_github_url(url):
            return self._ingest_github(url)
        return self._ingest_web_page(url)

    def _ingest_github(self, url: str) -> IngestResult:
        snapshot = fetch_github_repo(url)
        root = self._destination_for_github(snapshot.storage_key)
        root.mkdir(parents=True, exist_ok=True)
        meta_path = root / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "owner": snapshot.target.owner,
                    "repo": snapshot.target.repo,
                    "ref": snapshot.ref,
                    "commit_sha": snapshot.commit_sha,
                    "origin_url": snapshot.target.canonical_repo_url,
                    "files": [item.relative_path for item in snapshot.files],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        results: list[IngestResult] = []
        for item in snapshot.files:
            dest = (root / "tree" / item.relative_path).resolve()
            if not dest.is_relative_to(root.resolve()):
                raise ValueError("github file path escaped the snapshot directory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(item.text + "\n", encoding="utf-8")

            existing = self._catalog.get_by_path(dest)
            content_hash = hash_content(item.text)
            if (
                existing is not None
                and existing.source.content_hash == content_hash
                and self._index_is_complete(existing.source.id)
            ):
                chunks = self._catalog.get_chunks(existing.source.id)
                results.append(IngestResult(existing.source, len(chunks), "skipped"))
                continue

            if existing is None:
                source_id = str(uuid4())
                status: IngestStatus = "ingested"
            else:
                source_id = existing.source.id
                status = "updated"

            title = f"{snapshot.target.owner}/{snapshot.target.repo}:{item.relative_path}"
            source = Source(
                id=source_id,
                path=dest,
                title=title,
                text=item.text,
                content_hash=content_hash,
                pages=item.pages,
                origin_url=item.origin_url,
            )
            try:
                results.append(self._commit_source(source, status, was_new=existing is None))
            except Exception:
                if existing is None and dest.is_file():
                    dest.unlink(missing_ok=True)
                raise

        return _summarize_github_results(results, snapshot.target.canonical_repo_url)

    def _ingest_web_page(self, url: str) -> IngestResult:
        page = fetch_web_page(url)
        dest = self._destination_for_url(page.url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page.text + "\n", encoding="utf-8")

        existing = self._catalog.get_by_path(dest)
        content_hash = hash_content(page.text)

        if (
            existing is not None
            and existing.source.content_hash == content_hash
            and self._index_is_complete(existing.source.id)
        ):
            chunks = self._catalog.get_chunks(existing.source.id)
            return IngestResult(existing.source, len(chunks), "skipped")

        if existing is None:
            source_id = str(uuid4())
            status: IngestStatus = "ingested"
        else:
            source_id = existing.source.id
            status = "updated"

        source = Source(
            id=source_id,
            path=dest,
            title=page.title,
            text=page.text,
            content_hash=content_hash,
            pages=(),
            origin_url=page.url,
        )
        try:
            return self._commit_source(source, status, was_new=existing is None)
        except Exception:
            if existing is None and dest.is_file():
                dest.unlink(missing_ok=True)
            raise

    def _ingest_youtube(self, url: str) -> IngestResult:
        video = fetch_youtube_video(url, transcriber=self._get_transcriber())
        dest = self._destination_for_youtube(video.video_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(video.text + "\n", encoding="utf-8")

        existing = self._catalog.get_by_path(dest)
        content_hash = hash_content(video.text)

        if (
            existing is not None
            and existing.source.content_hash == content_hash
            and self._index_is_complete(existing.source.id)
        ):
            chunks = self._catalog.get_chunks(existing.source.id)
            return IngestResult(existing.source, len(chunks), "skipped")

        if existing is None:
            source_id = str(uuid4())
            status: IngestStatus = "ingested"
        else:
            source_id = existing.source.id
            status = "updated"

        source = Source(
            id=source_id,
            path=dest,
            title=video.title,
            text=video.text,
            content_hash=content_hash,
            pages=video.pages,
            origin_url=video.url,
        )
        try:
            return self._commit_source(source, status, was_new=existing is None)
        except Exception:
            if existing is None and dest.is_file():
                dest.unlink(missing_ok=True)
            raise

    def remove(self, source_id: str) -> bool:
        """Drop a document from both stores. Returns False if the id was unknown.

        Vectors first, then the catalog: if the catalog delete fails, the next ingest
        sees an incomplete index and repairs. The opposite order would leave search hits
        that SQLite cannot load.

        Files the API stored under `data/files/` are deleted. Files the CLI indexed in
        place are left on disk — they belong to the user, not the library.
        """
        record = self._catalog.get(source_id)
        if record is None:
            return False

        self._index.delete_by_source(source_id)
        self._catalog.delete(source_id)
        self._delete_owned_file(record.source.path)
        return True

    def search(
        self, question: str, top_k: int = DEFAULT_TOP_K
    ) -> tuple[list[SearchResult], dict[str, Source]]:
        """Qdrant finds ids; SQLite loads the passages. That join is this method."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        hits = self._index.search(self._embedder.embed_query(question), top_k=top_k)
        if not hits:
            return [], {}

        chunks = self._catalog.get_chunks_by_ids([hit.chunk_id for hit in hits])
        results = [
            SearchResult(chunk=chunk, score=hit.score)
            for chunk, hit in zip(chunks, hits, strict=True)
        ]
        results = self._expand_neighbors(results)

        sources: dict[str, Source] = {}
        for result in results:
            source_id = result.chunk.source_id
            if source_id in sources:
                continue
            record = self._catalog.get(source_id)
            if record is None:
                raise KeyError(f"chunk {result.chunk.id} refers to missing source {source_id}")
            sources[source_id] = record.source

        return results, sources

    def ask(self, question: str, model: ChatModel, top_k: int = DEFAULT_TOP_K) -> Answer:
        results, sources = self.search(question, top_k=top_k)
        return answer_question(question, results, sources, model)

    def ask_stream(
        self, question: str, model: ChatModel, top_k: int = DEFAULT_TOP_K
    ) -> Iterator[AnswerToken | Answer]:
        results, sources = self.search(question, top_k=top_k)
        yield from answer_question_stream(question, results, sources, model)

    def list_sources(self) -> list[SourceRecord]:
        return self._catalog.list_sources()

    def chunk_count(self, source_id: str) -> int:
        return len(self._catalog.get_chunks(source_id))

    def _expand_neighbors(
        self, results: list[SearchResult], radius: int = NEIGHBOR_RADIUS
    ) -> list[SearchResult]:
        """Include chunks next to each hit so a split list stays whole in the prompt.

        Vector search ranks by similarity. A "top 5 books" section often lives across
        several consecutive chunks; only some of them look like the question. Pulling
        ordinal ± radius from the same source fills the gaps without changing chunking.
        Neighbors keep a slightly lower score so `select_within_budget` still prefers
        the original hits when the excerpt budget is tight.
        """
        if radius < 1 or not results:
            return results

        seen = {result.chunk.id for result in results}
        siblings_by_source: dict[str, list[Chunk]] = {}
        expanded = list(results)

        for result in results:
            source_id = result.chunk.source_id
            if source_id not in siblings_by_source:
                siblings_by_source[source_id] = self._catalog.get_chunks(source_id)
            for sibling in siblings_by_source[source_id]:
                if sibling.id in seen:
                    continue
                if abs(sibling.ordinal - result.chunk.ordinal) > radius:
                    continue
                seen.add(sibling.id)
                expanded.append(SearchResult(chunk=sibling, score=result.score * 0.999))

        expanded.sort(key=lambda item: (-item.score, item.chunk.source_id, item.chunk.ordinal))
        return expanded

    def _commit_source(
        self, source: Source, status: IngestStatus, *, was_new: bool
    ) -> IngestResult:
        chunks = chunk_document(source, self._chunking)
        embedded = self._embedder.embed_chunks(chunks) if chunks else []
        meta = IngestMeta(
            embedding_model=self._embedder.model,
            chunk_max_chars=self._chunking.max_chars,
            chunk_overlap=self._chunking.overlap_chars,
        )
        try:
            # Catalog first (source of truth), then the index. Embeddings are already in
            # memory, so a failed Ollama call never leaves a half-written document.
            self._catalog.save(source, chunks, meta)
            self._index.delete_by_source(source.id)
            if embedded:
                self._index.add(embedded)
        except Exception:
            self._index.delete_by_source(source.id)
            if was_new:
                self._catalog.delete(source.id)
            raise
        return IngestResult(source, len(chunks), status)

    def _index_is_complete(self, source_id: str) -> bool:
        return self._index.count_by_source(source_id) == len(self._catalog.get_chunks(source_id))

    def _destination_for_url(self, url: str) -> Path:
        files_dir = owned_files_dir(self._data_dir).resolve()
        dest = (files_dir / "web" / url_storage_name(url)).resolve()
        if not dest.is_relative_to(files_dir):
            raise ValueError("url storage path escaped the files directory")
        return dest

    def _destination_for_youtube(self, video_id: str) -> Path:
        files_dir = owned_files_dir(self._data_dir).resolve()
        dest = (files_dir / "youtube" / youtube_storage_name(video_id)).resolve()
        if not dest.is_relative_to(files_dir):
            raise ValueError("youtube storage path escaped the files directory")
        return dest

    def _destination_for_github(self, storage_key: str) -> Path:
        files_dir = owned_files_dir(self._data_dir).resolve()
        dest = (files_dir / "github" / storage_key).resolve()
        if not dest.is_relative_to(files_dir):
            raise ValueError("github storage path escaped the files directory")
        return dest

    def _destination_for(self, filename: str) -> Path:
        name = Path(filename).name
        if not name or name in {".", ".."}:
            raise ValueError("filename is empty")
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise UnsupportedFileType(f"{name}: expected one of {sorted(SUPPORTED_SUFFIXES)}")

        files_dir = owned_files_dir(self._data_dir).resolve()
        dest = (files_dir / name).resolve()
        if not dest.is_relative_to(files_dir):
            raise ValueError("filename is not a simple file name")
        return dest

    def _delete_owned_file(self, path: Path) -> None:
        files_dir = owned_files_dir(self._data_dir).resolve()
        resolved = path.resolve()
        if resolved.is_relative_to(files_dir) and resolved.is_file():
            resolved.unlink()


def _summarize_github_results(results: list[IngestResult], repo_url: str) -> IngestResult:
    """Pick one representative result for the single-`IngestOut` HTTP/CLI contract.

    Prefer a README when present; otherwise the first file. Aggregate status across the
    batch so a fully-cached re-ingest reports `skipped`.
    """
    if not results:
        raise ValueError(f"{repo_url} produced no ingest results")

    preferred = next(
        (
            item
            for item in results
            if item.source.title.lower().endswith(":readme.md")
            or item.source.title.lower().endswith(":readme.markdown")
            or item.source.title.lower().endswith(":readme.rst")
            or item.source.title.lower().endswith(":readme")
        ),
        results[0],
    )
    statuses = {item.status for item in results}
    if statuses == {"skipped"}:
        status: IngestStatus = "skipped"
    elif "ingested" in statuses:
        status = "ingested"
    else:
        status = "updated"
    chunk_count = sum(item.chunk_count for item in results)
    return IngestResult(preferred.source, chunk_count, status)
