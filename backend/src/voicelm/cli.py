"""Command line entry point.

Deliberately thin: it parses arguments, calls KnowledgeBase, and prints. Progress goes
to stderr and the answer goes to stdout, so the answer can be piped without the
progress lines.
"""

import argparse
import sys
import time
from pathlib import Path

from voicelm.domain.models import Answer, Citation
from voicelm.embeddings.ollama import DEFAULT_MODEL as DEFAULT_EMBED_MODEL
from voicelm.embeddings.ollama import EmbeddingError, OllamaEmbedder
from voicelm.generation.ollama import DEFAULT_MODEL as DEFAULT_CHAT_MODEL
from voicelm.generation.ollama import GenerationError, OllamaChatModel
from voicelm.ingestion.chunking import ChunkingConfig
from voicelm.ingestion.github import GitHubError
from voicelm.ingestion.loader import UndecodableFile, UnsupportedFileType
from voicelm.ingestion.office import OfficeExtractionError
from voicelm.ingestion.pdf import PdfExtractionError
from voicelm.ingestion.web import WebFetchError
from voicelm.ingestion.youtube import YouTubeError
from voicelm.knowledge import KnowledgeBase
from voicelm.paths import default_data_dir

QUOTE_PREVIEW_CHARS = 220


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicelm", description="Ask questions about your own documents, locally."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="library directory (default: ./data or $VOICELM_DATA_DIR)",
    )
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)

    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="add documents to the local library")
    ingest.add_argument(
        "--source",
        action="append",
        type=Path,
        metavar="PATH",
        help="a .txt, .md, .pdf, .docx, or .pptx file; repeat for several",
    )
    ingest.add_argument(
        "--url",
        action="append",
        metavar="URL",
        help="fetch a web page, YouTube video, or GitHub repo; repeat for several",
    )
    ingest.add_argument(
        "--chunk-chars",
        type=int,
        default=ChunkingConfig.max_chars,
        help=f"characters per chunk (default {ChunkingConfig.max_chars})",
    )
    ingest.add_argument(
        "--chunk-overlap",
        type=int,
        default=ChunkingConfig.overlap_chars,
        help=f"characters repeated between chunks (default {ChunkingConfig.overlap_chars})",
    )

    ask = subcommands.add_parser("ask", help="ask a question about ingested documents")
    ask.add_argument("question")
    ask.add_argument(
        "--source",
        action="append",
        type=Path,
        metavar="PATH",
        help="ingest this file first, then ask (optional)",
    )
    ask.add_argument("--top-k", type=int, default=10, help="excerpts to retrieve (default 10)")
    ask.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)

    subcommands.add_parser("sources", help="list documents in the local library")

    remove = subcommands.add_parser("remove", help="drop a document from the local library")
    remove.add_argument("source_id", help="the id printed by `voicelm sources`")

    return parser


def main(
    argv: list[str] | None = None,
    *,
    knowledge_base: KnowledgeBase | None = None,
    chat_model: OllamaChatModel | None = None,
) -> int:
    arguments = build_parser().parse_args(argv)
    owned = knowledge_base is None

    try:
        base = knowledge_base or _open_knowledge_base(arguments)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        if arguments.command == "ingest":
            return _ingest(arguments, base)
        if arguments.command == "ask":
            return _ask(arguments, base, chat_model)
        if arguments.command == "sources":
            return _list_sources(base)
        if arguments.command == "remove":
            return _remove(arguments, base)
        raise ValueError(f"unknown command {arguments.command}")
    except (
        UnsupportedFileType,
        UndecodableFile,
        PdfExtractionError,
        OfficeExtractionError,
        WebFetchError,
        YouTubeError,
        GitHubError,
        FileNotFoundError,
    ) as error:
        print(f"error reading source: {error}", file=sys.stderr)
    except (EmbeddingError, GenerationError) as error:
        print(f"error talking to Ollama: {error}", file=sys.stderr)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
    finally:
        if owned:
            base.close()

    return 1


def _open_knowledge_base(arguments: argparse.Namespace) -> KnowledgeBase:
    data_dir = arguments.data_dir or default_data_dir()
    chunking = ChunkingConfig(
        max_chars=getattr(arguments, "chunk_chars", ChunkingConfig.max_chars),
        overlap_chars=getattr(arguments, "chunk_overlap", ChunkingConfig.overlap_chars),
    )
    return KnowledgeBase(data_dir, OllamaEmbedder(model=arguments.embed_model), chunking)


def _ingest(arguments: argparse.Namespace, base: KnowledgeBase) -> int:
    paths = arguments.source or []
    urls = arguments.url or []
    if not paths and not urls:
        print("provide at least one --source PATH or --url URL", file=sys.stderr)
        return 1

    for path in paths:
        started = time.monotonic()
        result = base.ingest(path)
        elapsed = time.monotonic() - started
        plural = "chunk" if result.chunk_count == 1 else "chunks"
        print(
            f"  {path.name}: {result.status}, {result.chunk_count} {plural} ({elapsed:.1f}s)",
            file=sys.stderr,
        )
    for url in urls:
        started = time.monotonic()
        result = base.ingest_url(url)
        elapsed = time.monotonic() - started
        plural = "chunk" if result.chunk_count == 1 else "chunks"
        print(
            f"  {result.title}: {result.status}, {result.chunk_count} {plural} ({elapsed:.1f}s)",
            file=sys.stderr,
        )
    return 0


def _ask(
    arguments: argparse.Namespace,
    base: KnowledgeBase,
    chat_model: OllamaChatModel | None,
) -> int:
    for path in arguments.source or []:
        started = time.monotonic()
        result = base.ingest(path)
        elapsed = time.monotonic() - started
        print(f"  {path.name}: {result.status} ({elapsed:.1f}s)", file=sys.stderr)

    if not base.list_sources():
        print("no documents in the library. Run: voicelm ingest --source FILE", file=sys.stderr)
        return 1

    model = chat_model or OllamaChatModel(model=arguments.chat_model)
    print("  searching...", file=sys.stderr)
    answer = base.ask(arguments.question, model, top_k=arguments.top_k)
    _print_answer(answer)
    return 0


def _list_sources(base: KnowledgeBase) -> int:
    records = base.list_sources()
    if not records:
        print("library is empty")
        return 0

    for record in records:
        chunks = base.chunk_count(record.source.id)
        plural = "chunk" if chunks == 1 else "chunks"
        where = record.source.origin_url or str(record.source.path)
        print(f"  {record.source.id}  {record.source.title}  {where}  {chunks} {plural}")
    return 0


def _remove(arguments: argparse.Namespace, base: KnowledgeBase) -> int:
    if not base.remove(arguments.source_id):
        print(f"error: no source {arguments.source_id}", file=sys.stderr)
        return 1
    print(f"  removed {arguments.source_id}", file=sys.stderr)
    return 0


def _print_answer(answer: Answer) -> None:
    print()
    print(answer.text)

    if answer.unsupported_markers:
        markers = ", ".join(f"[{marker}]" for marker in answer.unsupported_markers)
        print(
            f"  note: the model referenced {markers}, which matched no excerpt; removed",
            file=sys.stderr,
        )

    if not answer.citations:
        print()
        print("(no citations — this answer is not backed by a specific passage)")
        return

    print()
    print("Sources:")
    for citation in answer.citations:
        preview = " ".join(citation.quote.split())[:QUOTE_PREVIEW_CHARS]
        print(f"  [{citation.marker}] {citation.source_title}, {_location(citation)}")
        print(f'      "{preview}..."')


def _location(citation: Citation) -> str:
    from voicelm.ingestion.transcript import format_timestamp

    chars = f"characters {citation.start_char}-{citation.end_char}"
    parts: list[str] = []
    if citation.origin_url:
        parts.append(citation.origin_url)
    if citation.pages:
        if citation.location_kind == "timestamp":
            stamps = [format_timestamp(second) for second in citation.pages]
            if len(stamps) == 1:
                parts.append(stamps[0])
            else:
                parts.append(f"{stamps[0]}–{stamps[-1]}")
        else:
            unit = citation.location_kind or "page"
            if len(citation.pages) == 1:
                parts.append(f"{unit} {citation.pages[0]}")
            else:
                parts.append(f"{unit}s {citation.pages[0]}–{citation.pages[-1]}")
    parts.append(chars)
    return ", ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
