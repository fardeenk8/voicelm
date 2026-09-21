"""Command line entry point.

Deliberately thin: it parses arguments, calls KnowledgeBase, and prints. Progress goes
to stderr and the answer goes to stdout, so the answer can be piped without the
progress lines.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from voicelm.domain.models import Answer, Citation
from voicelm.embeddings.ollama import DEFAULT_MODEL as DEFAULT_EMBED_MODEL
from voicelm.embeddings.ollama import EmbeddingError, OllamaEmbedder
from voicelm.generation.ollama import DEFAULT_MODEL as DEFAULT_CHAT_MODEL
from voicelm.generation.ollama import GenerationError, OllamaChatModel
from voicelm.ingestion.chunking import ChunkingConfig
from voicelm.ingestion.loader import UndecodableFile, UnsupportedFileType
from voicelm.ingestion.pdf import PdfExtractionError
from voicelm.knowledge import KnowledgeBase

QUOTE_PREVIEW_CHARS = 220


def default_data_dir() -> Path:
    """Where the catalog and vector index live.

    Override with VOICELM_DATA_DIR or --data-dir. Default is ./data in the current
    working directory (typically backend/).
    """
    raw = os.environ.get("VOICELM_DATA_DIR")
    return Path(raw) if raw else Path.cwd() / "data"


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
        required=True,
        type=Path,
        metavar="PATH",
        help="a .txt, .md, or .pdf file; repeat for several",
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
    ask.add_argument("--top-k", type=int, default=5, help="excerpts to retrieve (default 5)")
    ask.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)

    subcommands.add_parser("sources", help="list documents in the local library")

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
        raise ValueError(f"unknown command {arguments.command}")
    except (
        UnsupportedFileType,
        UndecodableFile,
        PdfExtractionError,
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
    for path in arguments.source:
        started = time.monotonic()
        result = base.ingest(path)
        elapsed = time.monotonic() - started
        plural = "chunk" if result.chunk_count == 1 else "chunks"
        print(
            f"  {path.name}: {result.status}, {result.chunk_count} {plural} ({elapsed:.1f}s)",
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
        print(f"  {record.source.title}  {record.source.path}  {chunks} {plural}")
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
    chars = f"characters {citation.start_char}-{citation.end_char}"
    if not citation.pages:
        return chars
    if len(citation.pages) == 1:
        return f"page {citation.pages[0]}, {chars}"
    return f"pages {citation.pages[0]}–{citation.pages[-1]}, {chars}"


if __name__ == "__main__":
    raise SystemExit(main())
