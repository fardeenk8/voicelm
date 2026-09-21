"""Command line entry point.

Deliberately thin: it parses arguments, calls the modules that hold the logic, and prints.
The same functions will sit behind HTTP endpoints later without being rewritten, which only
works because none of the logic lives here.

Progress goes to stderr and the answer goes to stdout, so the answer can be piped
somewhere useful without the progress lines coming along.
"""

import argparse
import sys
import time
from pathlib import Path

from voicelm.domain.models import Answer, Source
from voicelm.embeddings.ollama import DEFAULT_MODEL as DEFAULT_EMBED_MODEL
from voicelm.embeddings.ollama import EmbeddingError, OllamaEmbedder
from voicelm.generation.answering import answer_question
from voicelm.generation.ollama import DEFAULT_MODEL as DEFAULT_CHAT_MODEL
from voicelm.generation.ollama import GenerationError, OllamaChatModel
from voicelm.ingestion.chunking import ChunkingConfig, chunk_document
from voicelm.ingestion.loader import UndecodableFile, UnsupportedFileType, load_source
from voicelm.retrieval.store import InMemoryVectorStore

QUOTE_PREVIEW_CHARS = 220


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicelm", description="Ask questions about your own documents, locally."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    ask = subcommands.add_parser("ask", help="ask a question about one or more documents")
    ask.add_argument("question")
    ask.add_argument(
        "--source",
        action="append",
        required=True,
        type=Path,
        metavar="PATH",
        help="a .txt or .md file to search; repeat for several",
    )
    ask.add_argument("--top-k", type=int, default=5, help="excerpts to retrieve (default 5)")
    # Exposed because chunk size is the main lever on citation precision: too large and a
    # citation points at most of the document, too small and excerpts lose their context.
    ask.add_argument(
        "--chunk-chars",
        type=int,
        default=ChunkingConfig.max_chars,
        help=f"characters per chunk (default {ChunkingConfig.max_chars})",
    )
    ask.add_argument(
        "--chunk-overlap",
        type=int,
        default=ChunkingConfig.overlap_chars,
        help=f"characters repeated between chunks (default {ChunkingConfig.overlap_chars})",
    )
    ask.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ask.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    try:
        return _ask(arguments)
    except (UnsupportedFileType, UndecodableFile, FileNotFoundError) as error:
        print(f"error reading source: {error}", file=sys.stderr)
    except (EmbeddingError, GenerationError) as error:
        print(f"error talking to Ollama: {error}", file=sys.stderr)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)

    return 1


def _ask(arguments: argparse.Namespace) -> int:
    embedder = OllamaEmbedder(model=arguments.embed_model)
    store = InMemoryVectorStore()
    sources: dict[str, Source] = {}
    config = ChunkingConfig(max_chars=arguments.chunk_chars, overlap_chars=arguments.chunk_overlap)

    # Ingestion happens on every run because nothing is persisted yet. Milestone 1B adds
    # storage so this becomes a one-time cost per document.
    for path in arguments.source:
        source = load_source(path)
        chunks = chunk_document(source, config)

        if not chunks:
            print(f"  {path.name}: no text found, skipping", file=sys.stderr)
            continue

        started = time.monotonic()
        store.add(embedder.embed_chunks(chunks))
        sources[source.id] = source
        elapsed = time.monotonic() - started
        plural = "chunk" if len(chunks) == 1 else "chunks"
        print(
            f"  {path.name}: {len(chunks)} {plural} embedded in {elapsed:.1f}s",
            file=sys.stderr,
        )

    if not sources:
        print("no readable text in any source", file=sys.stderr)
        return 1

    print(f"  searching {len(store)} chunks...", file=sys.stderr)
    results = store.search(embedder.embed_query(arguments.question), top_k=arguments.top_k)

    model = OllamaChatModel(model=arguments.chat_model)
    answer = answer_question(arguments.question, results, sources, model)

    _print_answer(answer)
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
        # No markers means the answer is either a refusal or ungrounded. Either way the
        # user should know it is not backed by a specific passage.
        print()
        print("(no citations — this answer is not backed by a specific passage)")
        return

    print()
    print("Sources:")
    for citation in answer.citations:
        preview = " ".join(citation.quote.split())[:QUOTE_PREVIEW_CHARS]
        print(
            f"  [{citation.marker}] {citation.source_title}, "
            f"characters {citation.start_char}-{citation.end_char}"
        )
        print(f'      "{preview}..."')


if __name__ == "__main__":
    raise SystemExit(main())
