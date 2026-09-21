# VoiceLM — Architecture

Last updated: 2026-09-20

## The one rule

**The UI talks only to the backend HTTP API. Nothing else.**

The Flutter app never opens a Qdrant connection, never calls Ollama, and never reads the
SQLite file. If it could, knowledge of our storage schema would leak into the UI, and a
future iOS client would have to reimplement all of it. One doorway is what makes client
reuse possible.

## System diagram

```
┌──────────────────────────────────────┐
│  Flutter desktop app (macOS)         │   Presentation only.
│                                      │   No AI logic, no direct DB access.
└──────────────┬───────────────────────┘
               │  HTTP/JSON  +  Server-Sent Events for token streaming
               │  127.0.0.1:8000
┌──────────────▼───────────────────────────────────────┐
│  Python backend — FastAPI                            │
│                                                      │
│   ingestion → embeddings → retrieval → generation    │
│                                                      │
└────┬──────────────────┬──────────────────┬───────────┘
     │ HTTP             │ HTTP             │ in-process
     │ :11434           │ :6333            │ (no network)
┌────▼─────────┐  ┌─────▼────────┐  ┌──────▼──────────────┐
│  Ollama      │  │  Qdrant      │  │  SQLite             │
│  local LLM   │  │  vectors     │  │  metadata + sources │
└──────────────┘  └──────────────┘  └─────────────────────┘
```

## Components

### Flutter desktop app (`desktop/`)
Renders the UI and calls the backend API. Contains no AI logic. Added in a later
milestone — there is deliberately nothing to show until the engine works.

### Python backend (`backend/`)
The entire product brain. Runs as a standalone process and is useful without any UI.

| Module | Responsibility |
|---|---|
| `domain/` | Plain data types: `Document`, `Chunk`, `Citation`. No I/O, depends on nothing. |
| `ingestion/` | Extract text from files, clean it, split it into chunks. |
| `embeddings/` | Turn text into vectors. |
| `retrieval/` | Given a question vector, find the most relevant chunks. |
| `generation/` | Assemble the prompt, call the LLM, attach citations. |
| `storage/` | SQLite and Qdrant access. |
| `api/` | FastAPI routes — a thin shell over the modules above. |

`domain/`, `ingestion/`, and `api/` exist today. The rest is created as each milestone
needs it, rather than as empty folders up front.

### The ingestion pipeline

```
load_source(path)          read UTF-8, reject anything else (ADR-0015)
      │                    id = sha256 of content, so re-ingesting is idempotent
      ▼
clean_text(raw)            NFC, line endings, invisible chars, blank-line runs
      │                    conservative: intra-line whitespace untouched (ADR-0013)
      ▼
chunk_document(source)     paragraphs → sentences → hard cut (ADR-0014)
      │                    overlap by extending each chunk's start backwards
      ▼
list[Chunk]                each carrying source_id, offsets, and ordinal
```

Chunking works entirely in `(start, end)` character spans and only slices out strings at
the end. That is what makes the provenance invariant true by construction rather than by
convention:

    source.text[chunk.start_char : chunk.end_char] == chunk.text

### Ollama — local LLM
A local HTTP server that runs quantized language models using the Mac's Metal GPU. Our
backend is just an HTTP client to it. Installed natively rather than in Docker, because
Docker on macOS runs inside a Linux VM with no GPU access (ADR-0005).

### Qdrant — vector database
Stores embedding vectors and answers "which stored text is semantically closest to this
question?" using approximate nearest neighbor search. In Phase 1 this runs in-process
via `qdrant-client`'s local mode, with the same API as the real server (ADR-0007).

### SQLite — metadata database
**Not a server.** A library that reads and writes a single file on disk, built into
Python. Stores the facts that make citations possible: which file a chunk came from, its
page number, its character offsets, when it was imported.

## Why two databases

Qdrant answers *which* chunks are relevant. SQLite answers *where they came from*. A
citation is exactly the combination of those two answers.

Collapsing them into one system means either doing vector math in SQL — slow, and you
hand-roll the nearest-neighbor algorithm — or storing relational metadata in a vector
database, which is not what it is built for.

## Data flow

### Ingestion (once per document)

```
file → extract raw text → clean → split into chunks → embed each chunk
                                        │                    │
                                        ▼                    ▼
                          SQLite: chunk metadata    Qdrant: vector + chunk id
                          (source, page, offsets)   (+ embedding model & version)
```

The critical invariant: **provenance is attached at chunk creation and never lost.** If
a chunk's source and page are dropped during ingestion, no amount of clever prompting
later can recover them, and citations become impossible.

### Query (once per question)

```
question → embed → Qdrant: find top-k nearest chunk ids
                              │
                              ▼
                   SQLite: load those chunks + their sources
                              │
                              ▼
         build prompt: "answer using ONLY these excerpts"
                              │
                              ▼
                   Ollama → stream tokens → SSE → Flutter
                              │
                              ▼
               attach citations from the chunks actually used
```

Note what this implies: **the model is never trained on your documents.** There is no
fine-tuning and no learning. Every question is a fresh, stateless prompt that happens to
contain relevant excerpts pasted into it. Most of the design above follows from that one
fact.

## Process and port map

| Process | How it starts | Address |
|---|---|---|
| Python backend | Manually, `uv run uvicorn ...` | `127.0.0.1:8000` |
| Ollama | macOS background service | `127.0.0.1:11434` |
| Qdrant | In-process (Phase 1) | none — local mode |
| SQLite | In-process | a file on disk |
| Flutter app | `flutter run` | n/a |

All addresses are loopback only. Nothing binds to a public interface.

## Deliberately deferred

Recorded so these read as decisions rather than oversights:

- **Packaging Python inside a distributable macOS `.app`.** Bundling an interpreter plus
  ML dependencies is genuinely hard. During development you start the backend by hand,
  which is also better for learning because you see the logs.
- **Authentication.** Single-user, loopback-only. Adding auth now would be ceremony.
- **Real Qdrant server.** Local mode until Milestone 2 (ADR-0007).
- **Abstract interfaces over storage and models.** Concrete classes until a second
  implementation actually exists (ADR-0010).
