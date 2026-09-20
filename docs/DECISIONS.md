# Architecture Decision Record

Each entry records a decision that would be expensive or confusing to reverse later, and
*why* we made it. Entries are append-only: if we change our mind, we add a new entry that
supersedes the old one rather than editing history.

---

## ADR-0001 — VoiceLM gets its own git repository

**Date:** 2026-09-20 · **Status:** Accepted

**Context.** The project folder sat inside a git repository rooted at
`~/Documents/Development`, which had no commits and around twenty unrelated sibling
projects. Committing there would entangle VoiceLM's history with every other project on
the machine.

**Decision.** `git init` a dedicated repository at `~/Documents/Development/voicelm`, and
add `voicelm/` to the outer repository's `.gitignore` so the outer repo does not track it
as a gitlink. Also renamed the folder from `voiceLM` to `voicelm` to match the Python
package name and avoid case-sensitivity bugs.

**Alternatives.** Moving the project outside the parent entirely — cleaner, but changes
where the project lives for no functional gain once it is ignored.

---

## ADR-0002 — Monorepo with a hard UI/backend boundary

**Date:** 2026-09-20 · **Status:** Accepted

**Context.** VoiceLM needs a desktop UI now and potentially mobile or web clients later.
All clients must share one AI implementation.

**Decision.** One repository containing `backend/` (Python) and `desktop/` (Flutter) as
independently runnable applications. The backend must remain fully runnable and testable
with `desktop/` deleted. The UI communicates only through the backend's HTTP API.

**Consequences.** Client reuse becomes a property enforced by the code rather than an
intention. The cost is that the two halves are built and run separately during
development.

**Alternatives.** Separate repositories (synchronizing API changes across repos is
painful at this stage); embedding Python in Flutter via FFI (welds them together and
destroys the reuse goal).

---

## ADR-0003 — Python 3.13 managed by uv

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** Pin the backend to Python 3.13 (already the machine's default) and manage
dependencies with `uv`, committing `uv.lock`.

**Why uv.** It replaces `pip` + `venv` + `pip-tools` with one fast tool and produces a
lockfile recording the exact version of every direct and transitive dependency. Without a
lockfile, a `sync` today and a `sync` in three months install different code, producing
bugs that cannot be reproduced.

**Known risk.** Machine learning libraries ship precompiled binary wheels per Python
version. When a wheel is missing, installation falls back to compiling from source, which
on macOS surfaces as cryptic compiler errors unrelated to our code. Python 3.12 currently
has broader coverage for the audio/ML stack that Phase 3 will need.

**Accepted mitigation.** The escape hatch is cheap: change `.python-version` to `3.12`
and re-run `uv sync`. We take the small risk now to avoid an unnecessary install, with
the fallback documented.

**Alternatives.** `poetry` (mature but slower, and its resolver has historically
struggled with ML packages); bare `pip` (no lockfile); `conda` (good at thorny binary
dependencies, but heavy and largely unnecessary on Apple Silicon).

---

## ADR-0004 — FastAPI, with REST plus Server-Sent Events

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** FastAPI over localhost. Ordinary REST for ingestion and search;
Server-Sent Events for streaming answer tokens.

**Why.** FastAPI derives request validation and interactive API documentation from
ordinary Python type hints, so the contract cannot silently drift from the code. SSE lets
tokens appear as the model produces them instead of the user watching a spinner for
twenty seconds.

**Alternatives.** gRPC (stronger typing and faster, but adds a code-generation build step
and real learning overhead for negligible gain over loopback); WebSockets (will be needed
in Phase 3 for bidirectional audio, but overkill for Phase 1's one-directional
streaming).

---

## ADR-0005 — Ollama installed natively, never in Docker

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** Install the native macOS Ollama app.

**Why.** Docker containers on macOS run inside a Linux virtual machine that has **no
access to the Apple Silicon GPU**. A model generating roughly 40 tokens/second using
Metal natively drops to single digits on CPU-only inside Docker. This is a ten-fold
performance difference, which is the gap between usable and unusable.

---

## ADR-0006 — Qdrant for vectors, SQLite for metadata

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** Two stores. Qdrant holds embedding vectors; SQLite holds document and chunk
metadata (source file, page number, character offsets, import timestamp).

**Why.** Qdrant answers *which* chunks are relevant; SQLite answers *where they came
from*. A citation is precisely the combination. Using one store for both means either
doing vector math in SQL — slow, and you hand-roll approximate nearest neighbor search —
or keeping relational metadata in a vector database, which is not its purpose.

**Consequence.** Two writes per chunk, and we are responsible for keeping them
consistent.

---

## ADR-0007 — Start with qdrant-client local mode, not Docker

**Date:** 2026-09-20 · **Status:** Accepted

**Context.** Docker is installed but its daemon is not running, and Docker Desktop
requires a paid license for commercial use.

**Decision.** Use `qdrant-client`'s local mode, which runs the vector engine in-process
against a directory with no server and no container. Migrate to a real Qdrant server at
Milestone 2.

**Why.** Identical Python API, so the migration is a configuration change rather than a
rewrite. It removes an entire category of "is the container running?" debugging while the
retrieval concepts are still being learned.

**Tradeoff accepted.** Local mode is not production-representative and does not exercise
the network path or persistence behaviour of the real server.

---

## ADR-0008 — Every vector records its embedding model and chunking parameters

**Date:** 2026-09-20 · **Status:** Accepted

**Context.** Vectors produced by different embedding models are not comparable. Changing
the embedding model invalidates *every vector ever stored* and requires re-embedding the
entire corpus.

**Decision.** From the very first write, store alongside each vector: the embedding model
name, its version, the vector dimensionality, and the chunking parameters used.

**Why.** This is the cheapest possible piece of foresight and it converts a future model
upgrade from "delete everything and hope" into a controlled, verifiable reindex. It also
makes it possible to detect a mismatched index instead of silently returning nonsense.

---

## ADR-0009 — No user document content leaves the device

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** A hard rule: no document content, chunk, embedding, or transcript is sent
to any non-local network destination. All services bind to loopback only.

**Why written down now.** "Privacy-first" as an intention loses to convenience the first
time a cloud embedding API would save an afternoon. Deciding this while no cloud
dependency exists is free; retrofitting it after one is load-bearing is not.

**How it will be enforced.** Eventually, a test that fails if the backend attempts a
network connection to anything other than localhost.

---

## ADR-0010 — Concrete implementations; no speculative interfaces

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** Write concrete classes such as `QdrantVectorStore` directly. Do not define
abstract interfaces until a second implementation actually exists.

**Why.** An interface with exactly one implementation is speculation. It adds a layer of
indirection a reader must trace through for zero present benefit. Extracting an interface
later is a mechanical refactor that tooling can largely perform; guessing the right
abstraction in advance is not.

---

## ADR-0011 — `src/` layout for the Python package

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** The package lives at `backend/src/voicelm/` rather than `backend/voicelm/`.

**Why.** With a `src/` layout, tests cannot accidentally import the package from the
current working directory — they are forced to import the *installed* package. This
catches "works when I run it here, breaks when packaged" bugs immediately, which matters
because Phase 1 ends with packaging as a real concern. It costs nothing.
