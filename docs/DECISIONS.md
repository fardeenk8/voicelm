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

**Amended by ADR-0012**, which keeps the layout but stops relying on the editable install
for imports.

---

## ADR-0012 — Import from the source tree, not the editable install

**Date:** 2026-09-20 · **Status:** Accepted · **Amends:** ADR-0011

**Context.** The editable install was silently broken on macOS. An editable install works
by writing a `.pth` file into `site-packages` containing the path to the source directory,
which Python's `site` module reads at startup. Ours contained the correct path to an
existing directory and was still ignored, because:

1. uv marks `.venv` as hidden on macOS so it does not clutter Finder.
2. macOS propagates the `UF_HIDDEN` flag to files created inside a hidden directory.
3. CPython's `site.addpackage()` skips hidden `.pth` files — silently, by design.

Upgrading uv (0.12.5 → 0.12.17) stopped *newly created* venvs from being affected, but
reinstalling the package into an existing venv still reintroduced the flag.

**Decision.** Do not depend on the `.pth` mechanism. Set `pythonpath = ["src"]` in the
pytest configuration, run the server with `uvicorn --app-dir src`, and invoke the CLI with
`PYTHONPATH=src`. The console script installed by `[project.scripts]` imports the package
the ordinary way, so it needs the same help.

**Consequences.** The repository now works on a fresh checkout with no manual step and no
dependence on an OS flag, a uv version, and a CPython behaviour all lining up. The cost is
real: we lose ADR-0011's guarantee that tests exercise the *installed* package rather than
the source tree. We plan to recover that deliberately in a later milestone with a
packaging check that builds the wheel, installs it into a throwaway environment, and runs
the tests against it — which is a better test of packaging than an editable install ever
was.

**Alternatives.** `chflags -R nohidden .venv` fixes it in one command and preserves
ADR-0011, but must be repeated every time the environment is recreated, and forgetting it
reproduces a failure whose cause is invisible. Not acceptable as the primary mechanism.

---

## ADR-0013 — Cleaned text is canonical; offsets index into it

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** `Source.text` holds the *cleaned* text, and chunk offsets index into it.
Cleaning is conservative: normalise Unicode to NFC, normalise line endings, strip
invisible characters, strip trailing whitespace per line, and collapse runs of blank lines
to one. It does **not** touch whitespace inside a line.

**Why cleaned text is canonical.** The alternative is to keep offsets pointing into the
original file, which is more faithful but requires maintaining an edit map through every
cleaning transformation. Not worth the complexity when the cleaned text is what we quote
back to the user anyway.

**Why conservative.** Indentation is meaningful in Markdown — code blocks, nested lists —
and because citations quote cleaned text directly, mangled indentation would be visible in
the product. NFC normalisation matters more than it looks: "é" can be one code point or
"e" plus a combining accent, which are visually identical but produce different
embeddings for the same word.

---

## ADR-0014 — Chunk at paragraph boundaries, then sentences, then by force

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** Split on paragraph breaks first, since a paragraph is usually one idea.
Split an oversized paragraph at sentence ends. Only hard-cut a single sentence that still
exceeds the limit. Defaults: 1000 characters with 150 characters of overlap.

Overlap is implemented by extending a chunk's *start* backwards rather than concatenating
text, which keeps every chunk one contiguous slice of the document and preserves the
invariant `source.text[chunk.start_char:chunk.end_char] == chunk.text`. A chunk's length
may therefore reach `max_chars + overlap_chars`.

**Consequence worth recording.** Chunks do not tile the document exactly: the blank line
between two paragraphs is dropped so chunk text never begins or ends with a decorative
gap. The invariant we assert is therefore the weaker but correct one — anything falling
between two consecutive chunks is whitespace only, so no content is lost.

**Alternatives.** Fixed-size character windows (simplest, but cuts mid-sentence and
degrades both retrieval and citation quality); token-based sizing (more accurate against
the context window, but requires a tokeniser and couples chunking to a specific model);
semantic chunking, which embeds sentences and splits where similarity drops (expensive,
unpredictable, and impossible to reason about when retrieval goes wrong).

---

## ADR-0015 — Reject non-UTF-8 files rather than guessing

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** `load_source` reads UTF-8 strictly and raises `UndecodableFile` otherwise.

**Why.** Legacy encodings such as cp1252 are common, and guessing is possible. But a wrong
guess silently corrupts text, corrupted text gets embedded and cited, and the user sees a
confident quotation of mojibake with no indication anything went wrong. A clear error the
user can act on is better than silent corruption. Encoding detection can be added later as
an explicit, visible step.

---

## ADR-0016 — nomic-embed-text for embeddings, llama3.1:8b for generation

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** Both models served by Ollama: `nomic-embed-text` (768 dimensions, ~274 MB)
for embeddings and `llama3.1:8b` (~4.9 GB, 4-bit quantized) for generation.

**Why.** Ollama is required for generation regardless, so using it for embeddings too
means one new tool instead of two and one consistent mental model: all inference is an
HTTP call to a local model server. Llama 3.1 is chosen over Qwen 2.5 and Gemma 2 mainly
for documentation density — when it misbehaves, others have written about it.

**Alternatives.** `fastembed` running `bge-small-en-v1.5` scores better on retrieval
benchmarks with smaller 384-dimension vectors and no PyTorch; it is the likely upgrade.
`sentence-transformers` has the widest model selection but pulls in PyTorch at 2–3 GB,
which the available disk cannot spare.

**Why this is reversible.** ADR-0008 requires every stored vector to record the model that
produced it, which turns switching embedding models into a controlled reindex rather than
silent corruption. This decision is where that foresight earns its keep.

**Operational note, corrected against the installed version.** Ollama does not use the
model's full context window by default. On this machine (Ollama 0.34.2, Apple M4, 11.8 GiB
of usable unified memory) the server log reports `vram-based default context
default_num_ctx=4096`, so the default is derived from available memory rather than being a
fixed 2048 as older versions used. Either way it is far below the 128k Llama 3.1 supports,
so `num_ctx` must be set explicitly — otherwise retrieved excerpts are silently truncated
and answers degrade with no error.

---

## ADR-0017 — Brute-force similarity search before Qdrant

**Date:** 2026-09-20 · **Status:** Accepted

**Decision.** Milestone 1 keeps embeddings in a Python list and scores every one with
cosine similarity. Qdrant arrives in Milestone 1B.

**Why.** It is roughly fifteen lines, it is fast enough at this scale (comparing a query
against a few hundred chunks is sub-millisecond), and writing it means understanding what
a vector database actually provides instead of treating it as magic. Approximate nearest
neighbour search matters at a million vectors, and we should adopt it when we can measure
why.

**Why it is not throwaway.** The naive implementation is retained as a **test oracle**:
when Qdrant is introduced, we assert it returns the same ranking. Validating a fast
implementation against a simple reference is a practice worth keeping.

---

## ADR-0018 — Remove citation markers that match no excerpt

**Date:** 2026-09-20 · **Status:** Accepted

**Context.** Found by a live test, not by reasoning. Given a prompt containing a single
excerpt numbered `[1]`, `llama3.1:8b` answered `"Tuesdays [2]."` — citing a number that did
not exist. Our validation correctly refused to fabricate a citation for `[2]`, but the
answer text still displayed `[2]` while no source 2 was listed, which reads as a bug in
VoiceLM rather than a limitation of the model.

**Decision.** Markers that match no excerpt are removed from the answer text, along with
the whitespace before them, so `"Tuesdays [2]."` becomes `"Tuesdays."`. The discarded
markers are recorded on `Answer.unsupported_markers` and surfaced as a note by the CLI.

**Why not map them to the nearest valid excerpt.** We cannot know what the model meant, and
guessing would produce a citation that points at text which may not support the claim —
precisely the failure this system exists to prevent. Dropping the marker while keeping the
claim is the only honest option.

**Why record rather than silently drop.** A non-empty `unsupported_markers` is direct
evidence the answer is less grounded than it appears. Hiding that would remove a signal we
will want when evaluating answer quality and comparing models.

**Invariant this establishes.** The markers visible in an answer and the citations listed
beneath it always agree. A live test asserts exactly that.

**Related finding.** Smaller chunks reduce the problem, because with several excerpts the
model has real numbers to choose from. A document that collapses into one chunk is the
worst case, and also produces citations spanning the whole document. This is why
`--chunk-chars` is exposed on the CLI.
