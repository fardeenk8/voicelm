# Learning Log

A running record of concepts covered while building VoiceLM, in the order they came up.
The goal is that this file can be reread months later and still make sense.

---

## Milestone 0 — Project skeleton (2026-09-20)

### Nested git repositories
A git repository is defined by a `.git` directory, and git attributes any file to the
*nearest* `.git` above it. VoiceLM's folder was inside a repository covering all of
`~/Documents/Development`, so commits there would have been recorded in that repo's
history alongside twenty unrelated projects. Running `git init` inside `voicelm/` creates
a nearer `.git`, which now wins. `git rev-parse --show-toplevel` is the command that tells
you which repository you are actually in — worth running whenever you are unsure.

### Virtual environments, and why global installs hurt
A virtual environment is a private folder of installed packages belonging to one project.
Without one, every project on the machine shares a single set of package versions, so
upgrading a library for project A silently breaks project B. `uv sync` creates `.venv/`
for this project alone, and `uv run <command>` executes inside it.

### Lockfile vs. dependency list
`pyproject.toml` records what we *asked for* (`fastapi>=0.115`). `uv.lock` records what we
actually *got* — the exact version of every direct and transitive dependency. Only the
lockfile makes an install reproducible, which is why it is committed to git even though it
is a generated file.

### Binary wheels
Python packages with compiled C or C++ components ship prebuilt binaries called wheels,
one per Python version and platform. When no matching wheel exists, installation tries to
compile from source and fails with compiler errors that have nothing to do with your code.
This is why the Python version is a real decision and not a detail (ADR-0003).

### The `src/` layout
Putting the package under `src/` means it is not importable from the project root, so
tests must import the *installed* package instead of accidentally picking up the local
folder. That difference is invisible until you try to ship the thing, at which point it
becomes hours of confusion.

### Type hints and FastAPI
Python does not enforce type hints at runtime — they are annotations. FastAPI reads them
via introspection and uses them to validate incoming requests and generate API
documentation automatically. So the annotation on a route function is not decoration; it
is the actual request contract.

### Test isolation and the application factory
`create_app()` builds and returns a fresh `FastAPI` instance rather than defining one
global app. Each test gets a clean application, so state left behind by one test cannot
influence another. Shared mutable state between tests produces failures that depend on
test *ordering*, which is one of the more miserable things to debug.

### What a linter is for
`ruff` checks for unused imports, undefined names, import ordering, and outdated idioms.
The value is not style policing — it is that a machine catches the boring class of mistake
so review attention goes to logic instead.

### Editable installs, and a real bug we hit
An *editable* install does not copy your code into `site-packages`. Instead it drops a
small `.pth` file there containing the path to your source directory. At startup Python's
`site` module reads every `.pth` file in `site-packages` and appends those paths to
`sys.path`. That indirection is why you can edit `src/voicelm/` and see changes
immediately without reinstalling.

Our first `uv sync` produced a broken one: `_editable_impl_voicelm.pth` existed, contained
the correct absolute path, and that directory genuinely existed — yet `import voicelm`
failed and `src` never appeared on `sys.path`.

The debugging sequence is worth remembering, because it is the general shape of narrowing
a problem down:

1. Confirm the file exists and read its exact bytes (`xxd`) — ruling out a BOM, stray
   whitespace, or a missing trailing newline.
2. Confirm the path inside it resolves (`os.path.exists`).
3. Confirm `.pth` processing works *at all* by dropping in a throwaway probe file
   containing `/tmp` and checking whether `/tmp` appears on `sys.path`. It did — which
   proved the mechanism was fine and the problem was specific to that one file.
4. Confirm the failure is stable. A targeted reinstall appeared to fix it, but the next
   command broke it again. **That intermittency was the most important clue:** it meant
   the artifact was in a bad state rather than the configuration being wrong.

5. Check the file's *metadata*, not just its contents. This is where the answer was.

**The root cause: macOS file flags.** Beyond permissions, macOS files carry *flags*, one of
which is `UF_HIDDEN` — a marker telling Finder not to display the file. Our `.pth` had it
set, which `ls -la` only hints at and `stat -f '%Sf'` shows outright.

CPython's `site.py` contains an explicit early return: if a `.pth` file has `UF_HIDDEN`
set, it is skipped **silently**, with no warning. So three independently reasonable
behaviours combined into one baffling failure:

1. uv marks `.venv` as hidden on macOS so it does not clutter Finder.
2. macOS propagates the hidden flag to files created inside a hidden directory.
3. CPython refuses to process hidden `.pth` files.

Each step is defensible alone. Together they make an editable install that looks perfect
and does nothing. Clearing the flag with `chflags -R nohidden .venv` fixes it immediately,
which is how we confirmed the diagnosis.

**Why we did not stop there.** Upgrading uv (0.12.5 → 0.12.17) stopped *fresh* venvs from
being affected, but reinstalling the package into an existing venv still reintroduced the
flag. A fix that depends on an OS quirk, a uv version, and a CPython behaviour all lining
up is not a fix. Instead we told pytest to import from the source tree directly with
`pythonpath = ["src"]`, and pass `--app-dir src` when running the server. Neither depends
on the `.pth` mechanism at all. See ADR-0012 for what that costs us and how we plan to
recover it.

The general lessons worth keeping: intermittent failures point at state, not
configuration. When a system contradicts its own visible configuration, check metadata you
have not looked at yet. And prefer a fix that removes the dependency over one that
patches the symptom.

### Reading deprecation warnings instead of ignoring them
The test run warned that using `httpx` with Starlette's `TestClient` is deprecated in
favour of `httpx2`, so we switched the dev dependency. The remaining warning comes from
inside Starlette's own code, not ours, so there is nothing for us to fix. Worth building
the habit of triaging warnings into "mine" and "not mine" rather than tuning them out
wholesale — at Milestone 0 there are two, and it is easy to stay at zero of your own.

---

## Concepts corrected before they caused trouble

**"RAG teaches the model about my documents."** It does not. There is no training, no
fine-tuning, and your documents never enter the model's weights. Retrieval-augmented
generation means: find the relevant excerpts, paste them into the prompt, and instruct the
model to answer only from them. Every question is a fresh, stateless prompt. Once this is
clear, much of the architecture explains itself — including why provenance must be tracked
in our own database rather than expected from the model.

**"SQLite needs to be installed and started."** It does not. SQLite is a library, not a
server. There is no port, no daemon, and nothing to launch. It reads and writes one file
on disk and ships inside Python's standard library. Ollama and Qdrant *are* servers;
SQLite is not, and that distinction is why the architecture diagram shows a network arrow
to two of them and not the third.

---

---

## Milestone 1, Step 1 — Ingestion (2026-09-20)

### Why we chunk at all
Two reasons, and the second is the one that is easy to miss. A whole document does not fit
in a model's input — that is the obvious one. But more importantly, **one embedding can
only represent one topic well.** Embed an 80-page manual and you get the average of eighty
pages, which is close to nothing in particular and matches no specific question sharply.

That creates a real tradeoff with a bad outcome at each end. Chunks too large: the chunk
holds the answer plus unrelated text, its embedding is diluted across topics, and it stops
matching precise questions. Chunks too small: the chunk loses the context that made it
meaningful, so "the third approach failed for the same reason" becomes useless.

### Why overlap exists
If a chunk boundary lands inside the sentence containing an answer, neither neighbouring
chunk holds the complete fact. Repeating the tail of each chunk at the start of the next
guarantees any short fact appears intact somewhere.

The implementation detail is the interesting part: we add overlap by moving a chunk's
*start offset backwards*, not by concatenating strings. That keeps every chunk one
contiguous slice of the document, so offsets remain meaningful. Concatenating text would
have broken the round-trip invariant immediately.

### Working in spans, not strings
The whole chunking module manipulates `(start, end)` integer pairs and only slices out
text at the very end. This is why `source.text[chunk.start_char:chunk.end_char] ==
chunk.text` is true *by construction* rather than by careful bookkeeping. Choosing a
representation that makes the invariant unavoidable is usually better than choosing one
that requires discipline to maintain.

### Frozen dataclasses
`@dataclass(frozen=True)` makes instances immutable — assigning to a field raises
`FrozenInstanceError`. Used here deliberately: once provenance is attached to a chunk,
later code cannot quietly change it. A bug becomes an exception instead of a wrong
citation.

### Unicode normalisation, which looks like a detail and is not
"é" can be a single code point or "e" followed by a combining accent. The two are visually
identical and compare as unequal in Python, so without normalising to NFC the same word
in two documents would produce two different embeddings and retrieval would quietly miss
matches. `unicodedata.normalize("NFC", text)` collapses them.

### Testing an invariant instead of an example
Most tests assert "this input gives that output." The most valuable test we wrote asserts
a *property* across many inputs: for every chunk of every document, the offsets slice back
to exactly the chunk text. If that ever fails, every citation VoiceLM produces is a lie,
so it is worth checking mechanically rather than trusting.

### Two bugs the tests caught, and what each taught
**`zip(chunks, chunks[1:], strict=True)` always raises.** `strict=True` demands equal
lengths, and `chunks[1:]` is by definition one shorter. The intent — walk consecutive
pairs — is spelled `itertools.pairwise(chunks)`. A case of reaching for a safety feature
in a place where it contradicts the goal.

**Chunks do not tile the document.** A test asserted that each chunk starts exactly where
the previous one ended; it failed by two characters. Those two characters are the `\n\n`
between paragraphs, which our paragraph splitter drops on purpose so chunk text never
begins or ends with a decorative blank line. The test encoded an assumption about the
implementation rather than a requirement of the product. The requirement is that no
*content* is lost, so the assertion became: whatever falls between two consecutive chunks
contains only whitespace. Worth noticing that the failing test was the wrong test — which
is not the same thing as the test being useless, since it forced the behaviour to be
decided consciously and written down (ADR-0014).

---

## Milestone 1, Step 2 — Embeddings and search (2026-09-20)

### What an embedding turned out to be, concretely
`nomic-embed-text` returns **768 floating point numbers** per piece of text. That list is a
position in a 768-dimensional space, arranged by training so that text with similar
meaning lands nearby. Nothing about it is human-readable; its only useful property is
relative distance to other vectors from the same model.

### Cosine similarity, and why length is ignored
Cosine similarity is the cosine of the angle between two vectors: 1.0 for the same
direction, 0.0 for perpendicular, -1.0 for opposite. The formula is the dot product
divided by both lengths, and dividing by the lengths is exactly what removes magnitude
from the comparison.

That property matters practically: a 900-character chunk and a 200-character chunk about
the same topic point the same way but have different magnitudes, and we want them to score
the same. The test `test_length_is_ignored` pins it down — `[1,2,3]` and `[10,20,30]` score
1.0.

A zero vector has no direction, so the angle to it is undefined. We raise rather than
return a made-up 0.0, because a silently wrong score is worse than a loud failure.

### Why production systems normalise vectors
If every vector is scaled to length 1 first, both divisions become divisions by 1, and
cosine similarity reduces to a plain dot product — far fewer operations per comparison.
We kept the explicit form because it is the definition and it is obviously correct;
Qdrant will do the fast version. Worth recognising the pattern: write the clear version,
let the specialised tool do the optimised one.

### Brute-force search is not much code
The entire vector store is a list plus a sort. Comparing a query against every stored
vector is *linear* — fine for hundreds of chunks, hopeless for millions, which is the gap
approximate nearest neighbour search exists to close. Having written the slow version, it
is clear what Qdrant will actually be buying us, and it stays as the reference we check
Qdrant against.

### Testing HTTP code without a network
`httpx2.MockTransport` intercepts requests and returns whatever response the test wants.
This makes it easy to test failures that are awkward to provoke for real: a 404, a
connection refusal, a response with one vector missing, a response with mismatched vector
lengths. The client's logic is covered offline and runs in milliseconds.

Separately, `test_embeddings_live.py` talks to the real Ollama and skips itself when
Ollama is not running. The division is deliberate: mocks verify *our* logic, and live tests
verify the thing no mock can — that the embeddings actually carry meaning.

### The test that proves the whole premise
`test_related_text_scores_higher_than_unrelated_text` embeds "How do I care for a young
cat?", "Kittens need feeding several times a day.", and a sentence about mortgage rates.
The related pair shares **no words** with the query while the unrelated pair shares one
("the"). Keyword search would rank these backwards. It passes against real Ollama, which
is the first concrete evidence that semantic search works rather than being a claim in a
diagram.

### Guarding against silent wrongness
Two checks exist purely to turn quiet corruption into loud errors. The embedder verifies it
got exactly as many vectors back as texts it sent, because a missing vector would shift
every subsequent pairing and make every citation point at the wrong text. The store refuses
vectors from a different model or of a different length, because vectors from different
models occupy unrelated spaces and comparing them produces confident nonsense with no
symptom. This is ADR-0008 as executable code rather than a note.

### Reading the log instead of trusting documentation
Ollama's startup log reported `library=Metal ... description="Apple M4"`, confirming GPU
acceleration, and `vram-based default context default_num_ctx=4096`. That last line
contradicted what had already been written in ADR-0016 (a fixed 2048 default), so the ADR
was corrected against the observed behaviour. Checking a claim against the running system
beats repeating it.

---

## Milestone 1, Step 3 — Grounded answers and citations (2026-09-20)

### Tokens and the context window as a shared budget
Models read *tokens*, roughly four characters of English each. The context window is a hard
ceiling on how many go in at once, and it is shared: instructions, retrieved excerpts,
conversation history, and the generated answer all draw from the same allowance. Exceed it
and text is dropped with no error.

We spend it deliberately. Excerpts get a 12,000-character budget (about 3,000 tokens),
which leaves the rest of an 8,192-token window for the instructions and the answer. When
the budget is exceeded, `select_within_budget` truncates the *ranking* rather than letting
Ollama truncate the *prompt*, because we would rather drop the least relevant excerpt on
purpose than have the server drop the beginning of the prompt silently.

### Quantization
`llama3.1:8b` has 8 billion parameters. At full 16-bit precision that would be about 16 GB
of weights; the file is 4.9 GB because each weight is compressed to roughly 4 bits. That
compression is the entire reason this runs on a laptop, and it costs a little accuracy.
Measured on this machine: 18.9 tokens per second, with an 11-second first load that is then
cached for five minutes.

### Temperature zero
Temperature controls randomness in token selection. We set it to 0 so the model reproduces
the excerpts faithfully rather than creatively, and so tests are deterministic. For a
grounded question-answering system, variety is not a feature.

### What actually makes an answer "grounded"
Three things working together, and none of them is the model being trustworthy:
1. The prompt contains only the retrieved excerpts and instructs the model to use nothing
   else.
2. The model is given a sanctioned way out — a specific refusal sentence — so it is not
   cornered into inventing an answer. Without that, refusal is not an available behaviour.
3. Citation details are read from *our* ingestion records, never from the model. The model
   supplies only a number. The source, title, and character offsets come from our data, so
   a citation cannot be hallucinated — the worst the model can do is point at the wrong
   excerpt, or at a number that does not exist.

### Detecting silent truncation
If Ollama's reported `prompt_eval_count` reaches `num_ctx`, the front of the prompt was
discarded and some excerpts never reached the model. The answer can look perfectly
reasonable while being ungrounded, so we raise an error instead. A guard for a failure with
no visible symptom is worth more than one for a failure that announces itself.

### The bug the live test found
Given a prompt with a single excerpt `[1]`, llama3.1:8b answered `"Tuesdays [2]."` — citing
a number that did not exist. Our validation correctly refused to invent a citation for
`[2]`, so the test failed with `citations = ()` while the text still displayed `[2]`.

That is a real product bug and no amount of reasoning about the code would have found it;
it needed a real model. The fix (ADR-0018) removes markers that match no excerpt, records
them on `Answer.unsupported_markers`, and establishes the invariant that the markers shown
in an answer and the citations listed beneath it always agree.

Worth sitting with the shape of this. The unit tests all passed. The integration tests all
passed. The thing that broke was the model not following an instruction, which is a class
of bug that only exists in systems with a model in them, and the only way to find it is to
run the model.

### Two test failures, two different lessons
The first failure was the bug above. The second was my own test fixture: the document was
small enough that the default 1000-character chunking collapsed it into one chunk, so the
citation tests were exercising a single-excerpt case that barely occurs in practice. Fixed
by chunking the fixture smaller, and by adding a test that asserts the fixture produces
several chunks — a guard on the test setup itself, so it cannot silently degrade into
checking nothing.

### Seeing the chunk-size tradeoff for real
Running the CLI on an 864-character document with default settings produced one chunk, and
therefore a citation reading `characters 0-864` — technically correct and useless, since it
points at the entire document. Re-running with `--chunk-chars 250` produced five chunks and
a citation of `characters 135-360`, which points at the actual passage. The abstract
tradeoff from Step 1 became visible in output: chunk size is the main lever on how precise
a citation can be.

### Keeping logic out of the entry point
`cli.py` parses arguments, calls functions, and prints. That is all. The same functions will
sit behind HTTP endpoints later with no rewrite, which only works because none of the logic
lives in the entry point. Progress goes to stderr and the answer to stdout, so the answer
can be piped somewhere without the progress lines coming along.

### A macOS debugging lesson: hardlinks and wedged binaries
Killing processes mid-`uv sync` left four `ruff` processes in state `UE` — uninterruptible
and unkillable, not even by `kill -9`. Every subsequent `ruff` invocation hung too, and so
did `pytest`, while `ls` and `df` stayed instant, which ruled out a general filesystem
problem.

The cause: uv saves space by *hardlinking* venv binaries to a single copy in its global
cache, so every project shares one inode. Once a process wedged that inode, every `exec` of
that file blocked. Deleting `.venv` alone would not have helped, because a fresh sync would
hardlink to the same wedged inode. `UV_LINK_MODE=copy uv sync` forced real copies with new
inodes, confirmed by `stat` showing `links=1`, and everything worked immediately.

The transferable part: when a binary hangs rather than failing, suspect the file itself
rather than the program, and check whether something else shares its inode.

---

## Milestone 1B, Step 1 — SQLite catalog (2026-09-20)

### SQLite is a library
There is no process to start. `sqlite3.connect(path)` opens (or creates) one file. The
standard library ships the driver; the only dependency is the file on disk. That is why
the architecture diagram has a network arrow to Ollama and Qdrant, and not to SQLite.

### Foreign keys are opt-in
SQLite parses `REFERENCES` and `ON DELETE CASCADE` in the schema, then ignores them until
you run `PRAGMA foreign_keys = ON` on **that connection**. Forgetting it means deleting a
source leaves orphan chunks with no error. Tests that assert cascade would fail if we
omitted it — which is why they exist.

### WAL mode
`PRAGMA journal_mode = WAL` writes changes to a `-wal` sidecar instead of overwriting the
main file. Readers can search while a writer is ingesting. Default rollback-journal mode
locks the whole file for the duration of a write.

### Identity vs content
A hash of the text answers "has this changed?" A UUID plus a unique path answers "which
document is this?" Mixing them (the Milestone 1 shortcut) made identical files collapse
into one source. Fine for a one-shot CLI; wrong for a workspace.

`load_source` still cannot look up an existing id: it does not know a database exists. It
mints a UUID every time. The knowledge base, next, will swap in the stored id when the
path is already known.

### Transactions
`with connection:` in Python's sqlite3 module begins a transaction and commits on success
or rolls back on exception. Saving a source and replacing its chunks in that block means
you cannot observe a source with yesterday's chunks.

### Order is part of `get_chunks_by_ids`
SQL `IN (...)` does not preserve the order of the id list. Search ranks by similarity, so
the catalog has to re-order the rows to match what the caller asked for. Missing ids raise:
that is Qdrant and SQLite having drifted, and silence would drop a hit.

---

## Milestone 1B, Step 2 — Qdrant local mode (2026-09-21)

### What Qdrant is, at this scale
A specialist for "here is a vector, which stored vectors are nearest?". Our brute-force
store does that with a Python loop. Qdrant does it with a graph index called HNSW
(Hierarchical Navigable Small World): it does not compare against every vector. It jumps
across a network of "nearby" points and returns an *approximate* nearest neighbour.

Approximate means it might, in theory, miss the true closest vector. At a handful of
chunks it agrees exactly with brute force — we asserted that, and that is why
`InMemoryVectorStore` was not deleted.

### Local mode
`QdrantClient(path="...")` runs the engine inside our Python process against a directory.
No Docker, no port 6333. Closing and reopening the same path still finds the vectors,
which is the whole difference from the in-memory list.

### Why the search hit has no text
`VectorHit` carries `chunk_id`, `source_id`, and a score. Not the chunk text. If Qdrant
also stored the text we would have two copies that can drift. SQLite remains the only
place a citation can quote from. Step 3 is the join: Qdrant returns ids, SQLite loads
those rows *in that order*.

### Point ids
Qdrant only accepts UUIDs or integers as point ids. Our chunk ids are strings
(`{source_uuid}:0`), so we derive a UUID with `uuid5` from the chunk id. The real id lives
in the payload. Deterministic means re-upserting the same chunk overwrites the same point
instead of creating a duplicate.

---

## Milestone 1B, Step 3 — KnowledgeBase (2026-09-21)

### The join
`search` is the sentence the last exercise was pointing at: Qdrant returns `VectorHit`
ids, SQLite returns chunks in that same order, and we wrap them as `SearchResult` so
`answer_question` does not care where the passages came from.

### Skip vs repair
"File unchanged" is not enough to skip. If Qdrant died after SQLite committed, the hash
still matches and a naive skip would never write vectors again. Completeness is
`vector_count == chunk_count` for that source.

### Dual-write without a shared transaction
Two libraries cannot commit together. Embed first (no writes), SQLite next, Qdrant last.
A new file that fails in Qdrant is deleted from SQLite so we do not list a document you
cannot search. An *update* that fails in Qdrant keeps the new text and drops old vectors;
the next ingest sees an incomplete index and repairs.

### Protocol, not a class hierarchy
`Embedder` is a typing `Protocol`: "has `embed_chunks` and `embed_query`." Tests pass a
fake; production passes `OllamaEmbedder`. No inheritance, no extra files.

---

## Milestone 1B, Step 4 — the CLI over the library (2026-09-21)

### Subcommands mark a change in lifetime
`ask --source FILE` treated a document as something that exists for one command. Once
ingestion is durable, importing and asking have genuinely different lifetimes, so they
became different verbs: `ingest` writes, `ask` reads, `sources` inspects. `ask --source`
survives as a convenience that ingests first, not as the only way in.

### The CLI is not allowed to know about storage
`cli.py` imports `KnowledgeBase` and never `SqliteMetadataStore` or `QdrantVectorIndex`.
It parses arguments, calls one object, and prints. That is the same discipline the API
routes will follow, which is why the next milestone is mostly wiring rather than redesign.

### stdout is the answer, stderr is the commentary
Progress lines, timings, and the unsupported-marker warning go to stderr; only the answer
and its citations go to stdout. `voicelm ask ... > answer.txt` therefore captures the
answer alone. Deciding which stream a line belongs on is a small API design decision.

### Injecting dependencies is what made this testable
`main(argv, knowledge_base=..., chat_model=...)` lets the tests pass a real
`KnowledgeBase` over a tmp directory — real SQLite, real Qdrant — with only Ollama faked.
Without those parameters the only way to test the CLI would be to spawn a subprocess and
match strings, which is slower and tells you less about why something broke.

### Closing what you opened
`main` opens a `KnowledgeBase` only when the caller did not supply one, tracks that with
`owned`, and closes it in a `finally`. A test that passed its own base would otherwise
find it closed underneath it. "Whoever opens it closes it" is worth being explicit about
once a program holds file handles.

---

## Milestone 1C — PDF ingestion and page citations (2026-09-21)

### A PDF does not contain text
It contains drawing instructions: "put glyph T at (72, 700)". Extraction reconstructs
a string by heuristic. That is why two libraries disagree, why columns can come out
interleaved, and why a scanned PDF — an image of paper — yields the empty string.

### Why we refused empty rather than ingesting it
A document with no text would list in `sources`, match nothing, and look like search is
broken. `NoTextLayer` makes the missing capability (OCR) the error, instead of a silent
empty library entry.

### The page map is a lookup, not a chunking rule
`Source.text` stays one string so chunking does not learn about PDFs. `PageSpan` records
where each page landed. After retrieval we ask "which spans overlap this chunk?" That is
how a character offset becomes "page 7", and how a chunk that crosses a break becomes
"pages 4–5" instead of a lie.

### Ligatures are a PDF problem
IEEE papers emit `ﬁ` as one character. A user searching for "identification" would miss
it. NFKC on PDF pages only turns that into `fi` without rewriting Markdown.

### Test PDFs are generated, not committed
A ~100-line helper writes a valid PDF-1.4 file (objects, xref table, trailer) so tests
do not need `reportlab` or a binary fixture. That helper is also a look at why extraction
is reconstruction: the only line break is a `T*` operator moving the cursor down.


