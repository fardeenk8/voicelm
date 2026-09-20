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

Deleting `.venv` entirely and re-running `uv sync` fixed it permanently. Honest limitation:
we never fully explained *why* `site` skipped that particular file, only that a clean
rebuild resolves it reproducibly. The lesson for next time is that when a virtual
environment misbehaves in a way that contradicts its own configuration, recreating it is
cheap and should be tried early. Environments are disposable; that is the whole point of
having a lockfile.

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

## Queued for Milestone 1

To be filled in as we cover them: embeddings, cosine similarity, chunking strategy,
approximate nearest neighbor search (HNSW), tokens and context windows, quantization, and
prompt construction for grounded answers.
