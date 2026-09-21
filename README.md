# VoiceLM

A privacy-first, voice-first personal knowledge workspace. Import your documents, ask
questions, get answers grounded in your own sources with citations back to the exact page.

Everything runs locally. Your documents never leave your machine.

## Status

**Milestone 1 — the retrieval loop works.** Import a `.txt` or `.md` file and ask a
question about it from the command line; answers come back grounded in the document with
citations to exact character ranges. Everything runs locally against Ollama.

Not yet: PDFs, persistent storage, an HTTP API for this, or any UI.

See [`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md) for where this is going and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how it is put together.

## Repository layout

```
docs/        Product spec, architecture, decision record, learning log
backend/     Python AI service (FastAPI) — the brain. Runs standalone.
desktop/     Flutter macOS app — added in a later milestone.
```

The backend is deliberately independent of the UI: it must remain fully runnable and
testable with `desktop/` deleted. That property is what lets a future mobile or web
client reuse the same AI services.

## Running the backend

Requires [uv](https://docs.astral.sh/uv/) (dependency manager) and Python 3.13.

```bash
cd backend
uv sync                 # create the virtual environment and install dependencies
uv run pytest           # run the tests
uv run ruff check .     # lint
uv run uvicorn --app-dir src voicelm.api.app:app --reload
```

`--app-dir src` is required rather than optional; see ADR-0012 for why we do not rely on
the editable install's import path.

## Asking a question

Requires Ollama running with two models pulled:

```bash
brew install ollama
brew services start ollama
ollama pull nomic-embed-text    # embeddings, 274 MB
ollama pull llama3.1:8b         # generation, 4.9 GB
```

Then, from `backend/`:

```bash
PYTHONPATH=src ./.venv/bin/voicelm ask \
  --source notes.md \
  "Why did we choose that approach?"
```

```
  notes.md: 5 chunks embedded in 0.1s
  searching 5 chunks...

Re-embedding unchanged documents was consuming roughly 80% of ingestion time. [1]

Sources:
  [1] notes, characters 135-360
      "...hashing by content means renaming a file does not invalidate its cache entry..."
```

`PYTHONPATH=src` is needed for the same reason as `--app-dir src` above (ADR-0012).

Useful options: `--source` may be repeated to search several documents; `--top-k` sets how
many excerpts are retrieved; `--chunk-chars` and `--chunk-overlap` control chunking, which
is the main lever on how precise a citation is.

If the documents do not contain the answer, VoiceLM says so rather than inventing one.

Then check that it is alive:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","version":"0.1.0"}
```

Interactive API documentation is generated automatically at
<http://127.0.0.1:8000/docs>.
