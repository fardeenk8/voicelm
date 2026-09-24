# VoiceLM

A privacy-first, voice-first personal knowledge workspace. Import your documents, ask
questions, get answers grounded in your own sources with citations back to the exact page.

Everything runs locally. Your documents never leave your machine.

## Status

**Milestone 2E — GitHub repositories.** Paste a `github.com` URL (repo, `/tree/…`, or
`/blob/…`) via CLI `--url`, `POST /sources/url`, or the desktop Add link / + menu. The
backend downloads a zipball, indexes text/code files as separate sources, and cites
**path + line** with a blob URL. Private repos need `VOICELM_GITHUB_TOKEN` or
`GITHUB_TOKEN`. Phase 2 multimodal ingest is complete; voice conversation is Phase 3.

See [`docs/PRODUCT_SPEC.md`](docs/PRODUCT_SPEC.md) for where this is going and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how it is put together.

## Repository layout

```
docs/        Product spec, architecture, decision record, learning log
backend/     Python AI service (FastAPI) — the brain. Runs standalone.
desktop/     Flutter macOS app — presentation only. Calls the HTTP API.
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
```

Commands that import `voicelm` need to be told where the source lives, either with
`PYTHONPATH=src` or uvicorn's `--app-dir src`. That is required rather than optional; see
ADR-0012 for why we do not rely on the editable install's import path.

## Building a library and asking questions

Requires Ollama running with two models pulled:

```bash
brew install ollama
brew services start ollama
ollama pull nomic-embed-text    # embeddings, 274 MB
ollama pull llama3.1:8b         # generation, 4.9 GB
```

Import documents once. From `backend/`:

```bash
PYTHONPATH=src ./.venv/bin/voicelm ingest \
  --source ../docs/ARCHITECTURE.md \
  --source paper.pdf
```

```
  ARCHITECTURE.md: ingested, 11 chunks (1.0s)
  DECISIONS.md: ingested, 25 chunks (1.0s)
```

Then ask, as many times as you like and in as many separate runs as you like:

```bash
PYTHONPATH=src ./.venv/bin/voicelm ask \
  "Why did we choose Qdrant local mode instead of running a Qdrant server?"
```

```
  searching...

Identical Python API, so the migration is a configuration change rather than a
rewrite. It removes an entire category of "is the container running?" debugging. [1]

Sources:
  [1] DECISIONS, characters 4985-6116
      "...keeping relational metadata in a vector database, which is not its purpose..."
  [2] paper, page 7, characters 18420-19110
      "...CataractNet is a lightweight CNN designed for fundus images..."
```

`voicelm sources` lists what is in the library (including each document's id).
`voicelm remove ID` drops a document. Re-running `ingest` on an unchanged file is
reported as `skipped` and costs nothing; edit the file and it is re-embedded in place,
keeping the same identity.

Useful options: `--top-k` sets how many excerpts are retrieved; `--chunk-chars` and
`--chunk-overlap` control chunking, which is the main lever on how precise a citation is;
`--data-dir` (or `$VOICELM_DATA_DIR`) chooses where the library is stored, defaulting to
`./data`.

If the documents do not contain the answer, VoiceLM says so rather than inventing one.

## The HTTP API

The same library the CLI uses, over JSON. Start it from `backend/`:

```bash
uv run uvicorn --app-dir src voicelm.api.app:app --reload
```

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","version":"0.1.0"}

curl -F file=@notes.md http://127.0.0.1:8000/sources
curl http://127.0.0.1:8000/sources
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Why did we choose that approach?"}'

curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "Why did we choose that approach?"}'
```

Uploaded files are stored under `data/files/` so the library owns a copy. Interactive
documentation is at <http://127.0.0.1:8000/docs>. `/ask` returns the complete answer.
`/ask/stream` sends `token` events, then a `done` event with the same JSON (citations
included).

## The desktop app

Requires the backend already running. From the repo root:

```bash
cd desktop
flutter run -d macos
```

The window lists the library, uploads a file, and asks a question. It never opens
SQLite, Qdrant, or Ollama — only `http://127.0.0.1:8000`. Override the URL with
`--dart-define=VOICELM_API=http://127.0.0.1:8000` if needed.

```bash
cd desktop
flutter test
flutter analyze
```
