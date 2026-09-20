# VoiceLM

A privacy-first, voice-first personal knowledge workspace. Import your documents, ask
questions, get answers grounded in your own sources with citations back to the exact page.

Everything runs locally. Your documents never leave your machine.

## Status

**Milestone 0 — project skeleton.** The repository, documentation, and a Python backend
with a health check and a passing test. No AI functionality yet.

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
uv run uvicorn voicelm.api.app:app --reload
```

Then check that it is alive:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","version":"0.1.0"}
```

Interactive API documentation is generated automatically at
<http://127.0.0.1:8000/docs>.
