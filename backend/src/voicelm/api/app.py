"""HTTP entry point for the VoiceLM backend.

Routes parse JSON (or a file), call KnowledgeBase, and serialise the result. They do not
open SQLite or Qdrant themselves — that rule is what lets the CLI and a future Flutter
client share one implementation (ADR-0026).
"""

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from voicelm import __version__
from voicelm.api.schemas import AskIn, AskOut, IngestOut, SourceOut, ask_out, ingest_out, source_out
from voicelm.domain.models import Answer
from voicelm.embeddings.ollama import EmbeddingError, OllamaEmbedder
from voicelm.generation.answering import AnswerToken
from voicelm.generation.ollama import ChatModel, GenerationError, OllamaChatModel
from voicelm.ingestion.loader import UndecodableFile, UnsupportedFileType
from voicelm.ingestion.pdf import PdfExtractionError
from voicelm.knowledge import KnowledgeBase
from voicelm.paths import default_data_dir

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

ClientError = (
    UnsupportedFileType,
    UndecodableFile,
    PdfExtractionError,
    FileNotFoundError,
    ValueError,
)


def create_app(
    *,
    knowledge_base: KnowledgeBase | None = None,
    chat_model: ChatModel | None = None,
    data_dir: Path | None = None,
) -> FastAPI:
    """Build a fresh application instance.

    A factory rather than a module-level app so each test gets a clean instance and
    cannot be affected by state another test left behind. Tests pass `knowledge_base`
    and `chat_model`; production opens them during lifespan.
    """
    owned = knowledge_base is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.knowledge_base = knowledge_base or KnowledgeBase(
            data_dir or default_data_dir(),
            OllamaEmbedder(),
        )
        app.state.chat_model = chat_model
        try:
            yield
        finally:
            if owned:
                app.state.knowledge_base.close()

    app = FastAPI(title="VoiceLM API", version=__version__, lifespan=lifespan)

    async def client_error(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=400)

    for error_type in ClientError:
        app.add_exception_handler(error_type, client_error)

    async def ollama_error(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=503)

    app.add_exception_handler(EmbeddingError, ollama_error)
    app.add_exception_handler(GenerationError, ollama_error)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/sources")
    def list_sources(base: Annotated[KnowledgeBase, Depends(_get_base)]) -> list[SourceOut]:
        return [
            source_out(record, base.chunk_count(record.source.id)) for record in base.list_sources()
        ]

    @app.post("/sources")
    def ingest_source(
        base: Annotated[KnowledgeBase, Depends(_get_base)],
        file: Annotated[UploadFile, File()],
    ) -> IngestOut:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file is larger than 50 MB")
        name = file.filename or ""
        result = base.ingest_upload(name, data)
        return ingest_out(result)

    @app.delete("/sources/{source_id}", status_code=204)
    def remove_source(
        source_id: str,
        base: Annotated[KnowledgeBase, Depends(_get_base)],
    ) -> None:
        if not base.remove(source_id):
            raise HTTPException(status_code=404, detail=f"no source {source_id}")

    @app.post("/ask")
    def ask(
        body: AskIn,
        base: Annotated[KnowledgeBase, Depends(_get_base)],
        model: Annotated[ChatModel, Depends(_get_chat)],
    ) -> AskOut:
        if not base.list_sources():
            raise HTTPException(
                status_code=409,
                detail="library is empty; POST a file to /sources first",
            )
        return ask_out(base.ask(body.question, model, top_k=body.top_k))

    @app.post("/ask/stream")
    def ask_stream(
        body: AskIn,
        base: Annotated[KnowledgeBase, Depends(_get_base)],
        model: Annotated[ChatModel, Depends(_get_chat)],
    ) -> StreamingResponse:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="question cannot be empty")
        if not base.list_sources():
            raise HTTPException(
                status_code=409,
                detail="library is empty; POST a file to /sources first",
            )
        return StreamingResponse(
            _ask_events(base, model, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _ask_events(base: KnowledgeBase, model: ChatModel, body: AskIn) -> Iterator[str]:
    """Turn KnowledgeBase.ask_stream into SSE frames.

    Errors after the stream has opened cannot become HTTP status codes, so they
    become an `error` event. Empty-library and blank-question stay real statuses
    because those are checked before this generator runs.
    """
    try:
        for item in base.ask_stream(body.question, model, top_k=body.top_k):
            if isinstance(item, AnswerToken):
                yield _sse("token", {"text": item.text})
            elif isinstance(item, Answer):
                yield _sse("done", ask_out(item).model_dump())
    except (GenerationError, EmbeddingError, *ClientError) as error:
        yield _sse("error", {"detail": str(error)})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _get_base(request: Request) -> KnowledgeBase:
    return request.app.state.knowledge_base


def _get_chat(request: Request) -> ChatModel:
    model = request.app.state.chat_model
    if model is None:
        model = OllamaChatModel()
        request.app.state.chat_model = model
    return model


# The instance uvicorn serves: `uvicorn voicelm.api.app:app`
app = create_app()
