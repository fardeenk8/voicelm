"""HTTP entry point for the VoiceLM backend."""

from fastapi import FastAPI

from voicelm import __version__


def create_app() -> FastAPI:
    """Build a fresh application instance.

    A factory rather than a module-level app so each test gets a clean instance and
    cannot be affected by state another test left behind.
    """
    app = FastAPI(title="VoiceLM API", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


# The instance uvicorn serves: `uvicorn voicelm.api.app:app`
app = create_app()
