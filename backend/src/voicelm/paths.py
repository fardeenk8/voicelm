"""Where VoiceLM keeps its library on disk.

CLI and API both import this so a flag, an env var, and a default cannot drift.
"""

import os
from pathlib import Path

FILES_DIR_NAME = "files"


def default_data_dir() -> Path:
    """Override with VOICELM_DATA_DIR. Default is ./data in the current working directory."""
    raw = os.environ.get("VOICELM_DATA_DIR")
    return Path(raw) if raw else Path.cwd() / "data"


def owned_files_dir(data_dir: Path) -> Path:
    """Copies of files the API ingested. The CLI still indexes files in place."""
    return data_dir / FILES_DIR_NAME
