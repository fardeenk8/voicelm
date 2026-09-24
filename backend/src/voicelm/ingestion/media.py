"""Local speech-to-text for audio and video files.

Whisper runs on this machine (faster-whisper). No audio leaves the library. The same
timed-segment assembly YouTube captions use produces `PageSpan`s whose numbers are
seconds from the start of the file.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from voicelm.domain.models import PageSpan
from voicelm.ingestion.transcript import TimedSegment, assemble_timed_segments

AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".mkv", ".mpeg", ".mpg", ".m4v"})
MEDIA_SUFFIXES = AUDIO_SUFFIXES | VIDEO_SUFFIXES

DEFAULT_WHISPER_MODEL = os.environ.get("VOICELM_WHISPER_MODEL", "base")


class MediaTranscriptionError(Exception):
    """Base for every reason a media file yielded no usable transcript."""


class CorruptMediaFile(MediaTranscriptionError):
    """The file could not be opened or decoded."""


class EmptyTranscript(MediaTranscriptionError):
    """Whisper ran, but produced no speech text."""


@dataclass(frozen=True)
class TranscriptResult:
    segments: tuple[TimedSegment, ...]
    language: str | None = None


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> TranscriptResult: ...


class FasterWhisperTranscriber:
    """Lazy wrapper around faster-whisper so importing the package is optional until use."""

    def __init__(self, model_size: str = DEFAULT_WHISPER_MODEL) -> None:
        self._model_size = model_size
        self._model = None

    def transcribe(self, path: Path) -> TranscriptResult:
        model = self._load()
        try:
            segments_iter, info = model.transcribe(str(path), vad_filter=True)
        except Exception as error:
            raise CorruptMediaFile(f"{path.name} could not be transcribed: {error}") from error

        segments = tuple(
            TimedSegment(start=float(segment.start), text=segment.text)
            for segment in segments_iter
            if segment.text and segment.text.strip()
        )
        language = getattr(info, "language", None)
        return TranscriptResult(segments=segments, language=language)

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise MediaTranscriptionError(
                "faster-whisper is not installed; cannot transcribe audio/video"
            ) from error
        # int8 on CPU is the practical default for a laptop library.
        self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        return self._model


_default_transcriber: FasterWhisperTranscriber | None = None


def default_transcriber() -> FasterWhisperTranscriber:
    global _default_transcriber
    if _default_transcriber is None:
        _default_transcriber = FasterWhisperTranscriber()
    return _default_transcriber


def hash_media_file(path: Path) -> str:
    """SHA-256 of file bytes — used to skip re-running Whisper on unchanged media."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_media(
    path: Path, *, transcriber: Transcriber | None = None
) -> tuple[str, tuple[PageSpan, ...], str]:
    """Transcribe `path` into cleaned text, timestamp spans, and a content hash.

    The content hash is of the *file bytes*, not the transcript: Whisper is expensive, and
    an unchanged recording should not be re-transcribed on every ingest (ADR-0032).
    """
    suffix = path.suffix.lower()
    if suffix not in MEDIA_SUFFIXES:
        raise CorruptMediaFile(f"{path.name} is not a supported audio/video type")

    content_hash = hash_media_file(path)
    engine = transcriber or default_transcriber()
    result = engine.transcribe(path)
    if not result.segments:
        raise EmptyTranscript(f"{path.name} produced no speech text")

    text, pages = assemble_timed_segments(result.segments)
    if not text:
        raise EmptyTranscript(f"{path.name} produced no speech text")
    return text, pages, content_hash
