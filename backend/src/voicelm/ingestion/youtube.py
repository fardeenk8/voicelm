"""Fetch a YouTube video's transcript and turn it into timed text.

Prefer on-platform captions (no download). If captions are missing or blocked, fall back
to downloading audio with yt-dlp and running the local Whisper transcriber — still on
this machine, never a cloud STT API.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import httpx2

from voicelm.domain.models import PageSpan
from voicelm.ingestion.transcript import TimedSegment, assemble_timed_segments
from voicelm.ingestion.urls import DEFAULT_HEADERS, InvalidUrl, normalize_url

_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }
)
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeError(Exception):
    """Base for every reason a YouTube URL did not become a usable source."""


class NotYouTubeUrl(YouTubeError):
    """The URL is not a YouTube watch / share link."""


class YouTubeTranscriptUnavailable(YouTubeError):
    """No captions and (if tried) speech-to-text also failed."""


@dataclass(frozen=True)
class YouTubeVideo:
    url: str
    video_id: str
    title: str
    text: str
    pages: tuple[PageSpan, ...]


def is_youtube_url(url: str) -> bool:
    """True when `url` looks like a YouTube video link (before full normalize)."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        return False
    try:
        extract_video_id(url)
    except (NotYouTubeUrl, InvalidUrl):
        return False
    return True


def extract_video_id(url: str) -> str:
    """Pull the 11-character video id out of common YouTube URL shapes."""
    canonical = normalize_url(url)
    parsed = urlparse(canonical)
    host = (parsed.hostname or "").lower()

    if host in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID.match(candidate):
            return candidate
        raise NotYouTubeUrl(f"{url} is not a YouTube video link")

    if host not in _YOUTUBE_HOSTS:
        raise NotYouTubeUrl(f"{url} is not a YouTube video link")

    query = parse_qs(parsed.query)
    if "v" in query and query["v"] and _VIDEO_ID.match(query["v"][0]):
        return query["v"][0]

    parts = [part for part in parsed.path.split("/") if part]
    if (
        len(parts) >= 2
        and parts[0] in {"embed", "shorts", "live", "v"}
        and _VIDEO_ID.match(parts[1])
    ):
        return parts[1]

    raise NotYouTubeUrl(f"{url} is not a YouTube video link")


def youtube_canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def youtube_storage_name(video_id: str) -> str:
    """Stable filename under data/files/youtube/. `.ytt` marks timestamp citations."""
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.ytt"


def fetch_youtube_video(
    url: str,
    *,
    transcriber: object | None = None,
    client: httpx2.Client | None = None,
) -> YouTubeVideo:
    """Resolve captions (or STT fallback) into cleaned text + second-level spans."""
    video_id = extract_video_id(url)
    canonical = youtube_canonical_url(video_id)
    title = _fetch_title(canonical, client=client) or f"YouTube {video_id}"

    segments = _fetch_caption_segments(video_id)
    if not segments and transcriber is not None:
        segments = _transcribe_via_download(video_id, transcriber)

    if not segments:
        raise YouTubeTranscriptUnavailable(
            f"{canonical} has no captions VoiceLM could read, and no local "
            "speech-to-text fallback produced text. Try a video with captions, or "
            "download the audio and add the file with +."
        )

    text, pages = assemble_timed_segments(segments)
    if not text:
        raise YouTubeTranscriptUnavailable(f"{canonical} transcript was empty after cleaning")

    return YouTubeVideo(
        url=canonical,
        video_id=video_id,
        title=title,
        text=text,
        pages=pages,
    )


def _fetch_caption_segments(video_id: str) -> list[TimedSegment]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import YouTubeTranscriptApiException
    except ImportError as error:
        raise YouTubeTranscriptUnavailable("youtube-transcript-api is not installed") from error

    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=("en", "en-US", "en-GB"))
    except YouTubeTranscriptApiException:
        try:
            listing = api.list(video_id)
            transcript = next(iter(listing))
            fetched = transcript.fetch()
        except Exception:
            return []
    except Exception:
        return []

    segments: list[TimedSegment] = []
    for item in fetched:
        # FetchedTranscriptSnippet uses attributes; raw dicts use keys.
        start = float(getattr(item, "start", None) if hasattr(item, "start") else item["start"])
        text = getattr(item, "text", None) if hasattr(item, "text") else item["text"]
        segments.append(TimedSegment(start=start, text=str(text)))
    return segments


def _fetch_title(url: str, *, client: httpx2.Client | None = None) -> str | None:
    owned = client is None
    http = client or httpx2.Client(timeout=20.0, headers=DEFAULT_HEADERS, follow_redirects=True)
    try:
        response = http.get("https://www.youtube.com/oembed", params={"url": url, "format": "json"})
        if response.status_code >= 400:
            return None
        data = response.json()
        title = data.get("title")
        return str(title).strip() if title else None
    except Exception:
        return None
    finally:
        if owned:
            http.close()


def _transcribe_via_download(video_id: str, transcriber: object) -> list[TimedSegment]:
    """Download audio with yt-dlp into a temp dir, then run the injected transcriber."""
    import tempfile
    from pathlib import Path

    try:
        import yt_dlp
    except ImportError:
        return []

    if not hasattr(transcriber, "transcribe"):
        return []

    with tempfile.TemporaryDirectory(prefix="voicelm-yt-") as tmp:
        outtmpl = str(Path(tmp) / "audio.%(ext)s")
        opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([youtube_canonical_url(video_id)])
        except Exception:
            return []

        audio_files = list(Path(tmp).glob("audio.*"))
        if not audio_files:
            return []
        result = transcriber.transcribe(audio_files[0])  # type: ignore[attr-defined]
        return list(result.segments)
