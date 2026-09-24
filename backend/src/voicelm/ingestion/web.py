"""Fetch a public web page and turn it into cleaned text.

The backend does the GET itself (local-first: your machine talks to the site). We keep
the main article body and drop chrome. Citations point at the URL, not a fake page number.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx2
import trafilatura

from voicelm.ingestion.cleaning import clean_text
from voicelm.ingestion.urls import DEFAULT_HEADERS
from voicelm.ingestion.urls import InvalidUrl as UrlInvalid
from voicelm.ingestion.urls import normalize_url as canonicalize_url

MAX_RESPONSE_BYTES = 5 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 30.0


class WebFetchError(Exception):
    """Base for every reason a URL did not become a usable source."""


class InvalidUrl(WebFetchError):
    """The string is not an http(s) URL we are willing to fetch."""


class UnreachableUrl(WebFetchError):
    """DNS, TLS, timeout, or connection failure."""


class HttpFetchError(WebFetchError):
    """The server answered, but not with a usable page."""


class EmptyWebPage(WebFetchError):
    """Fetched fine, but trafilatura found no main text."""


@dataclass(frozen=True)
class WebPage:
    url: str
    title: str
    text: str


def normalize_url(raw: str) -> str:
    """Require http(s), strip fragments, refuse private/literal hosts (SSRF posture)."""
    try:
        return canonicalize_url(raw)
    except UrlInvalid as error:
        raise InvalidUrl(str(error)) from error


def url_storage_name(url: str) -> str:
    """Stable filename for the owned copy under data/files/web/."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.md"


def fetch_web_page(
    url: str,
    *,
    client: httpx2.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> WebPage:
    """GET `url`, extract the main text, return title + cleaned body."""
    canonical = normalize_url(url)
    owned = client is None
    http = client or httpx2.Client(
        timeout=timeout,
        follow_redirects=True,
        headers=DEFAULT_HEADERS,
    )
    try:
        try:
            response = http.get(canonical)
        except httpx2.TimeoutException as error:
            raise UnreachableUrl(f"timed out fetching {canonical}") from error
        except httpx2.RequestError as error:
            raise UnreachableUrl(f"could not reach {canonical}: {error}") from error

        if response.status_code >= 400:
            raise HttpFetchError(_http_error_message(canonical, response.status_code))

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower() and not _looks_like_html(response.text):
            raise HttpFetchError(
                f"{canonical} is not an HTML page ({content_type or 'no content-type'})"
            )

        raw = response.content
        if len(raw) > MAX_RESPONSE_BYTES:
            raise HttpFetchError(
                f"{canonical} is larger than {MAX_RESPONSE_BYTES // (1024 * 1024)} MB"
            )

        html = raw.decode(response.encoding or "utf-8", errors="replace")
        extracted = trafilatura.extract(
            html,
            url=canonical,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
        if not extracted or not extracted.strip():
            raise EmptyWebPage(f"{canonical} has no extractable article text")

        metadata = trafilatura.extract_metadata(html, default_url=canonical)
        title = (metadata.title if metadata and metadata.title else "") or _title_from_url(
            canonical
        )
        text = clean_text(extracted)
        if not text:
            raise EmptyWebPage(f"{canonical} has no extractable article text")

        return WebPage(url=canonical, title=_clean_title(title), text=text)
    finally:
        if owned:
            http.close()


def _http_error_message(url: str, status: int) -> str:
    if status in {401, 403}:
        return (
            f"{url} returned HTTP {status} — the site blocked automated fetch. "
            "Try saving the page as PDF or HTML and add it with +, or pick another URL."
        )
    return f"{url} returned HTTP {status}"


def _looks_like_html(sample: str) -> bool:
    head = sample[:2000].lower()
    return "<html" in head or "<body" in head or "<article" in head


def _title_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    leaf = path.split("/")[-1] if path else urlparse(url).netloc
    leaf = re.sub(r"[-_]+", " ", leaf)
    return leaf or urlparse(url).netloc or url


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip() or "untitled"
