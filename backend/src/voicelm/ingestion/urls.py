"""Shared URL normalization and browser-like fetch headers.

Kept free of trafilatura/justext so YouTube ingest can import this without pulling the
web-extraction stack (and its data files) into every URL request.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse, urlunparse

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


class InvalidUrl(Exception):
    """The string is not an http(s) URL we are willing to fetch."""


def normalize_url(raw: str) -> str:
    """Require http(s), strip fragments, refuse private/literal hosts (SSRF posture)."""
    candidate = raw.strip()
    if not candidate:
        raise InvalidUrl("url is empty")

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise InvalidUrl("url must start with http:// or https://")
    if not parsed.netloc:
        raise InvalidUrl("url is missing a host")

    host = parsed.hostname or ""
    if host.lower() in {"localhost", "127.0.0.1", "::1"} or host.endswith(".localhost"):
        raise InvalidUrl("refusing to fetch localhost")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private or address.is_loopback or address.is_link_local or address.is_reserved
    ):
        raise InvalidUrl("refusing to fetch a private or reserved address")

    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.params,
            parsed.query,
            "",
        )
    )
