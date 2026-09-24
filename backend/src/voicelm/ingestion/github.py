"""Fetch a GitHub repository snapshot and turn text files into line-cited sources.

Prefer the zipball API (one HTTP get, no `git` binary) over clone or the Contents API.
Private repos use `VOICELM_GITHUB_TOKEN` or `GITHUB_TOKEN` — never embedded in owned files.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

import httpx2

from voicelm.domain.models import PageSpan
from voicelm.ingestion.cleaning import clean_text
from voicelm.ingestion.urls import DEFAULT_HEADERS, InvalidUrl, normalize_url

_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
_OWNER_REPO = re.compile(r"^[A-Za-z0-9_.-]+$")

# Text/code we index. Office/media/images stay out of 2E (those have dedicated ingest).
TEXT_SUFFIXES = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".py",
        ".pyi",
        ".ipynb",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".vue",
        ".svelte",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".m",
        ".mm",
        ".scala",
        ".clj",
        ".ex",
        ".exs",
        ".erl",
        ".hs",
        ".lua",
        ".r",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".bat",
        ".cmd",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".jsonc",
        ".xml",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".less",
        ".graphql",
        ".gql",
        ".proto",
        ".tf",
        ".hcl",
        ".ini",
        ".cfg",
        ".conf",
        ".env.example",
        ".gitignore",
        ".dockerignore",
        ".editorconfig",
        ".csv",
        ".tsv",
    }
)
EXTENSIONLESS_NAMES = frozenset(
    {
        "dockerfile",
        "makefile",
        "gemfile",
        "rakefile",
        "procfile",
        "license",
        "licence",
        "copying",
        "authors",
        "contributors",
        "changelog",
        "changes",
        "readme",
        "cmakelists.txt",
    }
)

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "vendor",
        "dist",
        "build",
        "out",
        "target",
        "__pycache__",
        ".idea",
        ".vscode",
        ".next",
        ".nuxt",
        ".turbo",
        "coverage",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "eggs",
        ".eggs",
        "Pods",
        "Carthage",
    }
)

MAX_ZIP_BYTES = 80 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 40 * 1024 * 1024
MAX_FILES = 400
DEFAULT_TIMEOUT_SECONDS = 60.0


class GitHubError(Exception):
    """Base for every reason a GitHub URL did not become usable sources."""


class NotGitHubUrl(GitHubError):
    """The URL is not a github.com repository link."""


class GitHubFetchError(GitHubError):
    """Network, auth, or API failure while fetching the snapshot."""


class EmptyGitHubRepo(GitHubError):
    """Snapshot had no indexable text files after filters."""


@dataclass(frozen=True)
class GitHubTarget:
    """Parsed github.com URL."""

    owner: str
    repo: str
    ref: str | None
    subpath: str  # "" = whole repo; for blob, the single file path
    is_blob: bool

    @property
    def canonical_repo_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"


@dataclass(frozen=True)
class GitHubFile:
    relative_path: str
    text: str
    pages: tuple[PageSpan, ...]
    origin_url: str


@dataclass(frozen=True)
class GitHubSnapshot:
    target: GitHubTarget
    ref: str
    commit_sha: str
    files: tuple[GitHubFile, ...]

    @property
    def storage_key(self) -> str:
        digest = hashlib.sha256(
            f"{self.target.owner}/{self.target.repo}@{self.commit_sha}".encode()
        ).hexdigest()[:16]
        return digest


def is_github_url(url: str) -> bool:
    """True when `url` looks like a github.com owner/repo link."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if host not in _GITHUB_HOSTS:
        return False
    try:
        parse_github_url(url)
    except (NotGitHubUrl, InvalidUrl, GitHubError):
        return False
    return True


def parse_github_url(url: str) -> GitHubTarget:
    """Pull owner/repo/(optional ref + path) out of common GitHub URL shapes."""
    canonical = normalize_url(url)
    parsed = urlparse(canonical)
    host = (parsed.hostname or "").lower()
    if host not in _GITHUB_HOSTS:
        raise NotGitHubUrl(f"{url} is not a GitHub repository link")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise NotGitHubUrl(f"{url} is not a GitHub repository link")

    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    if not _OWNER_REPO.match(owner) or not _OWNER_REPO.match(repo):
        raise NotGitHubUrl(f"{url} is not a GitHub repository link")

    if len(parts) == 2:
        return GitHubTarget(owner=owner, repo=repo, ref=None, subpath="", is_blob=False)

    kind = parts[2]
    if kind in {"tree", "blob"} and len(parts) >= 4:
        ref = parts[3]
        subpath = "/".join(parts[4:])
        return GitHubTarget(
            owner=owner,
            repo=repo,
            ref=ref,
            subpath=subpath,
            is_blob=(kind == "blob"),
        )
    if kind == "commit" and len(parts) >= 4:
        return GitHubTarget(owner=owner, repo=repo, ref=parts[3], subpath="", is_blob=False)

    # Issues, pulls, settings, etc. — not a repo snapshot.
    raise NotGitHubUrl(f"{url} is not a GitHub repository tree or blob link")


def github_token() -> str | None:
    token = os.environ.get("VOICELM_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token is None:
        return None
    cleaned = token.strip()
    return cleaned or None


def assemble_lines(raw: str) -> tuple[str, tuple[PageSpan, ...]]:
    """Clean text and map each line to a 1-based `PageSpan` (line number)."""
    cleaned = clean_text(raw)
    if not cleaned:
        return "", ()

    lines = cleaned.split("\n")
    spans: list[PageSpan] = []
    cursor = 0
    for number, line in enumerate(lines, start=1):
        if number > 1:
            cursor += 1  # the joining newline
        spans.append(PageSpan(number=number, start_char=cursor, end_char=cursor + len(line)))
        cursor += len(line)
    return "\n".join(lines), tuple(spans)


def fetch_github_repo(
    url: str,
    *,
    token: str | None = None,
    client: httpx2.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> GitHubSnapshot:
    """Download a zipball for `url` and return filtered text files with line spans."""
    target = parse_github_url(url)
    auth = token if token is not None else github_token()
    owned = client is None
    headers = {
        **DEFAULT_HEADERS,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if auth:
        headers["Authorization"] = f"Bearer {auth}"

    http = client or httpx2.Client(timeout=timeout, follow_redirects=True, headers=headers)
    try:
        ref = target.ref or _default_branch(http, target)
        commit_sha, zip_bytes = _download_zipball(http, target, ref)
        files = _files_from_zip(zip_bytes, target=target, ref=ref, commit_sha=commit_sha)
        if not files:
            raise EmptyGitHubRepo(
                f"{target.canonical_repo_url} had no indexable text files "
                f"(or the path filter matched nothing)"
            )
        return GitHubSnapshot(target=target, ref=ref, commit_sha=commit_sha, files=tuple(files))
    finally:
        if owned:
            http.close()


def _default_branch(http: httpx2.Client, target: GitHubTarget) -> str:
    api = f"https://api.github.com/repos/{target.owner}/{target.repo}"
    try:
        response = http.get(api)
    except httpx2.TimeoutException as error:
        raise GitHubFetchError(
            f"timed out reaching GitHub for {target.canonical_repo_url}"
        ) from error
    except httpx2.RequestError as error:
        raise GitHubFetchError(f"could not reach GitHub: {error}") from error

    if response.status_code in {401, 403}:
        raise GitHubFetchError(
            f"{target.canonical_repo_url} requires authentication "
            "(set VOICELM_GITHUB_TOKEN or GITHUB_TOKEN)"
        )
    if response.status_code == 404:
        raise GitHubFetchError(f"{target.canonical_repo_url} was not found")
    if response.status_code >= 400:
        raise GitHubFetchError(
            f"GitHub returned HTTP {response.status_code} for {target.canonical_repo_url}"
        )

    payload = response.json()
    branch = payload.get("default_branch")
    if not isinstance(branch, str) or not branch:
        raise GitHubFetchError(f"{target.canonical_repo_url} has no default branch")
    return branch


def _download_zipball(
    http: httpx2.Client, target: GitHubTarget, ref: str
) -> tuple[str, bytes]:
    api = f"https://api.github.com/repos/{target.owner}/{target.repo}/zipball/{ref}"
    try:
        response = http.get(api)
    except httpx2.TimeoutException as error:
        raise GitHubFetchError(f"timed out downloading zipball for {ref}") from error
    except httpx2.RequestError as error:
        raise GitHubFetchError(f"could not download zipball: {error}") from error

    if response.status_code in {401, 403}:
        raise GitHubFetchError(
            f"{target.canonical_repo_url}@{ref} requires authentication "
            "(set VOICELM_GITHUB_TOKEN or GITHUB_TOKEN)"
        )
    if response.status_code == 404:
        raise GitHubFetchError(f"{target.canonical_repo_url}@{ref} was not found")
    if response.status_code >= 400:
        raise GitHubFetchError(
            f"GitHub returned HTTP {response.status_code} downloading {target.repo}@{ref}"
        )

    raw = response.content
    if len(raw) > MAX_ZIP_BYTES:
        raise GitHubFetchError(
            f"{target.canonical_repo_url}@{ref} zipball is larger than "
            f"{MAX_ZIP_BYTES // (1024 * 1024)} MB"
        )

    commit_sha = _commit_sha_from_response(response, ref)
    return commit_sha, raw


def _commit_sha_from_response(response: httpx2.Response, fallback_ref: str) -> str:
    # Content-Disposition: attachment; filename=owner-repo-<sha>.zip
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r"filename[^;=\n]*=(['\"]?)([^'\"\s]+)\1", disposition, re.I)
    if match:
        name = match.group(2)
        stem = PurePosixPath(name).stem
        if "-" in stem:
            maybe_sha = stem.rsplit("-", 1)[-1]
            if re.fullmatch(r"[0-9a-f]{7,40}", maybe_sha):
                return maybe_sha
    return fallback_ref


def _files_from_zip(
    zip_bytes: bytes,
    *,
    target: GitHubTarget,
    ref: str,
    commit_sha: str,
) -> list[GitHubFile]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as error:
        raise GitHubFetchError("GitHub returned a corrupt zipball") from error

    selected: list[GitHubFile] = []
    total_bytes = 0

    with archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        for info in members:
            relative = _strip_zip_root(info.filename)
            if relative is None:
                continue
            if not _should_include(relative, target):
                continue
            if info.file_size > MAX_FILE_BYTES:
                continue

            raw = archive.read(info)
            total_bytes += len(raw)
            if total_bytes > MAX_TOTAL_BYTES:
                raise GitHubFetchError(
                    f"{target.canonical_repo_url} exceeds the "
                    f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB text budget after filters"
                )

            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    decoded = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    continue

            text, pages = assemble_lines(decoded)
            if not text:
                continue

            blob_url = (
                f"{target.canonical_repo_url}/blob/{ref}/{relative}"
                if ref
                else f"{target.canonical_repo_url}/blob/{commit_sha}/{relative}"
            )
            selected.append(
                GitHubFile(
                    relative_path=relative,
                    text=text,
                    pages=pages,
                    origin_url=blob_url,
                )
            )
            if len(selected) > MAX_FILES:
                raise GitHubFetchError(
                    f"{target.canonical_repo_url} has more than {MAX_FILES} indexable files; "
                    "narrow the URL to a subdirectory or blob"
                )

    return selected


def _strip_zip_root(name: str) -> str | None:
    # Zip entries look like "owner-repo-sha/path/to/file".
    parts = [part for part in name.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        return None
    return "/".join(parts[1:])


def _should_include(relative: str, target: GitHubTarget) -> bool:
    path = PurePosixPath(relative)
    for part in path.parts[:-1]:
        if part in SKIP_DIR_NAMES:
            return False
        if part.startswith(".") and part != ".github":
            return False

    if target.subpath:
        if target.is_blob:
            if relative != target.subpath:
                return False
        else:
            prefix = target.subpath.rstrip("/") + "/"
            if relative != target.subpath and not relative.startswith(prefix):
                return False

    return _is_text_path(path)


def _is_text_path(path: PurePosixPath) -> bool:
    name = path.name.lower()
    if name in EXTENSIONLESS_NAMES:
        return True
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return True
    return name.endswith(".env.example")
