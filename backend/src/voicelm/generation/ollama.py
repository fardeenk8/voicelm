"""Calling a local Ollama chat model.

POST /api/chat  {"model": ..., "messages": [...], "stream": false, "options": {...}}
-> {"message": {"role": "assistant", "content": ...}, "prompt_eval_count": N, ...}

With stream=true the same path returns NDJSON lines. Each line is a delta; the last has
done=true and the token counts. That is what chat_stream consumes.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx2

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_NUM_CTX = 8192
DEFAULT_TIMEOUT_SECONDS = 300.0


class GenerationError(Exception):
    """Raised when Ollama cannot produce a usable answer."""


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class ChatModel(Protocol):
    """Anything that turns a prompt into an answer. Tests pass a fake; production uses Ollama."""

    model: str

    def chat(self, system: str, user: str) -> ChatResult: ...

    def chat_stream(self, system: str, user: str) -> Iterator[str]: ...


class OllamaChatModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx2.Client | None = None,
    ) -> None:
        self.model = model
        # Set explicitly because Ollama otherwise picks a default from available memory
        # (4096 on this machine) and silently truncates anything longer. See ADR-0016.
        self.num_ctx = num_ctx
        # Zero temperature: we want the excerpts reproduced faithfully, not creatively.
        # It also makes the tests deterministic.
        self.temperature = temperature
        self._client = client or httpx2.Client(base_url=base_url, timeout=timeout)

    def chat(self, system: str, user: str) -> ChatResult:
        try:
            response = self._client.post("/api/chat", json=self._body(system, user, stream=False))
            response.raise_for_status()
        except httpx2.HTTPStatusError as error:
            raise GenerationError(
                f"Ollama returned {error.response.status_code}: {error.response.text}"
            ) from error
        except httpx2.RequestError as error:
            raise self._unreachable(error) from error

        payload = response.json()
        message = payload.get("message")

        if not isinstance(message, dict) or "content" not in message:
            raise GenerationError(f"unexpected response from Ollama; keys were {sorted(payload)}")

        self._reject_truncated_prompt(int(payload.get("prompt_eval_count", 0)))

        return ChatResult(
            text=str(message["content"]).strip(),
            prompt_tokens=int(payload.get("prompt_eval_count", 0)),
            completion_tokens=int(payload.get("eval_count", 0)),
        )

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        """Yield content deltas as Ollama produces them.

        The CLI and `POST /ask` still use `chat` (one JSON blob). The desktop UI uses
        this iterator so words can appear before the model finishes. Token-count
        truncation is checked on the final `done` frame, same rule as `chat`.
        """
        try:
            with self._client.stream(
                "POST", "/api/chat", json=self._body(system, user, stream=True)
            ) as response:
                response.raise_for_status()
                prompt_tokens = 0
                saw_done = False
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise GenerationError(
                            f"unexpected stream frame from Ollama: {line!r}"
                        ) from error
                    if not isinstance(payload, dict):
                        raise GenerationError("unexpected stream frame from Ollama")

                    message = payload.get("message")
                    if isinstance(message, dict):
                        piece = message.get("content")
                        if isinstance(piece, str) and piece:
                            yield piece

                    if payload.get("done"):
                        saw_done = True
                        prompt_tokens = int(payload.get("prompt_eval_count", 0))
                        break

                if not saw_done:
                    raise GenerationError("Ollama stream ended before a done frame")
                self._reject_truncated_prompt(prompt_tokens)
        except httpx2.HTTPStatusError as error:
            raise GenerationError(
                f"Ollama returned {error.response.status_code}: {error.response.text}"
            ) from error
        except httpx2.RequestError as error:
            raise self._unreachable(error) from error

    def _body(self, system: str, user: str, *, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }

    def _reject_truncated_prompt(self, prompt_tokens: int) -> None:
        # If the input filled the whole window, Ollama dropped the beginning of it without
        # saying so, and some excerpts never reached the model. The answer may look fine
        # and be ungrounded, so fail loudly instead.
        if prompt_tokens >= self.num_ctx:
            raise GenerationError(
                f"prompt used {prompt_tokens} tokens against a context window of "
                f"{self.num_ctx}; excerpts were truncated. Retrieve fewer chunks or "
                "raise num_ctx."
            )

    def _unreachable(self, error: httpx2.RequestError) -> GenerationError:
        return GenerationError(
            f"could not reach Ollama ({error}). Is it running? "
            "Start it with: brew services start ollama"
        )
