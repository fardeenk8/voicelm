"""Offline tests for the chat client."""

import json

import httpx2
import pytest

from voicelm.generation.ollama import GenerationError, OllamaChatModel


def model_returning(
    payload: dict, status_code: int = 200, **kwargs
) -> tuple[OllamaChatModel, list[dict]]:
    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(status_code, json=payload)

    client = httpx2.Client(base_url="http://testserver", transport=httpx2.MockTransport(handler))
    return OllamaChatModel(client=client, **kwargs), sent


def reply(content: str, prompt_tokens: int = 100, completion_tokens: int = 20) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    }


def test_returns_the_assistant_message() -> None:
    model, _ = model_returning(reply("  Caching helps [1].  "))

    result = model.chat("system", "user")

    assert result.text == "Caching helps [1]."
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20


def test_sends_both_messages_in_order() -> None:
    model, sent = model_returning(reply("ok"))

    model.chat("the rules", "the question")

    assert sent[0]["messages"] == [
        {"role": "system", "content": "the rules"},
        {"role": "user", "content": "the question"},
    ]


def test_context_window_and_temperature_are_sent_explicitly() -> None:
    # ADR-0016: left unset, Ollama picks a window from available memory and silently
    # truncates. Temperature zero keeps answers faithful and tests deterministic.
    model, sent = model_returning(reply("ok"), num_ctx=8192)

    model.chat("system", "user")

    assert sent[0]["options"] == {"num_ctx": 8192, "temperature": 0.0}
    assert sent[0]["stream"] is False


# --- failure modes ----------------------------------------------------------------


def test_a_full_context_window_is_treated_as_truncation() -> None:
    # The prompt filling the window means Ollama dropped the front of it, so some excerpts
    # never reached the model and the answer is not grounded in what we sent.
    model, _ = model_returning(reply("ok", prompt_tokens=4096), num_ctx=4096)

    with pytest.raises(GenerationError, match="truncated"):
        model.chat("system", "user")


def test_a_prompt_that_fits_is_accepted() -> None:
    model, _ = model_returning(reply("ok", prompt_tokens=4095), num_ctx=4096)

    assert model.chat("system", "user").text == "ok"


def test_missing_message_is_an_error() -> None:
    model, _ = model_returning({"error": "model not found"})

    with pytest.raises(GenerationError, match="unexpected response"):
        model.chat("system", "user")


def test_http_error_status_is_wrapped() -> None:
    model, _ = model_returning({"error": "no such model"}, status_code=404)

    with pytest.raises(GenerationError, match="404"):
        model.chat("system", "user")


def test_connection_failure_explains_how_to_fix_it() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    client = httpx2.Client(base_url="http://testserver", transport=httpx2.MockTransport(handler))

    with pytest.raises(GenerationError, match="brew services start ollama"):
        OllamaChatModel(client=client).chat("system", "user")


def _ndjson(*frames: dict) -> bytes:
    return "".join(json.dumps(frame) + "\n" for frame in frames).encode()


def model_streaming(frames: list[dict], **kwargs) -> tuple[OllamaChatModel, list[dict]]:
    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, content=_ndjson(*frames))

    client = httpx2.Client(base_url="http://testserver", transport=httpx2.MockTransport(handler))
    return OllamaChatModel(client=client, **kwargs), sent


def test_chat_stream_yields_deltas_and_asks_ollama_to_stream() -> None:
    model, sent = model_streaming(
        [
            {"message": {"role": "assistant", "content": "Hel"}, "done": False},
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 40,
                "eval_count": 2,
            },
        ]
    )

    assert "".join(model.chat_stream("system", "user")) == "Hello"
    assert sent[0]["stream"] is True


def test_chat_stream_treats_a_full_context_window_as_truncation() -> None:
    model, _ = model_streaming(
        [
            {"message": {"content": "ok"}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 4096},
        ],
        num_ctx=4096,
    )

    with pytest.raises(GenerationError, match="truncated"):
        list(model.chat_stream("system", "user"))


def test_chat_stream_errors_when_the_done_frame_never_arrives() -> None:
    model, _ = model_streaming([{"message": {"content": "Hel"}, "done": False}])

    with pytest.raises(GenerationError, match="done frame"):
        list(model.chat_stream("system", "user"))
