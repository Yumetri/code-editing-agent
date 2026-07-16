from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import AuthenticationError

from agent_lib.llm import (
    GEMINI_BASE_URL,
    MissingApiKeyError,
    OpenAICompatibleChatModel,
    FallbackChatModel,
    new_chat_model,
)


class FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.completions = FakeCompletions(response=response, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def auth_error(status_code: int = 401) -> AuthenticationError:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    response = httpx.Response(status_code, request=request)
    return AuthenticationError("auth failed", response=response, body=None)


def test_openai_compatible_model_passes_model_messages_and_tools() -> None:
    client = FakeClient(response="ok")
    model = OpenAICompatibleChatModel(
        provider_name="test-provider",
        model_name="test-model",
        client=client,
    )
    messages = [{"role": "user", "content": "hello"}]
    tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {}},
        },
    }]

    response = model.complete(messages=messages, tools=tools)

    assert response == "ok"
    assert client.completions.calls == [{
        "model": "test-model",
        "messages": messages,
        "tools": tools,
    }]


def test_new_chat_model_can_select_gemini_without_openrouter_key() -> None:
    created_clients: list[dict[str, Any]] = []

    def client_factory(**kwargs: Any) -> FakeClient:
        created_clients.append(kwargs)
        return FakeClient(response="gemini-response")

    model = new_chat_model(
        env={
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "gemini-secret",
        },
        client_factory=client_factory,
    )

    assert isinstance(model, OpenAICompatibleChatModel)
    assert model.provider_name == "gemini"
    assert model.model_name == "gemini-2.5-flash"
    assert created_clients == [{
        "base_url": GEMINI_BASE_URL,
        "api_key": "gemini-secret",
    }]


def test_openrouter_auth_error_falls_back_to_gemini(caplog: pytest.LogCaptureFixture) -> None:
    primary = OpenAICompatibleChatModel(
        provider_name="openrouter",
        model_name="openrouter-model",
        client=FakeClient(error=auth_error()),
    )
    fallback = OpenAICompatibleChatModel(
        provider_name="gemini",
        model_name="gemini-model",
        client=FakeClient(response="gemini-ok"),
    )
    model = FallbackChatModel(primary=primary, fallback=fallback)

    with caplog.at_level("INFO"):
        response = model.complete(messages=[{"role": "user", "content": "hello"}])

    assert response == "gemini-ok"
    assert "authentication failed" in caplog.text
    assert "Fallback provider 'gemini' completed" in caplog.text


def test_missing_openrouter_key_falls_back_to_gemini(caplog: pytest.LogCaptureFixture) -> None:
    def client_factory(**kwargs: Any) -> FakeClient:
        assert kwargs["api_key"] == "gemini-secret"
        return FakeClient(response="gemini-ok")

    model = new_chat_model(
        env={
            "LLM_PROVIDER": "openrouter",
            "GEMINI_API_KEY": "gemini-secret",
        },
        client_factory=client_factory,
    )

    with caplog.at_level("INFO"):
        response = model.complete(messages=[{"role": "user", "content": "hello"}])

    assert response == "gemini-ok"
    assert "MissingApiKeyError" in caplog.text
    assert "falling back to 'gemini'" in caplog.text


def test_openrouter_non_auth_error_does_not_fall_back(caplog: pytest.LogCaptureFixture) -> None:
    fallback_client = FakeClient(response="should-not-run")
    model = FallbackChatModel(
        primary=OpenAICompatibleChatModel(
            provider_name="openrouter",
            model_name="openrouter-model",
            client=FakeClient(error=RuntimeError("network down")),
        ),
        fallback=OpenAICompatibleChatModel(
            provider_name="gemini",
            model_name="gemini-model",
            client=fallback_client,
        ),
    )

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        model.complete(messages=[{"role": "user", "content": "hello"}])

    assert fallback_client.completions.calls == []
    assert "not falling back" in caplog.text


def test_gemini_fallback_failure_is_logged_and_raised(caplog: pytest.LogCaptureFixture) -> None:
    model = FallbackChatModel(
        primary=OpenAICompatibleChatModel(
            provider_name="openrouter",
            model_name="openrouter-model",
            client=FakeClient(error=auth_error()),
        ),
        fallback=OpenAICompatibleChatModel(
            provider_name="gemini",
            model_name="gemini-model",
            client=FakeClient(error=RuntimeError("gemini failed")),
        ),
    )

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="gemini failed"):
        model.complete(messages=[{"role": "user", "content": "hello"}])

    assert "Fallback provider 'gemini' failed" in caplog.text


def test_missing_direct_gemini_key_logs_without_leaking_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("ERROR"), pytest.raises(MissingApiKeyError):
        new_chat_model(
            env={
                "LLM_PROVIDER": "gemini",
                "OPENROUTER_API_KEY": "openrouter-secret",
            },
            client_factory=lambda **_: FakeClient(),
        )

    assert "GEMINI_API_KEY" in caplog.text
    assert "openrouter-secret" not in caplog.text


def test_fallback_logs_do_not_leak_api_keys(caplog: pytest.LogCaptureFixture) -> None:
    def client_factory(**kwargs: Any) -> FakeClient:
        if kwargs["api_key"] == "openrouter-secret":
            return FakeClient(error=auth_error())
        return FakeClient(response="gemini-ok")

    model = new_chat_model(
        env={
            "LLM_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-secret",
            "GEMINI_API_KEY": "gemini-secret",
        },
        client_factory=client_factory,
    )

    with caplog.at_level("INFO"):
        assert model.complete(messages=[{"role": "user", "content": "hello"}]) == "gemini-ok"

    assert "openrouter-secret" not in caplog.text
    assert "gemini-secret" not in caplog.text


def test_new_chat_model_quiets_http_client_info_logs() -> None:
    new_chat_model(
        env={
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "gemini-secret",
        },
        client_factory=lambda **_: FakeClient(),
    )

    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("openai").getEffectiveLevel() >= logging.WARNING
