"""Shared LLM provider setup for the tutorial agents."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import AuthenticationError, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PROVIDER = "openrouter"
DEFAULT_LOG_LEVEL = "WARNING"
QUIET_EXTERNAL_LOGGERS = ("httpx", "httpcore", "openai")

OPENROUTER_API_KEY_NAME = "OPENROUTER_API_KEY"
OPENROUTER_MODEL_ENV_NAME = "OPENROUTER_MODEL_NAME"
DEFAULT_OPENROUTER_MODEL = "poolside/laguna-m.1:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GEMINI_API_KEY_NAME = "GEMINI_API_KEY"
GEMINI_MODEL_ENV_NAME = "GEMINI_MODEL_NAME"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

AUTH_FAILURE_STATUS_CODES = {401, 403}


class MissingApiKeyError(RuntimeError):
    """Raised when a provider cannot run because its API key is missing."""

    def __init__(self, provider_name: str, api_key_name: str) -> None:
        self.provider_name = provider_name
        self.api_key_name = api_key_name
        super().__init__(f"Missing {api_key_name}.")


class ChatModel(Protocol):
    """Minimal interface used by the tutorial agents."""

    provider_name: str
    model_name: str

    def complete(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        """Return the next chat completion for the given conversation."""


@dataclass(frozen=True)
class ProviderConfig:
    """Connection settings for one OpenAI-compatible provider."""

    name: str
    api_key_name: str
    model_env_name: str
    default_model_name: str
    base_url: str

    def model_name_from(self, env: Mapping[str, str]) -> str:
        configured_model = env.get(self.model_env_name, "").strip()
        return configured_model or self.default_model_name


@dataclass
class OpenAICompatibleChatModel:
    """ChatModel backed by the OpenAI SDK's Chat Completions API."""

    provider_name: str
    model_name: str
    client: Any

    def complete(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        return self.client.chat.completions.create(**kwargs)


@dataclass
class MissingApiKeyChatModel:
    """ChatModel placeholder used so OpenRouter can fall back when its key is missing."""

    provider_name: str
    model_name: str
    api_key_name: str

    def complete(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        raise MissingApiKeyError(
            provider_name=self.provider_name,
            api_key_name=self.api_key_name,
        )


@dataclass
class FallbackChatModel:
    """Try the primary provider, then fall back on authentication failures."""

    primary: ChatModel
    fallback: ChatModel

    @property
    def provider_name(self) -> str:
        return self.primary.provider_name

    @property
    def model_name(self) -> str:
        return self.primary.model_name

    def complete(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> ChatCompletion:
        try:
            return self.primary.complete(messages=messages, tools=tools)
        except Exception as primary_error:
            if not _is_auth_failure(primary_error):
                LOGGER.exception(
                    "Provider '%s' failed with %s; not falling back.",
                    self.primary.provider_name,
                    _error_summary(primary_error),
                )
                raise

            LOGGER.warning(
                "Provider '%s' authentication failed with %s; falling back to '%s'.",
                self.primary.provider_name,
                _error_summary(primary_error),
                self.fallback.provider_name,
            )

        try:
            response = self.fallback.complete(messages=messages, tools=tools)
            LOGGER.info(
                "Fallback provider '%s' completed the request with model '%s'.",
                self.fallback.provider_name,
                self.fallback.model_name,
            )
            return response
        except Exception as fallback_error:
            LOGGER.exception(
                "Fallback provider '%s' failed with %s.",
                self.fallback.provider_name,
                _error_summary(fallback_error),
            )
            raise


ClientFactory = Callable[..., Any]


def new_chat_model(
    env: Mapping[str, str] | None = None,
    client_factory: ClientFactory = OpenAI,
) -> ChatModel:
    """Create the configured chat model and optional Gemini fallback."""
    load_dotenv()
    _configure_logging()

    env = os.environ if env is None else env
    provider_name = env.get("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()

    openrouter_config = ProviderConfig(
        name="openrouter",
        api_key_name=OPENROUTER_API_KEY_NAME,
        model_env_name=OPENROUTER_MODEL_ENV_NAME,
        default_model_name=DEFAULT_OPENROUTER_MODEL,
        base_url=OPENROUTER_BASE_URL,
    )
    gemini_config = ProviderConfig(
        name="gemini",
        api_key_name=GEMINI_API_KEY_NAME,
        model_env_name=GEMINI_MODEL_ENV_NAME,
        default_model_name=DEFAULT_GEMINI_MODEL,
        base_url=GEMINI_BASE_URL,
    )

    if provider_name == "gemini":
        model = _build_provider_model(
            config=gemini_config,
            env=env,
            client_factory=client_factory,
            required=True,
        )
        LOGGER.info(
            "Using provider '%s' with model '%s'.",
            model.provider_name,
            model.model_name,
        )
        return model

    if provider_name != "openrouter":
        raise ValueError("LLM_PROVIDER must be 'openrouter' or 'gemini'.")

    primary = _build_provider_model(
        config=openrouter_config,
        env=env,
        client_factory=client_factory,
        required=False,
    )
    fallback = _build_provider_model(
        config=gemini_config,
        env=env,
        client_factory=client_factory,
        required=False,
    )

    fallback_state = (
        "enabled"
        if not isinstance(fallback, MissingApiKeyChatModel)
        else f"waiting for {GEMINI_API_KEY_NAME}"
    )
    LOGGER.info(
        "Using provider '%s' with model '%s'; Gemini fallback is %s.",
        primary.provider_name,
        primary.model_name,
        fallback_state,
    )
    return FallbackChatModel(primary=primary, fallback=fallback)


def _build_provider_model(
    config: ProviderConfig,
    env: Mapping[str, str],
    client_factory: ClientFactory,
    required: bool,
) -> ChatModel:
    model_name = config.model_name_from(env)
    api_key = env.get(config.api_key_name, "").strip()

    if not api_key:
        if required:
            LOGGER.error(
                "Provider '%s' cannot start because %s is missing.",
                config.name,
                config.api_key_name,
            )
            raise MissingApiKeyError(
                provider_name=config.name,
                api_key_name=config.api_key_name,
            )

        return MissingApiKeyChatModel(
            provider_name=config.name,
            model_name=model_name,
            api_key_name=config.api_key_name,
        )

    client = client_factory(
        base_url=config.base_url,
        api_key=api_key,
    )
    return OpenAICompatibleChatModel(
        provider_name=config.name,
        model_name=model_name,
        client=client,
    )


def _is_auth_failure(error: Exception) -> bool:
    if isinstance(error, MissingApiKeyError | AuthenticationError):
        return True

    return _status_code(error) in AUTH_FAILURE_STATUS_CODES


def _status_code(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    response_status_code = getattr(response, "status_code", None)
    if isinstance(response_status_code, int):
        return response_status_code

    return None


def _error_summary(error: Exception) -> str:
    status_code = _status_code(error)
    if status_code is None:
        return error.__class__.__name__
    return f"{error.__class__.__name__}(status={status_code})"


def _configure_logging() -> None:
    _quiet_external_loggers()

    if logging.getLogger().handlers:
        return

    level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _quiet_external_loggers() -> None:
    for logger_name in QUIET_EXTERNAL_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
