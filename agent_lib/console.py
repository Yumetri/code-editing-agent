"""Shared terminal input and output helpers for the tutorial agents."""

from __future__ import annotations

from typing import Any

COLOR_USER = "\033[94m"  # Blue
COLOR_LLM = "\033[93m"  # Yellow
COLOR_TOOL = "\033[92m"  # Green
COLOR_RESET = "\033[0m"


def get_user_message() -> tuple[str, bool]:
    """Read one terminal input line and return False on normal exit signals."""
    try:
        text = input()
        return text, True
    except (EOFError, KeyboardInterrupt):
        return "", False


def print_chat_banner() -> None:
    print("Chat with the configured LLM provider. use ctrl-c to quit.")


def print_user_prompt() -> None:
    print(f"{COLOR_USER}You{COLOR_RESET}: ", end="")


def print_llm_message(content: str) -> None:
    print(f"{COLOR_LLM}LLM{COLOR_RESET}: {content}")


def print_tool_call(name: str, input_data: dict[str, Any]) -> None:
    print(f"{COLOR_TOOL}tool{COLOR_RESET}: {name}({input_data})")


def print_no_response() -> None:
    print("No response from the configured LLM provider.")


def print_error(error: Exception) -> None:
    print(f"Error: {error}")
