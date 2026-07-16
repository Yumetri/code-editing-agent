"""Small shared building blocks for the tutorial agents.

The numbered examples keep the learning flow, while this module holds repeated
mechanics that are not the main point of each step.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from .console import print_tool_call


@dataclass
class ToolDefinition:
    """A local Python function plus the metadata shown to the LLM."""

    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[[dict[str, Any]], str]


def append_user_message(
    conversation: list[ChatCompletionMessageParam],
    user_input: str,
) -> None:
    """Add a user's text input in Chat Completions message format."""
    conversation.append({
        "role": "user",
        "content": user_input,
    })


def to_openai_tools(tools: list[ToolDefinition]) -> list[ChatCompletionToolParam]:
    """Convert local ToolDefinition objects into OpenAI-compatible tool schemas."""
    openai_tools: list[ChatCompletionToolParam] = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        })
    return openai_tools


def parse_tool_arguments(tool_call: Any) -> dict[str, Any]:
    """Parse the JSON arguments attached to a model tool call."""
    arguments = tool_call.function.arguments
    try:
        parsed_arguments = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError):
        return {}

    if not isinstance(parsed_arguments, dict):
        return {}
    return parsed_arguments


def run_tool(
    tools: list[ToolDefinition],
    name: str,
    input_data: dict[str, Any],
) -> tuple[str, bool]:
    """Run a registered tool by name and return (result, is_error)."""
    tool_def = find_tool(tools=tools, name=name)
    if tool_def is None:
        return "tool not found", True

    print_tool_call(name=name, input_data=input_data)

    try:
        return tool_def.function(input_data), False
    except Exception as error:
        return str(error), True


def append_tool_result_message(
    conversation: list[ChatCompletionMessageParam],
    tool_call: Any,
    tool_name: str,
    tool_response: str,
    is_error: bool,
) -> None:
    """Append a tool response so the model can continue from the tool result."""
    conversation.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_name,
        "content": json.dumps(
            {
                "result": tool_response,
                "is_error": is_error,
            },
            ensure_ascii=False,
        ),
    })


def find_tool(tools: list[ToolDefinition], name: str) -> ToolDefinition | None:
    """Return the registered tool with the requested name, if one exists."""
    for tool in tools:
        if tool.name == name:
            return tool
    return None
