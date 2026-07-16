from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_example(filename: str) -> Any:
    module_name = filename.replace(".py", "").replace("_", "_test_")
    module_path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant"}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]
        return message


class FakeResponse:
    def __init__(self, message: FakeMessage) -> None:
        self.choices = [SimpleNamespace(message=message)]


class FakeChatModel:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> FakeResponse:
        self.calls.append({
            "messages": [dict(message) for message in messages],
            "tools": tools or [],
        })
        return self.responses.pop(0)


def fake_input(*messages: str):
    inputs = list(messages)

    def get_next_message() -> tuple[str, bool]:
        if not inputs:
            return "", False
        return inputs.pop(0), True

    return get_next_message


def tool_call(name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def test_stage1_sends_user_message_to_chat_model() -> None:
    stage1 = load_example("1_agent_basic.py")
    chat_model = FakeChatModel([
        FakeResponse(FakeMessage(content="hello")),
    ])
    agent = stage1.new_agent(
        chat_model=chat_model,
        get_user_msg_fn=fake_input("Hi"),
    )

    agent.run()

    assert chat_model.calls == [{
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [],
    }]


def test_stage2_exposes_tool_and_executes_it_without_feedback(tmp_path: Path) -> None:
    stage2 = load_example("2_agent_tool.py")
    target = tmp_path / "note.txt"
    target.write_text("secret", encoding="utf-8")
    chat_model = FakeChatModel([
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("read_file", {"path": str(target)}),
        ])),
    ])
    agent = stage2.new_agent(
        chat_model=chat_model,
        get_user_msg_fn=fake_input("Read the file"),
        tools=[stage2.READ_FILE_DEFINITION],
    )

    agent.run()

    assert len(chat_model.calls) == 1
    assert chat_model.calls[0]["tools"][0]["function"]["name"] == "read_file"


def test_stage3_appends_tool_result_before_second_model_call(tmp_path: Path) -> None:
    stage3 = load_example("3_agent_tool_loop.py")
    target = tmp_path / "note.txt"
    target.write_text("tool result", encoding="utf-8")
    chat_model = FakeChatModel([
        FakeResponse(FakeMessage(tool_calls=[
            tool_call("read_file", {"path": str(target)}),
        ])),
        FakeResponse(FakeMessage(content="done")),
    ])
    agent = stage3.new_agent(
        chat_model=chat_model,
        get_user_msg_fn=fake_input("Read the file"),
        tools=[stage3.READ_FILE_DEFINITION],
    )

    agent.run()

    assert len(chat_model.calls) == 2
    second_messages = chat_model.calls[1]["messages"]
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["name"] == "read_file"
    tool_content = json.loads(second_messages[-1]["content"])
    assert tool_content == {"result": "tool result", "is_error": False}


def test_stage4_file_tools_can_list_create_and_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage4 = load_example("4_agent_tool_extend.py")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing.txt").write_text("hello old", encoding="utf-8")

    assert "existing.txt" in stage4.list_dir({"path": "."})
    assert stage4.edit_file({
        "path": "new.txt",
        "old_str": "",
        "new_str": "created",
    }) == "Successfully created file new.txt"
    assert stage4.edit_file({
        "path": "existing.txt",
        "old_str": "old",
        "new_str": "new",
    }) == "OK"
    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "hello new"


def test_stage5_workspace_tools_are_safe_and_can_run_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage5 = load_example("5_code_agent.py")
    monkeypatch.setattr(stage5, "WORKSPACE_ROOT", tmp_path.resolve())

    assert stage5.write_file({
        "path": "app.js",
        "content": "console.log('ok');\n",
    }) == "Wrote app.js"
    assert stage5.read_file({"path": "app.js"}) == "console.log('ok');\n"
    assert stage5.edit_file({
        "path": "app.js",
        "old_str": "ok",
        "new_str": "done",
    }) == "Edited app.js"
    assert "app.js" in stage5.list_dir({"path": "."})

    with pytest.raises(ValueError, match="inside the working directory"):
        stage5.read_file({"path": "../outside.txt"})

    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")

    result = json.loads(stage5.run_node_file({"path": "app.js"}))
    assert result["exit_code"] == 0
    assert result["stdout"] == "done\n"
