"""5단계: 작업공간 보호와 실행 검증을 갖춘 코드 편집 에이전트.

이 파일은 앞 단계들의 완성형 예제입니다. 모델은 파일 읽기, 목록 조회,
파일 생성/수정, JavaScript 실행 도구를 사용할 수 있고, 프로그램은 모든 파일
경로가 현재 작업 디렉터리 안에 머무는지 검사합니다.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)

from agent_lib.console import (
    get_user_message,
    print_chat_banner,
    print_error,
    print_llm_message,
    print_no_response,
    print_user_prompt,
)
from agent_lib.core import (
    ToolDefinition,
    append_tool_result_message,
    append_user_message,
    parse_tool_arguments,
    run_tool,
    to_openai_tools,
)
from agent_lib.llm import ChatModel, new_chat_model

# LLM provider 생성과 fallback 설정은 agent_lib/llm.py에 모아 두고, 여기에는 파일 입출력과
# 작업공간 제한 설정만 남깁니다.
DEFAULT_ENCODING = "utf-8"
# 에이전트가 파일을 읽고 쓸 수 있는 최상위 폴더입니다.
# Path.cwd()는 이 스크립트를 실행한 현재 디렉터리를 의미합니다.
WORKSPACE_ROOT = Path.cwd().resolve()

# 시스템 프롬프트는 모델에게 "이 에이전트가 어떤 역할을 해야 하는지"와
# "어떤 상황에서 어떤 도구를 써야 하는지"를 알려주는 운영 지침입니다.
SYSTEM_PROMPT = """
You are a code editing agent working in the current directory.

Use the available tools whenever the user asks you to inspect files, create
files, edit files, or run JavaScript with Node.js.

Important behavior:
- When creating a file, call write_file with the complete file content.
- When changing an existing file, call edit_file with an exact old_str and new_str.
- When the user asks you to run or verify a JavaScript file, call run_node_file.
- Do not say the task is complete until the relevant tool calls have succeeded.
""".strip()


def resolve_workspace_path(path: str) -> Path:
    """사용자/모델이 준 상대 경로를 안전한 절대 경로로 변환합니다.

    `../secret.txt`처럼 작업공간 밖으로 나가려는 경로는 ValueError로 막습니다.
    코드 편집 에이전트에서 가장 중요한 안전장치 중 하나입니다.
    """
    if not path:
        raise ValueError("Path must be a non-empty string.")

    resolved_path = (WORKSPACE_ROOT / path).resolve()
    try:
        # resolved_path가 WORKSPACE_ROOT 하위인지 확인합니다.
        # 하위가 아니면 relative_to가 ValueError를 발생시킵니다.
        resolved_path.relative_to(WORKSPACE_ROOT)
    except ValueError as error:
        raise ValueError("Path must stay inside the working directory.") from error

    return resolved_path


def read_file(input_data: dict[str, Any]) -> str:
    """작업공간 내부의 텍스트 파일을 읽어 반환합니다."""
    path = input_data.get("path")

    if not isinstance(path, str):
        raise ValueError("Path must be a non-empty string.")

    with resolve_workspace_path(path).open("r", encoding=DEFAULT_ENCODING) as file:
        return file.read()


# read_file 도구의 입력 스키마입니다. 모델은 이 정보를 보고 {"path": "..."}를 만듭니다.
READ_FILE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The relative path of a file in the working directory.",
        },
    },
    "required": ["path"],
}

READ_FILE_DEFINITION = ToolDefinition(
    name="read_file",
    description=(
        "Read the contents of a given relative file path. "
        "Use this when you want to see what's inside a file. "
        "Do not use this with directory names."
    ),
    input_schema=READ_FILE_INPUT_SCHEMA,
    function=read_file,
)


def list_dir(input_data: dict[str, Any]) -> str:
    """작업공간 내부 디렉터리의 항목을 정렬된 문자열로 반환합니다."""
    path = input_data.get("path", ".")

    if not isinstance(path, str):
        raise ValueError("Path must be a string.")

    # resolve_workspace_path를 거치므로 디렉터리 조회도 작업공간 안으로 제한됩니다.
    directory_path = resolve_workspace_path(path)
    return "\n".join(sorted(entry.name for entry in directory_path.iterdir()))


# path를 생략하면 현재 작업공간 루트를 조회합니다.
LIST_DIR_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "The relative path of a directory in the working directory. "
                "Defaults to '.' if not specified."
            ),
        },
    },
}

LIST_DIR_DEFINITION = ToolDefinition(
    name="list_dir",
    description=(
        "List the contents of a given directory path. "
        "Use this to see what files and folders are present in a directory."
    ),
    input_schema=LIST_DIR_INPUT_SCHEMA,
    function=list_dir,
)


def write_file(input_data: dict[str, Any]) -> str:
    """파일 전체 내용을 새로 쓰거나 기존 파일을 덮어씁니다."""
    path = input_data.get("path")
    content = input_data.get("content")

    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string.")
    if not isinstance(content, str):
        raise ValueError("Content must be a string.")

    file_path = resolve_workspace_path(path)
    # 새 파일이 하위 폴더에 있어도 바로 만들 수 있도록 부모 디렉터리를 생성합니다.
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding=DEFAULT_ENCODING)
    return f"Wrote {path}"


# write_file은 "전체 파일 내용"을 요구합니다. 부분 수정에는 edit_file을 쓰도록
# 설명을 분리해 두는 것이 모델의 도구 선택 정확도에 도움이 됩니다.
WRITE_FILE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The relative path of the file to create or overwrite.",
        },
        "content": {
            "type": "string",
            "description": "The complete file content to write.",
        },
    },
    "required": ["path", "content"],
}

WRITE_FILE_DEFINITION = ToolDefinition(
    name="write_file",
    description=(
        "Create or overwrite a text file with complete content. "
        "Use this when the user asks you to create a new file."
    ),
    input_schema=WRITE_FILE_INPUT_SCHEMA,
    function=write_file,
)


def edit_file(input_data: dict[str, Any]) -> str:
    """기존 파일에서 정확히 한 번 등장하는 문자열만 교체합니다."""
    path = input_data.get("path")
    old_str = input_data.get("old_str")
    new_str = input_data.get("new_str")

    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string.")
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        raise ValueError("old_str and new_str must be strings.")
    if old_str == "":
        raise ValueError("old_str must be a non-empty exact string.")
    if old_str == new_str:
        raise ValueError("old_str and new_str must be different.")

    file_path = resolve_workspace_path(path)
    content = file_path.read_text(encoding=DEFAULT_ENCODING)
    match_count = content.count(old_str)

    # old_str이 0번 또는 여러 번 나오면 수정하지 않습니다. 이 제한 덕분에
    # 모델이 의도하지 않은 위치까지 한꺼번에 바꾸는 실수를 줄일 수 있습니다.
    if match_count == 0:
        raise ValueError("old_str not found in file.")
    if match_count > 1:
        raise ValueError("old_str must match exactly one location in the file.")

    file_path.write_text(content.replace(old_str, new_str), encoding=DEFAULT_ENCODING)
    return f"Edited {path}"


# edit_file은 작은 패치에 적합한 도구입니다. old_str이 정확히 한 번만 나와야
# 하므로, 모델은 충분히 긴 주변 문맥을 포함한 old_str을 선택해야 합니다.
EDIT_FILE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The relative path of a file in the working directory.",
        },
        "old_str": {
            "type": "string",
            "description": "Exact text to replace. It must occur exactly once.",
        },
        "new_str": {
            "type": "string",
            "description": "Text to replace old_str with.",
        },
    },
    "required": ["path", "old_str", "new_str"],
}

EDIT_FILE_DEFINITION = ToolDefinition(
    name="edit_file",
    description=(
        "Make an exact edit to a text file. "
        "Replace old_str with new_str when old_str appears exactly once."
    ),
    input_schema=EDIT_FILE_INPUT_SCHEMA,
    function=edit_file,
)


def run_node_file(input_data: dict[str, Any]) -> str:
    """JavaScript 파일을 Node.js로 실행하고 결과를 JSON 문자열로 반환합니다."""
    path = input_data.get("path")
    args = input_data.get("args", [])

    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string.")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("args must be a list of strings.")

    file_path = resolve_workspace_path(path)
    # capture_output=True로 stdout/stderr를 모두 수집해 모델에게 돌려줍니다.
    # check=False는 실패도 예외로 바로 던지지 않고 returncode로 확인하기 위함입니다.
    result = subprocess.run(
        ["node", str(file_path), *args],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    # 실행 결과를 구조화하면 모델이 stdout, stderr, exit code를 안정적으로 구분합니다.
    output = {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    formatted_output = json.dumps(output, ensure_ascii=False, indent=2)

    if result.returncode != 0:
        # 실패도 모델에게 전달되어야 하므로 stderr/exit_code를 포함한 JSON을 예외에 담습니다.
        raise RuntimeError(formatted_output)

    return formatted_output


# JavaScript 검증 도구의 입력 스키마입니다. args는 선택적인 문자열 배열입니다.
RUN_NODE_FILE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The relative path of the JavaScript file to run with Node.js.",
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional command-line arguments to pass to the JavaScript file.",
        },
    },
    "required": ["path"],
}

RUN_NODE_FILE_DEFINITION = ToolDefinition(
    name="run_node_file",
    description=(
        "Run a JavaScript file with Node.js and return exit code, stdout, and stderr. "
        "Use this after creating or editing JavaScript files."
    ),
    input_schema=RUN_NODE_FILE_INPUT_SCHEMA,
    function=run_node_file,
)


class Agent:
    """코드 편집 도구를 모델에게 제공하고 tool_call 루프를 관리합니다."""

    def __init__(
        self,
        chat_model: ChatModel,
        get_user_message: Callable[[], tuple[str, bool]],
        tools: list[ToolDefinition],
    ) -> None:
        # 등록된 tools만 모델이 볼 수 있고, execute_tool로 실행할 수 있습니다.
        self.chat_model = chat_model
        self.get_user_message = get_user_message
        self.tools = tools

    def run(self) -> None:
        """시스템 프롬프트, 사용자 메시지, 도구 결과를 이어가며 대화합니다."""
        conversation: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
        ]
        # True면 새 사용자 입력을 기다리고, False면 tool 결과를 추가한 직후라서
        # 사용자 입력 없이 모델을 다시 호출합니다.
        read_user_input = True

        print_chat_banner()

        while True:
            if read_user_input:
                print_user_prompt()

                user_input, is_input_valid = self.get_user_message()
                if not is_input_valid:
                    break

                append_user_message(conversation, user_input)

            response = self.run_inference(conversation)

            if not response.choices:
                print_no_response()
                continue

            model_message = response.choices[0].message
            conversation.append(model_message.model_dump(exclude_none=True))

            if model_message.content:
                print_llm_message(model_message.content)

            if not model_message.tool_calls:
                read_user_input = True
                continue

            # tool_call이 있으면 로컬 도구를 실행하고 결과를 conversation에 추가합니다.
            self._execute_and_append_tool_calls(conversation, model_message.tool_calls)
            read_user_input = False

    def _execute_and_append_tool_calls(
        self,
        conversation: list[ChatCompletionMessageParam],
        tool_calls: list[Any],
    ) -> None:
        """모델이 요청한 tool_call들을 실행하고 결과를 tool 메시지로 추가합니다."""
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = parse_tool_arguments(tool_call)

            tool_response, is_error = run_tool(
                tools=self.tools,
                name=tool_name,
                input_data=tool_args,
            )

            # OpenAI tool-calling에서는 tool_call_id로 요청과 응답을 연결합니다.
            append_tool_result_message(
                conversation=conversation,
                tool_call=tool_call,
                tool_name=tool_name,
                tool_response=tool_response,
                is_error=is_error,
            )

    def run_inference(
        self,
        conversation: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        """현재 대화 기록과 사용 가능한 도구 목록을 모델에게 전달합니다."""
        return self.chat_model.complete(
            messages=conversation,
            tools=to_openai_tools(self.tools),
        )


def new_agent(
    chat_model: ChatModel,
    get_user_msg_fn: Callable[[], tuple[str, bool]],
    tools: list[ToolDefinition],
) -> Agent:
    """Agent 생성을 한 곳에 모아 의존성 주입 구조를 유지합니다."""
    return Agent(
        chat_model=chat_model,
        get_user_message=get_user_msg_fn,
        tools=tools,
    )


def main() -> None:
    """LLM provider 생성, 도구 등록, Agent 실행을 담당합니다."""
    chat_model = new_chat_model()

    # 최종 단계에서는 코드 편집에 필요한 읽기/쓰기/수정/실행 도구를 모두 제공합니다.
    tools: list[ToolDefinition] = [
        READ_FILE_DEFINITION,
        LIST_DIR_DEFINITION,
        WRITE_FILE_DEFINITION,
        EDIT_FILE_DEFINITION,
        RUN_NODE_FILE_DEFINITION,
    ]
    agent = new_agent(
        chat_model=chat_model,
        get_user_msg_fn=get_user_message,
        tools=tools,
    )

    try:
        agent.run()
    except Exception as error:
        print_error(error)


if __name__ == "__main__":
    main()
