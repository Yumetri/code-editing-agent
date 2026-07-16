"""4단계: 파일 목록 조회와 파일 생성/수정 도구를 추가한 에이전트.

3단계의 tool feedback loop 위에 `list_dir`, `edit_file`을 더해 에이전트가
작업 디렉터리의 파일을 살펴보고 텍스트 파일을 만들거나 수정할 수 있게 합니다.
아직 작업공간 밖 경로 접근을 막는 보호 장치는 없으며, 그런 보완은 5단계에서
추가됩니다.
"""

from __future__ import annotations

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

# LLM provider 생성과 fallback 설정은 agent_lib/llm.py에 모아 두고, 여기에는 파일 입출력
# 설정만 남깁니다.
DEFAULT_ENCODING = "utf-8"


def read_file(input_data: dict[str, Any]) -> str:
    """지정된 파일을 읽어 모델에게 돌려줄 문자열로 반환합니다."""
    path = input_data.get("path")

    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string.")
    
    with open(path, "r", encoding=DEFAULT_ENCODING) as file:
        return file.read()


# 모델이 read_file 도구를 사용할 때 만들어야 할 입력 구조입니다.
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
    """지정된 디렉터리의 파일/폴더 이름을 줄바꿈 문자열로 반환합니다."""
    path = input_data.get("path", ".")

    if not isinstance(path, str):
        raise ValueError("Path must be a string.")
    
    # Path.iterdir는 파일 이름만 쉽게 꺼낼 수 있습니다. 여기서는 간단한 예제이므로
    # 파일인지 디렉터리인지 같은 추가 메타데이터는 붙이지 않습니다.
    entries = Path(path).iterdir()
    return "\n".join(sorted(entry.name for entry in entries))


# path는 선택값입니다. 모델이 path를 생략하면 현재 디렉터리(".")를 조회합니다.
LIST_DIR_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The relative path of a directory in the working directory. Defaults to '.' if not specified.",
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


def create_new_file(file_path: str, content: str) -> str:
    """없는 파일을 만들 때 사용하는 내부 헬퍼 함수입니다."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=DEFAULT_ENCODING)
    return f"Successfully created file {file_path}"


def edit_file(input_data: dict[str, Any]) -> str:
    """파일에서 old_str을 찾아 new_str로 교체합니다."""
    path = input_data.get("path")
    old_str = input_data.get("old_str")
    new_str = input_data.get("new_str")

    if not isinstance(path, str) or not path:
        raise ValueError("Invalid input parameters: path must be a non-empty string.")
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        raise ValueError("Invalid input parameters: old_str and new_str must be strings.")
    if old_str == new_str:
        raise ValueError("Invalid input parameters: old_str and new_str must be different.")

    try:
        content = Path(path).read_text(encoding=DEFAULT_ENCODING)
    except FileNotFoundError:
        if old_str == "":
            # old_str이 빈 문자열이고 파일이 없으면 "수정" 대신 새 파일 생성으로 봅니다.
            return create_new_file(path, new_str)
        raise

    # 주의: 이 단계의 구현은 str.replace를 그대로 사용하므로 old_str이 여러 번
    # 나오면 모두 교체됩니다. 5단계에서는 정확히 한 번만 매칭되도록 보완합니다.
    new_content = content.replace(old_str, new_str)
    
    if content == new_content and old_str != "":
        raise ValueError("old_str not found in file")

    Path(path).write_text(new_content, encoding=DEFAULT_ENCODING)

    return "OK"


# edit_file은 기존 파일 수정과 새 파일 생성을 모두 담당하므로
# old_str/new_str의 의미를 모델 설명에 자세히 적어 둡니다.
EDIT_FILE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "The relative path of a file in the working directory.",
        },
        "old_str": {
            "type": "string",
            "description": "Text to search for - must match exactly and must only have one match exactly.",
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
        "Make edits to a text file.\n\n"
        "Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.\n\n"
        "If the file specified with path doesn't exist, it will be created."
    ),
    input_schema=EDIT_FILE_INPUT_SCHEMA,
    function=edit_file,
)


class Agent:
    """여러 파일 도구를 제공하는 대화형 코드 수정 에이전트입니다."""

    def __init__(
        self,
        chat_model: ChatModel,
        get_user_message: Callable[[], tuple[str, bool]],
        tools: list[ToolDefinition],
    ) -> None:
        # 이 리스트가 커질수록 모델이 사용할 수 있는 행동도 늘어납니다.
        self.chat_model = chat_model
        self.get_user_message = get_user_message
        self.tools = tools
    
    def run(self) -> None:
        """사용자 입력, 모델 응답, 도구 실행 결과 전달을 반복합니다."""
        conversation: list[ChatCompletionMessageParam] = []
        # tool_call 직후에는 새 사용자 입력을 받지 않고, tool 결과를 포함한
        # conversation으로 모델을 다시 호출해야 합니다.
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

            # 모델이 요청한 도구를 실행하고 결과를 conversation에 붙입니다.
            self._execute_and_append_tool_calls(conversation, model_message.tool_calls)
            read_user_input = False

    def _execute_and_append_tool_calls(
        self,
        conversation: list[ChatCompletionMessageParam],
        tool_calls: list[Any],
    ) -> None:
        """tool_call을 실제 함수 실행으로 바꾸고 결과를 tool 메시지로 기록합니다."""
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = parse_tool_arguments(tool_call)

            tool_response, is_error = run_tool(
                tools=self.tools,
                name=tool_name,
                input_data=tool_args,
            )
            
            # tool_call_id는 모델의 요청과 우리가 돌려주는 결과를 연결하는 식별자입니다.
            append_tool_result_message(
                conversation=conversation,
                tool_call=tool_call,
                tool_name=tool_name,
                tool_response=tool_response,
                is_error=is_error,
            )

    def run_inference(
        self,
        conversation: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        """대화 기록과 도구 목록을 모델에게 보내 다음 응답을 받습니다."""
        return self.chat_model.complete(
            messages=conversation,
            tools=to_openai_tools(self.tools),
        )


def new_agent(
    chat_model: ChatModel,
    get_user_msg_fn: Callable[[], tuple[str, bool]],
    tools: list[ToolDefinition],
) -> Agent:
    """main 함수에서 Agent 생성 세부사항을 숨기는 팩토리 함수입니다."""
    return Agent(
        chat_model=chat_model,
        get_user_message=get_user_msg_fn,
        tools=tools
    )


def main() -> None:
    """LLM provider 생성, 도구 등록, Agent 실행을 담당합니다."""
    chat_model = new_chat_model()
    
    # 4단계에서는 읽기, 목록 보기, 파일 생성/수정 도구를 함께 제공합니다.
    tools: list[ToolDefinition] = [
        READ_FILE_DEFINITION,
        LIST_DIR_DEFINITION,
        EDIT_FILE_DEFINITION,
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
