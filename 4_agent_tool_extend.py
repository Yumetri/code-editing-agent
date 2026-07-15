"""4단계: 파일 목록 조회와 파일 생성/수정 도구를 추가한 에이전트.

3단계의 tool feedback loop 위에 `list_dir`, `edit_file`을 더해 에이전트가
작업 디렉터리의 파일을 살펴보고 텍스트 파일을 만들거나 수정할 수 있게 합니다.
아직 작업공간 밖 경로 접근을 막는 보호 장치는 없으며, 그런 보완은 5단계에서
추가됩니다.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

# .env 파일에서 OPENROUTER_API_KEY를 읽어옵니다.
load_dotenv()

# LLM 호출과 파일 입출력에 필요한 공통 설정입니다.
LLM_API_KEY_NAME = "OPENROUTER_API_KEY"
MODEL_NAME = "poolside/laguna-m.1:free"  # OpenRouter free model
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ENCODING = "utf-8"

# 터미널 출력에서 사용자, 모델, 도구 실행 로그를 구분하기 위한 색상입니다.
COLOR_USER = "\033[94m"     # Blue
COLOR_LLM = "\033[93m"      # Yellow
COLOR_TOOL = "\033[92m"     # Green
COLOR_RESET = "\033[0m"


@dataclass
class ToolDefinition:
    """LLM에게 보여줄 도구 정보와 실제 실행 함수를 연결합니다."""

    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[[dict[str, Any]], str]


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
    
    # os.listdir은 파일 이름만 반환합니다. 여기서는 간단한 예제이므로
    # 파일인지 디렉터리인지 같은 추가 메타데이터는 붙이지 않습니다.
    files = os.listdir(path)
    return "\n".join(files)


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
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        # 하위 폴더까지 한 번에 만들 수 있도록 exist_ok=True를 사용합니다.
        os.makedirs(directory, exist_ok=True)
    with open(file_path, "w", encoding=DEFAULT_ENCODING) as file:
        file.write(content)
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
        with open(path, "r", encoding=DEFAULT_ENCODING) as file:
            content = file.read()
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

    with open(path, "w", encoding=DEFAULT_ENCODING) as file:
        file.write(new_content)

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
        client: OpenAI,
        get_user_message: Callable[[], tuple[str, bool]],
        tools: list[ToolDefinition],
    ) -> None:
        # 이 리스트가 커질수록 모델이 사용할 수 있는 행동도 늘어납니다.
        self.client = client
        self.get_user_message = get_user_message
        self.tools = tools
    
    def run(self) -> None:
        """사용자 입력, 모델 응답, 도구 실행 결과 전달을 반복합니다."""
        conversation: list[ChatCompletionMessageParam] = []
        # tool_call 직후에는 새 사용자 입력을 받지 않고, tool 결과를 포함한
        # conversation으로 모델을 다시 호출해야 합니다.
        read_user_input = True

        print("Chat with OpenRouter. use ctrl-c to quit.")
        
        while True:
            if read_user_input:
                print(f"{COLOR_USER}You{COLOR_RESET}: ", end="")

                user_input, is_input_valid = self.get_user_message()
                if not is_input_valid:
                    break

                self._add_user_message(conversation, user_input)

            response = self.run_inference(conversation)

            if not response.choices:
                print("No response from OpenRouter.")
                continue
            
            model_message = response.choices[0].message
            conversation.append(model_message.model_dump(exclude_none=True))

            if model_message.content:
                print(f"{COLOR_LLM}LLM{COLOR_RESET}: {model_message.content}")

            if not model_message.tool_calls:
                read_user_input = True
                continue

            # 모델이 요청한 도구를 실행하고 결과를 conversation에 붙입니다.
            self._execute_and_append_tool_calls(conversation, model_message.tool_calls)
            read_user_input = False

    def _add_user_message(
        self,
        conversation: list[ChatCompletionMessageParam],
        user_input: str,
    ) -> None:
        """사용자 메시지를 Chat Completions API 형식으로 추가합니다."""
        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": user_input,
        }
        conversation.append(user_message)

    def _execute_and_append_tool_calls(
        self,
        conversation: list[ChatCompletionMessageParam],
        tool_calls: list[Any],
    ) -> None:
        """tool_call을 실제 함수 실행으로 바꾸고 결과를 tool 메시지로 기록합니다."""
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            
            # 모델이 준 arguments는 JSON 문자열입니다. 잘못된 JSON이면 빈 입력으로
            # 도구를 실행해 에러 메시지를 conversation에 남기게 합니다.
            try:
                tool_args = json.loads(tool_args_str) if tool_args_str else {}
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            tool_response, is_error = self.execute_tool(
                name=tool_name,
                input_data=tool_args,
            )
            
            # tool_call_id는 모델의 요청과 우리가 돌려주는 결과를 연결하는 식별자입니다.
            tool_message: ChatCompletionMessageParam = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps({
                    "result": tool_response,
                    "is_error": is_error,
                }),
            }
            conversation.append(tool_message)

    def to_openai_tools(self) -> list[ChatCompletionToolParam]:
        """등록된 ToolDefinition들을 OpenAI tools 파라미터 형식으로 변환합니다."""
        if not self.tools:
            return []
        
        openai_tools: list[ChatCompletionToolParam] = []
        for tool in self.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        return openai_tools

    def execute_tool(
        self,
        name: str,
        input_data: dict[str, Any],
    ) -> tuple[str, bool]:
        """도구 이름으로 ToolDefinition을 찾고 실행 결과와 에러 여부를 반환합니다."""
        tool_def: ToolDefinition | None = None

        # 간단한 예제라 선형 탐색을 사용합니다. 도구가 많아지면 dict 매핑으로
        # 바꿔도 됩니다.
        for tool in self.tools:
            if tool.name == name:
                tool_def = tool
                break
        
        if tool_def is None:
            return "tool not found", True
        
        print(f"{COLOR_TOOL}tool{COLOR_RESET}: {name}({input_data})")

        try:
            response = tool_def.function(input_data)
            return response, False
        except Exception as error:
            return str(error), True

    def run_inference(
        self,
        conversation: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        """대화 기록과 도구 목록을 모델에게 보내 다음 응답을 받습니다."""
        openai_tools = self.to_openai_tools()

        kwargs: dict[str, Any] = {
            "model": MODEL_NAME,
            "messages": conversation,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)
        return response


def get_user_message() -> tuple[str, bool]:
    """터미널 한 줄 입력을 읽고, 종료 신호면 False를 반환합니다."""
    try:
        text = input()
        return text, True
    except (EOFError, KeyboardInterrupt):
        return "", False


def new_agent(
    client: OpenAI,
    get_user_msg_fn: Callable[[], tuple[str, bool]],
    tools: list[ToolDefinition],
) -> Agent:
    """main 함수에서 Agent 생성 세부사항을 숨기는 팩토리 함수입니다."""
    return Agent(
        client=client,
        get_user_message=get_user_msg_fn,
        tools=tools
    )


def main() -> None:
    """환경 설정, 클라이언트 생성, 도구 등록, Agent 실행을 담당합니다."""
    if not os.getenv(LLM_API_KEY_NAME):
        raise RuntimeError(f"Missing {LLM_API_KEY_NAME}.")
    
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv(LLM_API_KEY_NAME),
    )
    
    # 4단계에서는 읽기, 목록 보기, 파일 생성/수정 도구를 함께 제공합니다.
    tools: list[ToolDefinition] = [
        READ_FILE_DEFINITION,
        LIST_DIR_DEFINITION,
        EDIT_FILE_DEFINITION,
    ]
    agent = new_agent(
        client=client,
        get_user_msg_fn=get_user_message,
        tools=tools,    
    )

    try:
        agent.run()
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
