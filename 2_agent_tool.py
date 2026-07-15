"""2단계: LLM에게 파일 읽기 도구를 제공하는 예제.

1단계와 비교하면 모델에게 `tools` 목록을 전달하고, 모델이 요청한 tool_call을
파싱해서 실제 파이썬 함수(`read_file`)로 실행하는 흐름이 추가됩니다.
다만 이 단계에서는 툴 실행 결과를 다시 LLM에게 보내지 않으므로, 모델은
툴 결과를 바탕으로 최종 답변을 이어서 만들지는 못합니다.
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

# .env 파일의 OPENROUTER_API_KEY를 환경 변수로 로드합니다.
load_dotenv()

# LLM 호출과 파일 입출력에 필요한 공통 설정입니다.
LLM_API_KEY_NAME = "OPENROUTER_API_KEY"
MODEL_NAME = "poolside/laguna-m.1:free"  # OpenRouter free model
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ENCODING = "utf-8"

# 터미널 출력 색상입니다. 사용자, LLM, tool 실행 로그를 눈으로 구분합니다.
COLOR_USER = "\033[94m"     # Blue
COLOR_LLM = "\033[93m"      # Yellow
COLOR_TOOL = "\033[92m"     # Green
COLOR_RESET = "\033[0m"


@dataclass
class ToolDefinition:
    """LLM에게 공개할 도구 하나의 메타데이터와 실제 실행 함수를 묶습니다.

    name/description/input_schema는 모델에게 전달되는 설명이고, function은
    모델이 해당 도구를 호출했을 때 로컬에서 실행할 파이썬 함수입니다.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[[dict[str, Any]], str]


def read_file(input_data: dict[str, Any]) -> str:
    """`{"path": "파일경로"}` 입력을 받아 파일 내용을 문자열로 반환합니다."""
    path = input_data.get("path")

    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string.")
    
    with open(path, "r", encoding=DEFAULT_ENCODING) as file:
        return file.read()


# JSON Schema 형식으로 도구 입력값을 설명합니다. 모델은 이 스키마를 보고
# 어떤 인자를 만들어야 하는지 판단합니다.
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

# ToolDefinition은 "모델에게 보여줄 도구 설명"과 "실제로 실행할 함수"를
# 하나로 연결하는 등록 정보입니다.
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


class Agent:
    """대화 루프와 tool_call 실행을 관리하는 간단한 에이전트입니다."""

    def __init__(
        self,
        client: OpenAI,
        get_user_message: Callable[[], tuple[str, bool]],
        tools: list[ToolDefinition],
    ) -> None:
        # tools는 이 에이전트가 모델에게 공개할 수 있는 함수 목록입니다.
        self.client = client
        self.get_user_message = get_user_message
        self.tools = tools
    
    def run(self) -> None:
        """사용자 입력, 모델 응답, 단발성 tool_call 실행을 반복합니다."""
        conversation: list[ChatCompletionMessageParam] = []

        print("Chat with OpenRouter. use ctrl-c to quit.")
        
        while True:
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

            if model_message.tool_calls:
                for tool_call in model_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    
                    try:
                        tool_args = json.loads(tool_args_str) if tool_args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {}

                    # 이 단계의 핵심 한계:
                    # 툴을 실행하고 결과만 화면에 출력합니다. 아직 conversation에
                    # tool 메시지를 추가하지 않으므로 LLM은 실행 결과를 모릅니다.
                    self.execute_tool(name=tool_name, input_data=tool_args)

    def _add_user_message(
        self,
        conversation: list[ChatCompletionMessageParam],
        user_input: str,
    ) -> None:
        """사용자 입력을 OpenAI 메시지 형식으로 대화 기록에 추가합니다."""
        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": user_input,
        }
        conversation.append(user_message)

    def to_openai_tools(self) -> list[ChatCompletionToolParam]:
        """내부 ToolDefinition 목록을 OpenAI API가 요구하는 tools 형식으로 바꿉니다."""
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
        """이름으로 도구를 찾아 실행하고, (결과 문자열, 에러 여부)를 반환합니다."""
        tool_def: ToolDefinition | None = None

        # 모델이 요청한 tool 이름과 우리가 등록한 ToolDefinition을 매칭합니다.
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
        """대화 기록과 사용 가능한 tools를 함께 모델에게 전달합니다."""
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
    """터미널에서 사용자 입력을 읽고, 종료 신호면 False를 반환합니다."""
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
    """Agent 생성을 한 곳에 모아 main 함수의 역할을 단순하게 유지합니다."""
    return Agent(
        client=client,
        get_user_message=get_user_msg_fn,
        tools=tools
    )


def main() -> None:
    """API 클라이언트와 도구 목록을 준비한 뒤 에이전트를 실행합니다."""
    if not os.getenv(LLM_API_KEY_NAME):
        raise RuntimeError(f"Missing {LLM_API_KEY_NAME}.")
    
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv(LLM_API_KEY_NAME),
    )
    
    # 이 단계에서는 read_file 하나만 모델에게 공개합니다.
    tools: list[ToolDefinition] = [
        READ_FILE_DEFINITION,
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
