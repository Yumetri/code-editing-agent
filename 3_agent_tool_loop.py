"""3단계: tool_call 결과를 다시 LLM에게 전달하는 에이전트 예제.

2단계에서는 도구를 실행만 하고 결과를 모델에게 돌려주지 않았습니다.
이 파일은 `role="tool"` 메시지를 conversation에 추가한 뒤 모델을 다시 호출해,
LLM이 파일 내용 같은 도구 실행 결과를 바탕으로 최종 답변을 만들 수 있게 합니다.
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

# .env 파일의 OPENROUTER_API_KEY를 환경 변수로 불러옵니다.
load_dotenv()

# LLM 호출과 파일 읽기에 필요한 설정입니다.
LLM_API_KEY_NAME = "OPENROUTER_API_KEY"
MODEL_NAME = "poolside/laguna-m.1:free"  # OpenRouter free model
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ENCODING = "utf-8"

# 터미널에서 user/LLM/tool 로그를 색상으로 구분하기 위한 값입니다.
COLOR_USER = "\033[94m"     # Blue
COLOR_LLM = "\033[93m"      # Yellow
COLOR_TOOL = "\033[92m"     # Green
COLOR_RESET = "\033[0m"


@dataclass
class ToolDefinition:
    """모델에게 공개할 도구 설명과 실제 파이썬 실행 함수를 함께 보관합니다."""

    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[[dict[str, Any]], str]


def read_file(input_data: dict[str, Any]) -> str:
    """상대 경로로 지정된 파일을 읽어 문자열로 반환합니다."""
    path = input_data.get("path")

    if not isinstance(path, str) or not path:
        raise ValueError("Path must be a non-empty string.")
    
    with open(path, "r", encoding=DEFAULT_ENCODING) as file:
        return file.read()


# 모델이 read_file을 호출할 때 필요한 인자 모양을 JSON Schema로 알려줍니다.
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

# 도구 이름, 설명, 입력 스키마, 실제 실행 함수를 하나로 등록합니다.
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
    """도구 실행 결과를 다시 모델에게 전달하는 대화형 에이전트입니다."""

    def __init__(
        self,
        client: OpenAI,
        get_user_message: Callable[[], tuple[str, bool]],
        tools: list[ToolDefinition],
    ) -> None:
        # tools에 들어간 도구만 모델에게 공개되고 실제로 실행할 수 있습니다.
        self.client = client
        self.get_user_message = get_user_message
        self.tools = tools
    
    def run(self) -> None:
        """사용자 입력과 LLM/tool 상호작용을 하나의 대화 기록으로 이어갑니다."""
        conversation: list[ChatCompletionMessageParam] = []
        # True면 새 사용자 입력을 받고, False면 방금 추가한 tool 결과를 바탕으로
        # 사용자 입력 없이 LLM을 다시 호출합니다.
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

            # 모델이 tool_call을 요청한 경우, 로컬에서 도구를 실행하고 그 결과를
            # conversation에 넣은 뒤 다음 loop에서 다시 LLM을 호출합니다.
            self._execute_and_append_tool_calls(conversation, model_message.tool_calls)
            read_user_input = False

    def _add_user_message(
        self,
        conversation: list[ChatCompletionMessageParam],
        user_input: str,
    ) -> None:
        """사용자 입력을 Chat Completions 메시지 형식으로 추가합니다."""
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
        """모델이 요청한 모든 tool_call을 실행하고 tool 메시지로 기록합니다."""
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            
            # tool_call.function.arguments는 JSON 문자열이므로 파싱이 필요합니다.
            # 모델이 잘못된 JSON을 만들 수도 있으므로 실패 시 빈 dict로 처리합니다.
            try:
                tool_args = json.loads(tool_args_str) if tool_args_str else {}
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            tool_response, is_error = self.execute_tool(
                name=tool_name,
                input_data=tool_args,
            )
            
            # OpenAI tool-calling 프로토콜에서는 tool_call_id를 반드시 다시 넣어야
            # 어떤 tool_call에 대한 응답인지 모델이 연결할 수 있습니다.
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
        """ToolDefinition 목록을 API 요청의 tools 필드로 변환합니다."""
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
        """등록된 도구를 이름으로 찾아 실행합니다."""
        tool_def: ToolDefinition | None = None

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
        """현재 conversation과 tools를 모델에게 보내 다음 메시지를 받습니다."""
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
    """터미널 입력을 읽고, Ctrl-C/Ctrl-D는 정상 종료 신호로 다룹니다."""
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
    """Agent 객체 생성을 감싸는 작은 팩토리 함수입니다."""
    return Agent(
        client=client,
        get_user_message=get_user_msg_fn,
        tools=tools
    )


def main() -> None:
    """환경 변수 검증, API 클라이언트 생성, 도구 등록, Agent 실행을 담당합니다."""
    if not os.getenv(LLM_API_KEY_NAME):
        raise RuntimeError(f"Missing {LLM_API_KEY_NAME}.")
    
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv(LLM_API_KEY_NAME),
    )
    
    # 이 단계의 도구는 파일 읽기 하나입니다. 다음 단계에서 도구가 늘어납니다.
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
