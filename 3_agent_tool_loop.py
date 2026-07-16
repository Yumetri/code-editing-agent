"""3단계: tool_call 결과를 다시 LLM에게 전달하는 에이전트 예제.

2단계에서는 도구를 실행만 하고 결과를 모델에게 돌려주지 않았습니다.
이 파일은 `role="tool"` 메시지를 conversation에 추가한 뒤 모델을 다시 호출해,
LLM이 파일 내용 같은 도구 실행 결과를 바탕으로 최종 답변을 만들 수 있게 합니다.
"""

from __future__ import annotations

from collections.abc import Callable
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

# LLM provider 생성과 fallback 설정은 agent_lib/llm.py에 모아 두고, 여기에는 파일 읽기
# 설정만 남깁니다.
DEFAULT_ENCODING = "utf-8"


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
        chat_model: ChatModel,
        get_user_message: Callable[[], tuple[str, bool]],
        tools: list[ToolDefinition],
    ) -> None:
        # tools에 들어간 도구만 모델에게 공개되고 실제로 실행할 수 있습니다.
        self.chat_model = chat_model
        self.get_user_message = get_user_message
        self.tools = tools
    
    def run(self) -> None:
        """사용자 입력과 LLM/tool 상호작용을 하나의 대화 기록으로 이어갑니다."""
        conversation: list[ChatCompletionMessageParam] = []
        # True면 새 사용자 입력을 받고, False면 방금 추가한 tool 결과를 바탕으로
        # 사용자 입력 없이 LLM을 다시 호출합니다.
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

            # 모델이 tool_call을 요청한 경우, 로컬에서 도구를 실행하고 그 결과를
            # conversation에 넣은 뒤 다음 loop에서 다시 LLM을 호출합니다.
            self._execute_and_append_tool_calls(conversation, model_message.tool_calls)
            read_user_input = False

    def _execute_and_append_tool_calls(
        self,
        conversation: list[ChatCompletionMessageParam],
        tool_calls: list[Any],
    ) -> None:
        """모델이 요청한 모든 tool_call을 실행하고 tool 메시지로 기록합니다."""
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = parse_tool_arguments(tool_call)

            tool_response, is_error = run_tool(
                tools=self.tools,
                name=tool_name,
                input_data=tool_args,
            )
            
            # OpenAI tool-calling 프로토콜에서는 tool_call_id를 반드시 다시 넣어야
            # 어떤 tool_call에 대한 응답인지 모델이 연결할 수 있습니다.
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
        """현재 conversation과 tools를 모델에게 보내 다음 메시지를 받습니다."""
        return self.chat_model.complete(
            messages=conversation,
            tools=to_openai_tools(self.tools),
        )


def new_agent(
    chat_model: ChatModel,
    get_user_msg_fn: Callable[[], tuple[str, bool]],
    tools: list[ToolDefinition],
) -> Agent:
    """Agent 객체 생성을 감싸는 작은 팩토리 함수입니다."""
    return Agent(
        chat_model=chat_model,
        get_user_message=get_user_msg_fn,
        tools=tools
    )


def main() -> None:
    """LLM provider 생성, 도구 등록, Agent 실행을 담당합니다."""
    chat_model = new_chat_model()
    
    # 이 단계의 도구는 파일 읽기 하나입니다. 다음 단계에서 도구가 늘어납니다.
    tools: list[ToolDefinition] = [
        READ_FILE_DEFINITION,
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
