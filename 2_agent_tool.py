"""2단계: LLM에게 파일 읽기 도구를 제공하는 예제.

1단계와 비교하면 모델에게 `tools` 목록을 전달하고, 모델이 요청한 tool_call을
파싱해서 실제 파이썬 함수(`read_file`)로 실행하는 흐름이 추가됩니다.
다만 이 단계에서는 툴 실행 결과를 다시 LLM에게 보내지 않으므로, 모델은
툴 결과를 바탕으로 최종 답변을 이어서 만들지는 못합니다.
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
        chat_model: ChatModel,
        get_user_message: Callable[[], tuple[str, bool]],
        tools: list[ToolDefinition],
    ) -> None:
        # tools는 이 에이전트가 모델에게 공개할 수 있는 함수 목록입니다.
        self.chat_model = chat_model
        self.get_user_message = get_user_message
        self.tools = tools
    
    def run(self) -> None:
        """사용자 입력, 모델 응답, 단발성 tool_call 실행을 반복합니다."""
        conversation: list[ChatCompletionMessageParam] = []

        print_chat_banner()
        
        while True:
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

            if model_message.tool_calls:
                for tool_call in model_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = parse_tool_arguments(tool_call)

                    # 이 단계의 핵심 한계:
                    # 툴을 실행하고 결과만 화면에 출력합니다. 아직 conversation에
                    # tool 메시지를 추가하지 않으므로 LLM은 실행 결과를 모릅니다.
                    run_tool(tools=self.tools, name=tool_name, input_data=tool_args)

    def run_inference(
        self,
        conversation: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        """대화 기록과 사용 가능한 tools를 함께 모델에게 전달합니다."""
        return self.chat_model.complete(
            messages=conversation,
            tools=to_openai_tools(self.tools),
        )


def new_agent(
    chat_model: ChatModel,
    get_user_msg_fn: Callable[[], tuple[str, bool]],
    tools: list[ToolDefinition],
) -> Agent:
    """Agent 생성을 한 곳에 모아 main 함수의 역할을 단순하게 유지합니다."""
    return Agent(
        chat_model=chat_model,
        get_user_message=get_user_msg_fn,
        tools=tools
    )


def main() -> None:
    """LLM provider와 도구 목록을 준비한 뒤 에이전트를 실행합니다."""
    chat_model = new_chat_model()
    
    # 이 단계에서는 read_file 하나만 모델에게 공개합니다.
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
