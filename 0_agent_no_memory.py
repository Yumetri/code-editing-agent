"""0단계: conversation 없이 매 요청을 독립적으로 보내는 LLM 예제.

이 파일은 LLM API가 이전 요청을 자동으로 기억하지 않는다는 점을 보여줍니다.
사용자가 첫 메시지에서 이름을 알려 줘도, 두 번째 요청에는 그 메시지를 다시
보내지 않으므로 모델은 이름을 알 수 없습니다.
"""

from __future__ import annotations

from collections.abc import Callable

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
from agent_lib.llm import ChatModel, new_chat_model

# LLM provider 생성과 fallback 설정은 agent_lib/llm.py에 모아 둡니다.


class Agent:
    """사용자 입력 하나를 독립적인 LLM 요청 하나로 보내는 에이전트입니다."""

    def __init__(
        self,
        chat_model: ChatModel,
        get_user_message: Callable[[], tuple[str, bool]],
    ) -> None:
        self.chat_model = chat_model
        self.get_user_message = get_user_message

    def run(self) -> None:
        """사용자가 종료할 때까지 독립적인 LLM 요청을 반복합니다."""
        print_chat_banner()

        while True:
            print_user_prompt()

            user_input, is_input_valid = self.get_user_message()
            if not is_input_valid:
                break

            request_messages: list[ChatCompletionMessageParam] = [{
                "role": "user",
                "content": user_input,
            }]
            response = self.run_inference(request_messages)

            if not response.choices:
                print_no_response()
                continue

            model_message = response.choices[0].message
            if model_message.content:
                print_llm_message(model_message.content)

    def run_inference(
        self,
        messages: list[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        """이번 요청 하나에 포함할 메시지만 모델에 보내고 응답 객체를 반환합니다."""
        return self.chat_model.complete(messages=messages)


def new_agent(
    chat_model: ChatModel,
    get_user_msg_fn: Callable[[], tuple[str, bool]],
) -> Agent:
    """Agent 생성 로직을 함수로 분리해 main을 읽기 쉽게 유지합니다."""
    return Agent(
        chat_model=chat_model,
        get_user_message=get_user_msg_fn,
    )


def main() -> None:
    """LLM provider를 준비하고 Agent를 실행하는 진입점입니다."""
    chat_model = new_chat_model()

    agent = new_agent(
        chat_model=chat_model,
        get_user_msg_fn=get_user_message,
    )

    try:
        agent.run()
    except Exception as error:
        print_error(error)


if __name__ == "__main__":
    main()

