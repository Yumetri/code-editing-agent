"""1단계: 가장 단순한 LLM 채팅 에이전트 예제.

이 파일은 사용자의 입력을 대화 기록에 쌓고, 그 전체 기록을 LLM API에
보내 응답을 받는 최소 구조를 보여줍니다. 아직 파일을 읽거나 코드를 실행하는
"도구(tool)" 기능은 없고, 순수하게 채팅만 담당합니다.
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
from agent_lib.core import append_user_message
from agent_lib.llm import ChatModel, new_chat_model

# LLM provider 생성과 fallback 설정은 agent_lib/llm.py에 모아 둡니다.


class Agent:
    """대화 상태를 관리하고 OpenAI 호환 Chat Completions API를 호출합니다."""

    def __init__(
        self,
        chat_model: ChatModel,
        get_user_message: Callable[[], tuple[str, bool]],
    ) -> None:
        # chat_model은 실제 LLM API 호출을 담당하고, get_user_message는 입력 방식을
        # 주입받습니다. 이렇게 분리하면 나중에 테스트나 자동 입력으로 바꾸기 쉽습니다.
        self.chat_model = chat_model
        self.get_user_message = get_user_message
    
    def run(self) -> None:
        """사용자가 종료할 때까지 입력과 LLM 응답을 반복합니다."""
        conversation: list[ChatCompletionMessageParam] = []

        print_chat_banner()
        
        while True:
            print_user_prompt()

            user_input, is_input_valid = self.get_user_message()
            if not is_input_valid:
                break

            # Chat Completions API는 이전 메시지들을 함께 보내야 문맥을 기억합니다.
            append_user_message(conversation, user_input)

            response = self.run_inference(conversation)

            if not response.choices:
                print_no_response()
                continue
            
            model_message = response.choices[0].message
            # 모델 응답도 conversation에 저장해야 다음 요청에서 이전 답변을
            # 기억한 상태로 이어서 대화할 수 있습니다.
            conversation.append(model_message.model_dump(exclude_none=True))

            if model_message.content:
                print_llm_message(model_message.content)

    def run_inference(
        self,
        conversation: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        """현재까지의 대화 기록을 모델에 보내고 응답 객체를 반환합니다."""
        return self.chat_model.complete(messages=conversation)


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
