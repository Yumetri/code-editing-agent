"""1단계: 가장 단순한 LLM 채팅 에이전트 예제.

이 파일은 사용자의 입력을 대화 기록에 쌓고, 그 전체 기록을 LLM API에
보내 응답을 받는 최소 구조를 보여줍니다. 아직 파일을 읽거나 코드를 실행하는
"도구(tool)" 기능은 없고, 순수하게 채팅만 담당합니다.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)

# .env 파일에 적어둔 OPENROUTER_API_KEY 같은 값을 os.getenv로 읽을 수 있게 합니다.
load_dotenv()

# LLM 호출에 필요한 설정값입니다. 모델이나 OpenRouter 주소를 바꾸고 싶다면
# 이 상수들을 먼저 보면 됩니다.
LLM_API_KEY_NAME = "OPENROUTER_API_KEY"
MODEL_NAME = "poolside/laguna-m.1:free"  # OpenRouter free model
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 터미널 출력 색상용 ANSI escape code입니다. 로직에는 영향을 주지 않고
# 사람 눈에 user/LLM 응답을 구분하기 쉽게 만드는 용도입니다.
COLOR_USER = "\033[94m"     # Blue
COLOR_LLM = "\033[93m"      # Yellow
COLOR_RESET = "\033[0m"


class Agent:
    """대화 상태를 관리하고 OpenAI 호환 Chat Completions API를 호출합니다."""

    def __init__(
        self,
        client: OpenAI,
        get_user_message: Callable[[], tuple[str, bool]],
    ) -> None:
        # client는 실제 LLM API 호출을 담당하고, get_user_message는 입력 방식을
        # 주입받습니다. 이렇게 분리하면 나중에 테스트나 자동 입력으로 바꾸기 쉽습니다.
        self.client = client
        self.get_user_message = get_user_message
    
    def run(self) -> None:
        """사용자가 종료할 때까지 입력과 LLM 응답을 반복합니다."""
        conversation: list[ChatCompletionMessageParam] = []

        print("Chat with OpenRouter. use ctrl-c to quit.")
        
        while True:
            print(f"{COLOR_USER}You{COLOR_RESET}: ", end="")

            user_input, is_input_valid = self.get_user_message()
            if not is_input_valid:
                break

            # Chat Completions API는 이전 메시지들을 함께 보내야 문맥을 기억합니다.
            self._add_user_message(conversation, user_input)

            response = self.run_inference(conversation)

            if not response.choices:
                print("No response from OpenRouter.")
                continue
            
            model_message = response.choices[0].message
            # 모델 응답도 conversation에 저장해야 다음 요청에서 이전 답변을
            # 기억한 상태로 이어서 대화할 수 있습니다.
            conversation.append(model_message.model_dump(exclude_none=True))

            if model_message.content:
                print(f"{COLOR_LLM}LLM{COLOR_RESET}: {model_message.content}")

    def _add_user_message(
        self,
        conversation: list[ChatCompletionMessageParam],
        user_input: str,
    ) -> None:
        """사용자 입력을 OpenAI 메시지 형식으로 변환해 대화 기록에 추가합니다."""
        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": user_input,
        }
        conversation.append(user_message)

    def run_inference(
        self,
        conversation: list[ChatCompletionMessageParam]
    ) -> ChatCompletion:
        """현재까지의 대화 기록을 모델에 보내고 응답 객체를 반환합니다."""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
        )
        return response


def get_user_message() -> tuple[str, bool]:
    """터미널에서 한 줄을 읽고, 종료 입력이면 False를 함께 반환합니다."""
    try:
        text = input()
        return text, True
    except (EOFError, KeyboardInterrupt):
        return "", False


def new_agent(
    client: OpenAI,
    get_user_msg_fn: Callable[[], tuple[str, bool]],
) -> Agent:
    """Agent 생성 로직을 함수로 분리해 main을 읽기 쉽게 유지합니다."""
    return Agent(
        client=client,
        get_user_message=get_user_msg_fn,
    )


def main() -> None:
    """환경 변수를 확인하고 Agent를 실행하는 진입점입니다."""
    if not os.getenv(LLM_API_KEY_NAME):
        raise RuntimeError(f"Missing {LLM_API_KEY_NAME}.")
    
    # OpenRouter는 OpenAI 호환 API를 제공하므로 OpenAI 클라이언트에
    # base_url만 OpenRouter 주소로 지정하면 같은 방식으로 호출할 수 있습니다.
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv(LLM_API_KEY_NAME),
    )
    
    agent = new_agent(
        client=client,
        get_user_msg_fn=get_user_message,
    )

    try:
        agent.run()
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
