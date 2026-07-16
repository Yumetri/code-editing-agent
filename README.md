# Code Editing Agent 학습 가이드

이 저장소는 LLM 기반 코드 편집 에이전트를 아주 작은 단계부터 확장해 가며
구현하는 튜토리얼입니다.

단계별 핵심 파일은 다음 6개입니다.

0. `0_agent_no_memory.py`: conversation 없이 매 요청을 독립적으로 보내는 예제
1. `1_agent_basic.py`: conversation으로 대화 기억을 추가한 기본 채팅 에이전트
2. `2_agent_tool.py`: LLM tool calling 첫 도입
3. `3_agent_tool_loop.py`: tool 실행 결과를 다시 LLM에게 전달하는 루프
4. `4_agent_tool_extend.py`: 파일 목록 조회, 파일 생성, 파일 수정 도구 확장
5. `5_code_agent.py`: 작업공간 보호와 Node.js 실행 검증을 갖춘 코드 편집 에이전트

공통 지원 코드는 `agent_lib/` 패키지에 모았습니다.

- `agent_lib/llm.py`: OpenRouter/Gemini provider 생성, fallback, logging
- `agent_lib/console.py`: 터미널 입력, 색상 출력, tool 실행 로그 출력
- `agent_lib/core.py`: 사용자 메시지 추가, tool schema 변환, tool 실행, tool 결과 메시지 생성

각 단계의 실험은 먼저 예제 파일을 실행한 뒤, README의 `text` 블록을 프롬프트에
복사해서 붙여넣는 방식으로 진행합니다.

## 준비

이 프로젝트는 `uv`와 OpenRouter 또는 Gemini API 키를 사용합니다.

```bash
uv sync
```

`.env.example`을 `.env`로 복사한 뒤, 사용할 provider의 API 키를 넣습니다.

```bash
cp .env.example .env
```

`.env` 예시:

```bash
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL_NAME=poolside/laguna-m.1:free
GEMINI_API_KEY=...
GEMINI_MODEL_NAME=gemini-2.5-flash
```

모든 예제는 `agent_lib/llm.py`의 `ChatModel`을 주입받아 실행됩니다. 기본값인
`LLM_PROVIDER=openrouter`에서는 OpenRouter API를 우선 사용합니다.
`OPENROUTER_API_KEY`가 없거나 OpenRouter 인증에 실패하면 Gemini API로 fallback을
시도합니다. Gemini를 직접 쓰고 싶다면 `LLM_PROVIDER=gemini`로 설정합니다.

즉, 일반적인 실행 설정은 OpenRouter 키와 Gemini 키를 둘 다 넣어 두고,
OpenRouter를 먼저 쓰다가 인증 문제가 있을 때만 Gemini가 대신 동작하게 두는 것입니다.

모델을 바꾸고 싶다면 `.env`의 `OPENROUTER_MODEL_NAME` 또는 `GEMINI_MODEL_NAME`을
수정하면 됩니다.

### OpenRouter 무료 모델 사용량 안내

OpenRouter의 `:free` 모델은 학습과 테스트에 유용하지만, 무료 사용량은 계정 상태와
일일 한도에 따라 제한됩니다. 무료 사용량 남용을 줄이기 위해 OpenRouter는
`$10` 이상 크레딧 구매 이력이 있는 계정에 더 높은 무료 모델 사용 한도를 제공합니다.

무료 Gemini API를 직접 사용해도 되지만, 무료 사용량이 제한적이라 이 저장소의
예제를 여러 번 테스트하다 보면 금방 한도에 도달할 수 있습니다. 가능하다면
OpenRouter에 `$10` 정도를 충전한 뒤 무료 모델을 충분히 써 보고, 유료 모델도 몇 번
테스트해 보는 편이 LLM API 사용 경험을 쌓는 데 더 좋습니다.

테스트를 실행하려면:

```bash
uv run --group dev pytest
```

테스트는 실제 LLM API를 호출하지 않습니다. `tests/test_agents.py`는 0-5단계의
대화 흐름과 파일 도구를 fake 모델로 검증하고, `tests/test_llm.py`는 provider 선택,
OpenRouter 인증 실패 시 Gemini fallback, logging 정책을 검증합니다.

## 전체 발전 흐름

| 단계 | 파일 | 발전 포인트 | 핵심 한계 |
| --- | --- | --- | --- |
| 0 | `0_agent_no_memory.py` | 현재 입력 하나만 보내 LLM이 요청 사이를 기억하지 못함을 확인함 | 이전 대화를 기억할 수 없음 |
| 1 | `1_agent_basic.py` | conversation을 누적해 이전 사용자/모델 메시지를 함께 보냄 | 파일을 읽거나 실행할 수 없음 |
| 2 | `2_agent_tool.py` | `read_file` 도구를 LLM에게 공개함 | 도구 결과를 LLM에게 다시 전달하지 않음 |
| 3 | `3_agent_tool_loop.py` | `role="tool"` 메시지로 도구 결과를 대화에 추가함 | 읽기 도구만 있어서 수정/실행 불가 |
| 4 | `4_agent_tool_extend.py` | `list_dir`, `edit_file` 추가로 파일 탐색/생성/수정 가능 | 작업공간 밖 경로 접근 보호가 약하고 실행 검증 없음 |
| 5 | `5_code_agent.py` | 경로 보호, `write_file`, 정확한 `edit_file`, `run_node_file` 검증 추가 | Python 실행 도구는 없고 JavaScript 실행만 지원 |

## 0단계: 기억 없는 LLM 호출

실행:

```bash
uv run python 0_agent_no_memory.py
```

제공 기능:

- 터미널에서 사용자 입력을 받습니다.
- 매 입력을 독립적인 LLM 요청 하나로 보냅니다.
- 이전 사용자 입력이나 모델 응답을 다음 요청에 포함하지 않습니다.

이 단계에서 확인할 것:

- LLM API는 이전 요청을 자동으로 기억하지 않습니다.
- 첫 번째 요청에서 이름을 알려 줘도, 두 번째 요청에 그 내용을 다시 보내지 않으면 모델은 이름을 알 수 없습니다.
- 그래서 채팅처럼 보이는 프로그램을 만들려면 conversation을 직접 관리해야 합니다.

### 실험: 이름을 기억하지 못하는지 확인

```text
My name is Jangjun.
```

응답을 확인한 뒤 같은 실행 상태에서 이어서 입력합니다.

```text
What is my name?
```

0단계는 두 번째 요청에 첫 번째 메시지를 함께 보내지 않습니다. 모델이 이름을 맞힌다면
대부분 추측이거나 일반적인 대화 패턴 때문이지, API가 이전 요청을 기억해서가 아닙니다.

## 1단계: conversation으로 대화 기억 추가

실행:

```bash
uv run python 1_agent_basic.py
```

제공 기능:

- 터미널에서 사용자 입력을 받습니다.
- 사용자 입력과 모델 응답을 `conversation` 리스트에 누적합니다.
- 매 요청마다 지금까지의 conversation 전체를 모델에 다시 보냅니다.
- 그래서 모델이 앞에서 말한 이름이나 맥락을 기억하는 것처럼 동작합니다.

한계점:

- 모델은 로컬 파일을 볼 수 없습니다.
- 도구 호출 기능이 없습니다.
- 사용자가 프롬프트로 "함수처럼 답해줘"라고 유도할 수는 있지만, 실제 함수가 실행되지는 않습니다.

이 단계에서 확인할 것:

- 0단계와 달리 두 번째 요청에는 첫 번째 사용자 메시지와 모델 응답이 함께 들어갑니다.
- 기억은 모델 내부에 저장되는 것이 아니라, 우리가 conversation을 다시 보내기 때문에 생깁니다.
- 아직 실제 함수 실행은 없으므로, 프롬프트로 함수 호출처럼 답하게 해도 텍스트일 뿐입니다.

### 실험 1: conversation으로 이름 기억하기

첫 번째 입력:

```text
My name is Jangjun.
```

응답을 확인한 뒤 두 번째 입력:

```text
What is my name?
```

1단계는 첫 번째 사용자 메시지와 모델 응답을 conversation에 저장하고, 두 번째
요청 때 함께 보냅니다. 이 차이 때문에 모델이 이름을 기억할 수 있습니다.

### 실험 2: 프롬프트로만 함수 호출 흉내 내기

아직 실제 도구는 없지만, 프롬프트만으로 모델에게 함수 호출처럼 말하게 할 수
있습니다.

첫 번째 입력:

```text
You are a weather expert. When I ask you about the weather in a given location, reply only with get_weather(<location_name>). Do not explain anything else. Understood?
```

응답을 확인한 뒤 두 번째 입력:

```text
What's the weather in Seoul?
```

이 실험의 목표는 한계를 확인하는 것입니다. 모델이 `get_weather(Seoul)` 같은
텍스트를 만들 수는 있지만, 실제 `get_weather` 함수가 실행되는 것은 아닙니다.
2단계부터는 이 한계를 해결하기 위해 진짜 tool calling을 추가합니다.

## 2단계: 파일 읽기 도구 추가

실행:

```bash
uv run python 2_agent_tool.py
```

제공 기능:

- `ToolDefinition` 데이터 클래스로 도구의 이름, 설명, 입력 스키마, 실행 함수를 묶습니다.
- `read_file` 도구를 모델에게 공개합니다.
- 모델이 `tool_call`을 만들면 파이썬 코드가 해당 도구를 실제로 실행합니다.
- `docs/secret-file.txt` 같은 로컬 파일을 읽을 수 있습니다.

새로 등장한 핵심 개념:

- `ChatCompletionToolParam`
- JSON Schema 기반 tool input
- `tool_call.function.name`
- `tool_call.function.arguments`
- `json.loads(...)`로 tool arguments 파싱

한계점:

- 도구는 실행되지만 실행 결과를 LLM에게 다시 보내지 않습니다.
- 따라서 모델이 파일 내용을 읽은 뒤 그 내용을 바탕으로 최종 답변을 만들 수 없습니다.
- 사용자는 터미널의 `tool:` 로그로 도구가 실행된 사실만 볼 수 있습니다.

### 실험: 파일 읽기 도구 호출 확인

```bash
uv run python 2_agent_tool.py
```

프롬프트에 붙여넣을 문장:

```text
help me solve the riddle in the docs/secret-file.txt file
```

관찰 포인트:

- 모델이 `read_file` 도구 호출을 만들 수 있습니다.
- 프로그램은 실제로 파일을 읽습니다.
- 하지만 이 단계에서는 파일 내용이 다시 모델에게 전달되지 않기 때문에 답변이 완성되지 않거나 어색할 수 있습니다.

## 3단계: Tool Feedback Loop

실행:

```bash
uv run python 3_agent_tool_loop.py
```

제공 기능:

- 2단계의 `read_file` 도구를 그대로 사용합니다.
- 도구 실행 결과를 `role="tool"` 메시지로 `conversation`에 추가합니다.
- 새 사용자 입력을 받기 전에 모델을 한 번 더 호출합니다.
- 모델이 도구 실행 결과를 읽고 최종 답변을 만들 수 있습니다.

2단계보다 발전된 점:

- 2단계는 "도구 실행"까지만 합니다.
- 3단계는 "도구 실행 결과를 모델에게 돌려주기"까지 합니다.
- 이 차이 때문에 파일 기반 질문에 실제로 답할 수 있습니다.

한계점:

- 여전히 도구는 `read_file` 하나뿐입니다.
- 디렉터리 목록을 볼 수 없습니다.
- 파일을 만들거나 수정할 수 없습니다.
- 코드를 실행해서 검증할 수 없습니다.

### 실험 1: 수수께끼 파일 읽고 답하기

```bash
uv run python 3_agent_tool_loop.py
```

프롬프트에 붙여넣을 문장:

```text
help me solve the riddle in the docs/secret-file.txt file
```

`docs/secret-file.txt`의 수수께끼는 다음 내용입니다.

```text
'what animal is the most disagreeable because it always says neigh?'
```

이 단계에서는 모델이 파일을 읽고, 그 결과를 다시 받아 답을 추론할 수 있습니다.

### 실험 2: 파일 내용 읽고 설명하기

```bash
uv run python 3_agent_tool_loop.py
```

프롬프트에 붙여넣을 문장:

```text
What's going on in 3_agent_tool_loop.py? Be brief!
```

관찰 포인트:

- 모델은 먼저 `read_file`로 `3_agent_tool_loop.py`를 읽어야 합니다.
- 그 뒤 tool 결과를 바탕으로 간단한 설명을 생성합니다.

## 4단계: 파일 도구 확장

실행:

```bash
uv run python 4_agent_tool_extend.py
```

제공 기능:

- `read_file`: 파일 내용 읽기
- `list_dir`: 디렉터리 목록 보기
- `edit_file`: 파일 수정
- `create_new_file`: `edit_file` 내부에서 새 파일 생성 시 사용하는 헬퍼

3단계보다 발전된 점:

- 이제 모델이 "무슨 파일이 있는지" 먼저 확인할 수 있습니다.
- 여러 파일을 읽고 비교할 수 있습니다.
- 텍스트 파일을 만들거나 수정할 수 있습니다.

한계점:

- `list_dir`와 `read_file`이 경로를 작업공간 안으로 제한하지 않습니다.
- `edit_file`은 `str.replace(...)`를 그대로 사용하므로 `old_str`이 여러 번 나오면 모두 바뀔 수 있습니다.
- 별도의 `write_file` 도구가 없고, 새 파일 생성은 `edit_file`에서 `old_str == ""`이고 파일이 없을 때만 일어납니다.
- JavaScript 파일을 만들어도 Node.js로 실행 검증하지 않습니다.
- 시스템 프롬프트가 없어서 모델에게 작업 규칙을 강하게 고정하지 않습니다.

### 실험 1: 디렉터리 목록 확인

```bash
uv run python 4_agent_tool_extend.py
```

프롬프트에 붙여넣을 문장:

```text
what do you see in this directory?
```

### 실험 2: 파이썬 파일 요약

```bash
uv run python 4_agent_tool_extend.py
```

프롬프트에 붙여넣을 문장:

```text
Tell me about all the python files in here. Be brief!
```

### 실험 3: FizzBuzz 파일 생성

```bash
uv run python 4_agent_tool_extend.py
```

프롬프트에 붙여넣을 문장:

```text
create fizzbuzz.js that I can run with Nodejs and that has fizzbuzz in it and executes it
```

### 실험 4: FizzBuzz 출력 범위 수정

```bash
uv run python 4_agent_tool_extend.py
```

프롬프트에 붙여넣을 문장:

```text
Please edit fizzbuzz.js so that it only prints until 15
```

생성된 JavaScript를 직접 실행하려면:

```bash
node fizzbuzz.js
```

4단계 자체는 Node.js 실행 도구가 없기 때문에, 실행 검증은 사용자가 직접 해야 합니다.

## 5단계: 코드 편집 에이전트 완성형

실행:

```bash
uv run python 5_code_agent.py
```

제공 기능:

- `read_file`: 작업공간 내부 파일 읽기
- `list_dir`: 작업공간 내부 디렉터리 목록 보기
- `write_file`: 파일 전체 내용 생성 또는 덮어쓰기
- `edit_file`: 기존 파일의 특정 문자열을 정확히 한 번만 교체
- `run_node_file`: JavaScript 파일을 Node.js로 실행하고 결과 반환
- `SYSTEM_PROMPT`: 모델에게 코드 편집 에이전트로서의 행동 규칙 부여
- `resolve_workspace_path`: 작업공간 밖 경로 접근 방지

4단계보다 발전된 점:

- 파일 경로가 현재 작업 디렉터리 안에 있는지 검사합니다.
- 새 파일 생성은 `write_file`, 기존 파일 수정은 `edit_file`로 역할이 분리됩니다.
- `edit_file`은 `old_str`이 정확히 한 번만 나와야 수정합니다.
- 사용자가 요청하면 JavaScript 파일을 `run_node_file`로 실행 검증할 수 있습니다.
- 모델에게 "도구를 사용하고, 성공 전 완료라고 말하지 말라"는 시스템 프롬프트를 제공합니다.

한계점:

- `run_node_file`은 JavaScript만 실행합니다.
- Python 테스트 실행 도구는 없습니다.
- 모델이 항상 완벽한 `old_str`을 고르는 것은 아니므로 수정 실패 후 재시도할 수 있습니다.
- 파일 편집은 텍스트 기반이며, 바이너리 파일 편집은 고려하지 않습니다.

### 실험 1: 디렉터리 확인

```bash
uv run python 5_code_agent.py
```

프롬프트에 붙여넣을 문장:

```text
what do you see in this directory?
```

### 실험 2: FizzBuzz 생성, 직접 실행, 수정 검증

```bash
uv run python 5_code_agent.py
```

프롬프트에 붙여넣을 문장:

```text
Create fizzbuzz.js.
```

파일 생성이 끝나면 `Ctrl-C`로 종료합니다. 그다음 사용자가 직접 Node.js로 실행해
생성된 파일이 동작하는지 확인합니다.

```bash
node fizzbuzz.js
```

그다음 다시 실행해서, 이미 만들어진 파일을 15까지만 출력하도록 수정합니다.

```bash
uv run python 5_code_agent.py
```

```text
Now edit fizzbuzz.js so that it only prints through 15, then run it with Node.js.
```

이 시나리오의 핵심은 첫 실행에서 파일 생성만 확인하고, 사용자가 직접 `node
fizzbuzz.js`로 결과를 본 뒤, 두 번째 실행에서 이미 만들어진 파일을 수정하고
에이전트가 실행 검증까지 이어가는지 확인하는 것입니다.

### 실험 3: 작업공간 보호 확인

```bash
uv run python 5_code_agent.py
```

프롬프트에 붙여넣을 문장:

```text
read ../docs/secret-file.txt
```

이 요청은 `resolve_workspace_path`에서 차단되어야 합니다. 5단계의 에이전트는
현재 작업 디렉터리 밖의 파일을 읽거나 쓸 수 없도록 설계되어 있습니다.

## 코드 구조 핵심 개념

### Conversation

모든 파일에서 대화 기록은 다음 타입의 리스트로 관리됩니다.

```python
conversation: list[ChatCompletionMessageParam] = []
```

LLM은 이전 메시지를 자동으로 기억하지 않습니다. 따라서 매 요청마다 지금까지의
대화 기록을 함께 보내야 합니다.

### ToolDefinition

2단계부터 사용하는 `ToolDefinition`은 도구 하나를 표현합니다. 반복되는 자료구조라
`agent_lib/core.py`에 모아 두고 각 단계 파일에서 가져다 씁니다.

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    function: Callable[[dict[str, Any]], str]
```

각 필드의 의미:

- `name`: 모델이 호출할 도구 이름
- `description`: 모델이 언제 이 도구를 써야 하는지 판단할 설명
- `input_schema`: 도구 인자 구조를 설명하는 JSON Schema
- `function`: 실제로 로컬에서 실행할 파이썬 함수

### Tool Calling

모델에게 도구 목록을 전달하는 부분은 다음 흐름입니다.

```python
response = self.chat_model.complete(
    messages=conversation,
    tools=openai_tools,
)
```

모델은 직접 파일을 읽는 것이 아니라, "이 도구를 이 인자로 호출해 달라"는
구조화된 요청을 반환합니다. `agent_lib/llm.py`의 provider 계층이 실제 모델 호출을 담당하고,
로컬 프로그램이 tool 요청을 받아 실제 함수를 실행합니다.

### Tool Feedback Loop

3단계부터는 도구 실행 결과를 다시 모델에게 전달합니다.

```python
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
```

이 구조가 있어야 모델이 "파일을 읽은 뒤 그 내용을 바탕으로 답변하기"를 할 수 있습니다.

## 추천 학습 순서

1. `0_agent_no_memory.py`에서 conversation이 없으면 이전 입력을 기억하지 못한다는 점을 확인합니다.
2. `1_agent_basic.py`에서 conversation을 추가하면 대화 기억이 생기는 과정을 확인합니다.
3. `2_agent_tool.py`에서 도구 호출은 되지만 답변이 완성되지 않는 한계를 봅니다.
4. `3_agent_tool_loop.py`에서 tool 결과를 다시 모델에게 전달하는 차이를 확인합니다.
5. `4_agent_tool_extend.py`에서 파일 목록 조회와 파일 수정 도구를 실험합니다.
6. `5_code_agent.py`에서 안전한 경로 처리와 실행 검증까지 확인합니다.

## 빠른 실험 모음

각 항목은 먼저 `bash` 블록의 명령을 실행하고, 프롬프트가 뜨면 바로 아래 `text`
블록을 복사해서 붙여넣습니다.

### 0단계: conversation 없이 기억 못 하는지 확인

```bash
uv run python 0_agent_no_memory.py
```

```text
My name is Jangjun.
```

응답을 확인한 뒤 같은 실행 상태에서 이어서 입력합니다.

```text
What is my name?
```

### 1단계: conversation으로 기억하는지 확인

```bash
uv run python 1_agent_basic.py
```

```text
My name is Jangjun.
```

응답을 확인한 뒤 같은 실행 상태에서 이어서 입력합니다.

```text
What is my name?
```

### 1단계: 프롬프트만으로 함수 호출 흉내 내기

```bash
uv run python 1_agent_basic.py
```

```text
You are a weather expert. When I ask you about the weather in a given location, reply only with get_weather(<location_name>). Do not explain anything else. Understood?
```

응답을 확인한 뒤 같은 실행 상태에서 이어서 입력합니다.

```text
What's the weather in Seoul?
```

### 2단계: 파일 읽기 도구

```bash
uv run python 2_agent_tool.py
```

```text
help me solve the riddle in the docs/secret-file.txt file
```

### 3단계: tool feedback loop

```bash
uv run python 3_agent_tool_loop.py
```

```text
help me solve the riddle in the docs/secret-file.txt file
```

### 3단계: 파일 설명

```bash
uv run python 3_agent_tool_loop.py
```

```text
What's going on in 3_agent_tool_loop.py? Be brief!
```

### 4단계: 디렉터리 확인

```bash
uv run python 4_agent_tool_extend.py
```

```text
what do you see in this directory?
```

### 4단계: 파이썬 파일 요약

```bash
uv run python 4_agent_tool_extend.py
```

```text
Tell me about all the python files in here. Be brief!
```

### 5단계: FizzBuzz 생성, 직접 실행, 수정 검증

이 실험은 두 번 따로 실행해서 진행합니다.

```bash
uv run python 5_code_agent.py
```

프롬프트에 붙여넣을 문장:

```text
Create fizzbuzz.js.
```

파일 생성이 끝나면 종료하고, 사용자가 직접 실행합니다.

```bash
node fizzbuzz.js
```

그다음 다시 실행합니다.

```bash
uv run python 5_code_agent.py
```

프롬프트에 붙여넣을 문장:

```text
Now edit fizzbuzz.js so that it only prints through 15, then run it with Node.js.
```

## 문제 해결

### `Missing OPENROUTER_API_KEY.` 또는 `Missing GEMINI_API_KEY.`

`.env` 파일이 없거나 provider에 필요한 API 키가 비어 있는 상태입니다.

```bash
cp .env.example .env
```

기본값인 `LLM_PROVIDER=openrouter`에서는 OpenRouter API를 먼저 호출합니다.
`OPENROUTER_API_KEY`가 없거나 OpenRouter 인증에 실패하면 Gemini API로 fallback을
시도합니다. fallback까지 쓰려면 `OPENROUTER_API_KEY`와 `GEMINI_API_KEY`를 모두
설정하세요. Gemini만 쓰려면 `LLM_PROVIDER=gemini`로 바꾸고 `GEMINI_API_KEY`를
설정합니다.

### `No response from the configured LLM provider.`

모델 응답에 `choices`가 비어 있는 경우입니다. API 키, 모델 이름, 네트워크 상태,
OpenRouter 또는 Gemini 계정 상태를 확인하세요. 기본 로그 레벨은 `WARNING`이라
정상 HTTP 요청 로그는 숨기고 fallback 시작/실패 같은 의미 있는 알림만 출력합니다.
provider 선택이나 fallback 성공까지 보고 싶다면 `.env`에 `LOG_LEVEL=INFO`를
설정하세요.

### 도구 호출이 기대처럼 안 되는 경우

LLM의 tool calling 여부는 모델 성능과 프롬프트에 영향을 받습니다. 같은 코드를
실행해도 모델이 도구를 바로 쓰지 않을 수 있습니다. 이때는 프롬프트에 파일명이나
원하는 행동을 더 명확히 적어 보세요.

예시:

```text
Read docs/secret-file.txt using the available file-reading tool, then solve the riddle.
```

### `node` 실행 오류

`5_code_agent.py`의 `run_node_file`은 로컬에 Node.js가 설치되어 있어야 합니다.

```bash
node --version
```

Node.js가 없다면 `run_node_file` 도구는 실패합니다.
