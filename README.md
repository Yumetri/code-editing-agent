# Code Editing Agent 학습 가이드

이 저장소는 LLM 기반 코드 편집 에이전트를 아주 작은 단계부터 확장해 가며
구현하는 튜토리얼입니다.

핵심 파일은 다음 5개입니다.

1. `1_agent_basic.py`: 기본 채팅 에이전트
2. `2_agent_tool.py`: LLM tool calling 첫 도입
3. `3_agent_tool_loop.py`: tool 실행 결과를 다시 LLM에게 전달하는 루프
4. `4_agent_tool_extend.py`: 파일 목록 조회, 파일 생성, 파일 수정 도구 확장
5. `5_code_agent.py`: 작업공간 보호와 Node.js 실행 검증을 갖춘 코드 편집 에이전트

각 단계에서 직접 입력해 볼 실습 프롬프트는 이 README의 예제 명령과
빠른 실습 모음에 통합되어 있습니다.

## 준비

이 프로젝트는 `uv`와 OpenRouter API 키를 사용합니다.

```bash
uv sync
```

`.env.example`을 `.env`로 복사한 뒤, OpenRouter API 키를 넣습니다.

```bash
cp .env.example .env
```

`.env` 예시:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

모든 예제는 기본적으로 다음 모델과 OpenRouter 호환 API를 사용합니다.

```python
MODEL_NAME = "poolside/laguna-m.1:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

모델을 바꾸고 싶다면 각 파이썬 파일의 `MODEL_NAME` 값을 수정하면 됩니다.

## 전체 발전 흐름

| 단계 | 파일 | 발전 포인트 | 핵심 한계 |
| --- | --- | --- | --- |
| 1 | `1_agent_basic.py` | 사용자 입력과 LLM 응답을 대화 기록으로 이어감 | 파일을 읽거나 실행할 수 없음 |
| 2 | `2_agent_tool.py` | `read_file` 도구를 LLM에게 공개함 | 도구 결과를 LLM에게 다시 전달하지 않음 |
| 3 | `3_agent_tool_loop.py` | `role="tool"` 메시지로 도구 결과를 대화에 추가함 | 읽기 도구만 있어서 수정/실행 불가 |
| 4 | `4_agent_tool_extend.py` | `list_dir`, `edit_file` 추가로 파일 탐색/생성/수정 가능 | 작업공간 밖 경로 접근 보호가 약하고 실행 검증 없음 |
| 5 | `5_code_agent.py` | 경로 보호, `write_file`, 정확한 `edit_file`, `run_node_file` 검증 추가 | Python 실행 도구는 없고 JavaScript 실행만 지원 |

## 1단계: 기본 채팅 에이전트

실행:

```bash
uv run python 1_agent_basic.py
```

제공 기능:

- 터미널에서 사용자 입력을 받습니다.
- 입력과 모델 응답을 `conversation` 리스트에 누적합니다.
- 매 요청마다 이전 대화 기록 전체를 모델에 전달합니다.
- 그래서 모델이 앞에서 말한 이름이나 맥락을 어느 정도 기억할 수 있습니다.

한계점:

- 모델은 로컬 파일을 볼 수 없습니다.
- 도구 호출 기능이 없습니다.
- 사용자가 프롬프트로 "함수처럼 답해줘"라고 유도할 수는 있지만, 실제 함수가 실행되지는 않습니다.

실습 입력 예시:

```bash
uv run python 1_agent_basic.py
```

실행 후 아래처럼 입력합니다.

```text
Hey! I'm Jangjun! How are you?
Can you come up with any horse-related nicknames that make fun of my first name?
```

복사해서 한 번에 흘려보내고 싶다면:

```bash
printf '%s\n%s\n' \
  "Hey! I'm Jangjun! How are you?" \
  "Can you come up with any horse-related nicknames that make fun of my first name?" \
  | uv run python 1_agent_basic.py
```

프롬프트만으로 가짜 tool calling을 유도하는 예시:

```bash
printf '%s\n%s\n' \
  "You are a weather expert. When I ask you about the weather in a given location, I want you to reply with \`get_weather(<location_name>)\`. I will then tell you what the weather in that location is. Understood?" \
  "Hey, what's the weather in Seoul?" \
  | uv run python 1_agent_basic.py
```

이 예시는 실제 `get_weather` 함수를 실행하는 것이 아닙니다. 모델이 텍스트로
`get_weather(Seoul)` 같은 답변을 하도록 프롬프트로 유도할 뿐입니다.

## 2단계: 파일 읽기 도구 추가

실행:

```bash
uv run python 2_agent_tool.py
```

제공 기능:

- `ToolDefinition` 데이터 클래스로 도구의 이름, 설명, 입력 스키마, 실행 함수를 묶습니다.
- `read_file` 도구를 모델에게 공개합니다.
- 모델이 `tool_call`을 만들면 파이썬 코드가 해당 도구를 실제로 실행합니다.
- `secret-file.txt` 같은 로컬 파일을 읽을 수 있습니다.

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

실습 입력 예시:

```bash
printf '%s\n' \
  "help me solve the riddle in the secret-file.txt file" \
  | uv run python 2_agent_tool.py
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

실습 입력 예시:

```bash
printf '%s\n' \
  "help me solve the riddle in the secret-file.txt file" \
  | uv run python 3_agent_tool_loop.py
```

`secret-file.txt`의 수수께끼는 다음 내용입니다.

```text
'what animal is the most disagreeable because it always says neigh?'
```

이 단계에서는 모델이 파일을 읽고, 그 결과를 다시 받아 답을 추론할 수 있습니다.

추상적인 파일 설명 요청 예시:

```bash
printf '%s\n' \
  "What's going on in 3_agent_tool_loop.py? Be brief!" \
  | uv run python 3_agent_tool_loop.py
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

디렉터리 목록 확인:

```bash
printf '%s\n' \
  "what do you see in this directory?" \
  | uv run python 4_agent_tool_extend.py
```

파이썬 파일 요약:

```bash
printf '%s\n' \
  "Tell me about all the python files in here. Be brief!" \
  | uv run python 4_agent_tool_extend.py
```

FizzBuzz 파일 생성:

```bash
printf '%s\n' \
  "create fizzbuzz.js that I can run with Nodejs and that has fizzbuzz in it and executes it" \
  | uv run python 4_agent_tool_extend.py
```

FizzBuzz 출력 범위 수정:

```bash
printf '%s\n' \
  "Please edit fizzbuzz.js so that it only prints until 15" \
  | uv run python 4_agent_tool_extend.py
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
- JavaScript 파일을 만들거나 수정한 뒤 `run_node_file`로 실행 검증할 수 있습니다.
- 모델에게 "도구를 사용하고, 성공 전 완료라고 말하지 말라"는 시스템 프롬프트를 제공합니다.

한계점:

- `run_node_file`은 JavaScript만 실행합니다.
- Python 테스트 실행 도구는 없습니다.
- 모델이 항상 완벽한 `old_str`을 고르는 것은 아니므로 수정 실패 후 재시도할 수 있습니다.
- 파일 편집은 텍스트 기반이며, 바이너리 파일 편집은 고려하지 않습니다.

디렉터리 확인:

```bash
printf '%s\n' \
  "what do you see in this directory?" \
  | uv run python 5_code_agent.py
```

FizzBuzz 생성 후 실행 검증까지 요청:

```bash
printf '%s\n' \
  "create fizzbuzz.js that I can run with Nodejs and that has fizzbuzz in it and executes it" \
  | uv run python 5_code_agent.py
```

FizzBuzz를 15까지만 출력하도록 수정하고 검증:

```bash
printf '%s\n' \
  "Please edit fizzbuzz.js so that it only prints until 15, then run it with Nodejs" \
  | uv run python 5_code_agent.py
```

작업공간 보호 확인 예시:

```bash
printf '%s\n' \
  "read ../secret-file.txt" \
  | uv run python 5_code_agent.py
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

2단계부터 등장하는 `ToolDefinition`은 도구 하나를 표현합니다.

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
kwargs: dict[str, Any] = {
    "model": MODEL_NAME,
    "messages": conversation,
}
if openai_tools:
    kwargs["tools"] = openai_tools
```

모델은 직접 파일을 읽는 것이 아니라, "이 도구를 이 인자로 호출해 달라"는
구조화된 요청을 반환합니다. 로컬 프로그램이 그 요청을 받아 실제 함수를 실행합니다.

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

1. `1_agent_basic.py`를 실행해서 대화 기록이 어떻게 쌓이는지 확인합니다.
2. `2_agent_tool.py`에서 도구 호출은 되지만 답변이 완성되지 않는 한계를 봅니다.
3. `3_agent_tool_loop.py`에서 tool 결과를 다시 모델에게 전달하는 차이를 확인합니다.
4. `4_agent_tool_extend.py`에서 파일 목록 조회와 파일 수정 도구를 실습합니다.
5. `5_code_agent.py`에서 안전한 경로 처리와 실행 검증까지 확인합니다.

## 빠른 실습 모음

아래 명령들은 단계별 실습 질문을 바로 실행할 수 있게 정리한 것입니다.

```bash
printf '%s\n%s\n' \
  "Hey! I'm Thorsten! How are you?" \
  "Can you come up with any horse-related nicknames that make fun of my first name?" \
  | uv run python 1_agent_basic.py
```

```bash
printf '%s\n%s\n' \
  "You are a weather expert. When I ask you about the weather in a given location, I want you to reply with \`get_weather(<location_name>)\`. I will then tell you what the weather in that location is. Understood?" \
  "Hey, what's the weather in Seoul?" \
  | uv run python 1_agent_basic.py
```

```bash
printf '%s\n' \
  "help me solve the riddle in the secret-file.txt file" \
  | uv run python 2_agent_tool.py
```

```bash
printf '%s\n' \
  "help me solve the riddle in the secret-file.txt file" \
  | uv run python 3_agent_tool_loop.py
```

```bash
printf '%s\n' \
  "What's going on in 3_agent_tool_loop.py? Be brief!" \
  | uv run python 3_agent_tool_loop.py
```

```bash
printf '%s\n' \
  "what do you see in this directory?" \
  | uv run python 4_agent_tool_extend.py
```

```bash
printf '%s\n' \
  "Tell me about all the python files in here. Be brief!" \
  | uv run python 4_agent_tool_extend.py
```

```bash
printf '%s\n' \
  "create fizzbuzz.js that I can run with Nodejs and that has fizzbuzz in it and executes it" \
  | uv run python 5_code_agent.py
```

```bash
printf '%s\n' \
  "Please edit fizzbuzz.js so that it only prints until 15, then run it with Nodejs" \
  | uv run python 5_code_agent.py
```

## 문제 해결

### `Missing OPENROUTER_API_KEY.`

`.env` 파일이 없거나 API 키가 비어 있는 상태입니다.

```bash
cp .env.example .env
```

그 뒤 `.env` 안의 값을 실제 OpenRouter API 키로 바꿉니다.

### `No response from OpenRouter.`

모델 응답에 `choices`가 비어 있는 경우입니다. API 키, 모델 이름, 네트워크 상태,
OpenRouter 계정 상태를 확인하세요.

### 도구 호출이 기대처럼 안 되는 경우

LLM의 tool calling 여부는 모델 성능과 프롬프트에 영향을 받습니다. 같은 코드를
실행해도 모델이 도구를 바로 쓰지 않을 수 있습니다. 이때는 프롬프트에 파일명이나
원하는 행동을 더 명확히 적어 보세요.

예시:

```text
Read secret-file.txt using the available file-reading tool, then solve the riddle.
```

### `node` 실행 오류

`5_code_agent.py`의 `run_node_file`은 로컬에 Node.js가 설치되어 있어야 합니다.

```bash
node --version
```

Node.js가 없다면 `run_node_file` 도구는 실패합니다.
