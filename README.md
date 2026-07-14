# langchain-basic-course

## 1번째 코드: 챗봇 만들기

[LLM_01_mini_chatbot.py](LLM_01_mini_chatbot.py) — Gemini/

### OpenAI 자동 전환 미니 챗봇

- 스크립트를 실행하면 질문하고 싶은 메시지를 입력
- `quit` 입력 전까지 대화가 계속 반복됨
- `.env`에 `GOOGLE_API_KEY` 또는 `OPENAI_API_KEY` 중 하나가 있으면 자동 감지하여 사용 (Gemini 우선, 없으면 OpenAI)


---

## 2번째 코드: LangChain 기반 한→영 번역 챗봇

[LLM_02_langchain_translation.py](LLM_02_langchain_translation.py)

- LCEL 방식으로 chain 정의
- batch를 활용한 병렬 API 호출 (invoke로 선택 가능)
- StrOutputParser 설정 (JsonOutputParser로 선택 가능)

---

## 3번째 코드: 구조화 출력 기반 CS 리뷰 티켓 분류

[LLM_03_langchain_cs_analysis.py](LLM_03_langchain_cs_analysis.py)

- Pydantic 스키마(`TicketClassification`) + `with_structured_output`으로 리뷰를 키워드/감정/긴급도/행동제안으로 구조화 추출
- `batch`로 여러 리뷰를 동시에 처리 (`max_concurrency`, `return_exceptions` 옵션 사용)

---

## 4번째 코드: 도구 호출 기반 날씨/환율 에이전트

[LLM_04_tool_calling_weather_exchange_agent.py](LLM_04_tool_calling_weather_exchange_agent.py)

- `@tool` 데코레이터로 `get_weather`, `convert_currency` 업무 도구 정의 + `TavilySearch`로 실시간 웹 검색 도구 추가
- `bind_tools`로 LLM에 도구를 연결하고, `tool_calls` 응답을 직접 순회하며 실행하는 tool calling 루프(`run_tool_loop`) 구현
- 도구 실행 결과를 `ToolMessage`로 대화 이력에 추가해 LLM이 최종 답변을 생성할 때까지 반복(`max_iter`로 무한 루프 방지)
- Groq `qwen/qwen3-32b` 모델 사용, `reasoning_effort="none"`으로 `<think>` 토큰 비활성화

---

## 5번째 코드: create_agent 기반 사내 업무 에이전트 (실습 과제)

[LLM_05_workplace_agent.py](LLM_05_workplace_agent.py)

- `@tool`로 `meeting_room`(회의실 예약 조회), `check_supply_stock`(사무용품 재고), `find_employee_seat`(직원 좌석/내선) 3개 사내 도구 정의 + `TavilySearch` 웹 검색 도구 추가
- `create_agent` + `MemorySaver`로 에이전트를 구성하고, `thread_id`별로 `invoke`를 호출해 세션 메모리 분리
- `@dynamic_prompt` 미들웨어와 `UserContext`(`context_schema`)로 호출자 권한(`user`/`admin`)에 따라 시스템 프롬프트를 동적으로 분기 — 관리자는 사번/정책 수치까지 상세 공개, 일반 직원은 요약된 안내만 제공
- 하나의 질문 안에서 재고 조회 → 내선 조회 → 회의실 예약 조회 → 웹검색까지 여러 도구를 연쇄 호출하도록 설계
- Google Gemini `gemini-3.1-flash-lite` 모델 사용, `thinking_budget=0`으로 추론 토큰 비활성화

---

## 6번째 코드: 사내 FAQ RAG 파이프라인 (실습 과제)

[LLM_06_workplace_faq_rag.py](LLM_06_workplace_faq_rag.py)

- `TextLoader`로 `data/faq.txt`(사내 총무·시설 안내 문서)를 로드하고 `RecursiveCharacterTextSplitter`(`chunk_size=500`, `chunk_overlap=50`)로 분할
- Ollama `bge-m3` 임베딩(1024차원)으로 청크를 벡터화해 `FAISS` 인덱스 생성 후 로컬 저장(`example/faiss_index`) 및 재로드
- `as_retriever(search_type="similarity", search_kwargs={"k": 4})`로 벡터스토어를 리트리버로 변환해 LCEL 체인에 연결
- `{"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()` 형태로 RAG 체인 구성 — 컨텍스트에 없는 질문은 "찾을 수 없는 정보입니다."로 정직하게 응답하도록 프롬프트 설계
- 문서 내 질문 / 문서에 없는 질문 / 복합 질문 3가지 케이스로 검증, `retriever.invoke()`로 실제 검색된 문서 개수 확인
- Google Gemini `gemini-3.1-flash-lite` LLM + Ollama `bge-m3` 임베딩 조합 사용

---

참고강의 : 판다스 스튜디오 — langchain-basic-course
https://github.com/pandas-studio/langchain-basic-course