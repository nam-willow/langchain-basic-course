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

참고강의 : 판다스 스튜디오 — langchain-basic-course
https://github.com/pandas-studio/langchain-basic-course