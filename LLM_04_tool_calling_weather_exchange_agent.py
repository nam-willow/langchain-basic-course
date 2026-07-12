# .env 파일에 있는 GROQ_API_KEY, TAVILY_API_KEY 등을 환경변수로 로드
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
import os
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain.messages import HumanMessage, ToolMessage

# TAVILY_API_KEY 존재 여부(현재는 로그용으로만 사용, 실제 분기 처리는 안 함)
TAVILY_OK = bool(os.getenv("TAVILY_API_KEY"))

# Groq Qwen3 32B (무료) — reasoning_effort="none" 으로 <think> 토큰 비활성화
# init_chat_model()은 "provider:model" 형식 문자열로 여러 벤더 모델을 동일한 인터페이스로 불러옴
llm = init_chat_model("groq:qwen/qwen3-32b", temperature=0, reasoning_effort="none")

# Tavily 실시간 웹 검색 도구. LangChain 자체 제공 Tool이라 @tool 데코레이터 없이도 바로 tool_calling에 사용 가능
web_search = TavilySearch(max_results=3, topic="general")
print("✓ 실제 Tavily 웹검색 도구 활성화")

@tool
def get_weather(city: str) -> dict:
    """도시의 현재 날씨를 알려줍니다."""
    weather_db = {"서울": {"temp": 37, "condition": "맑음", "caution":"폭염주의"}, "경기도":{"temp": 35, "condition": "맑음", "caution":None}}
    if city not in weather_db :
        return {"error": f"지원하지 않는 도시: {city}"}
    weather = weather_db.get(city)
    return {"city": city, "weather": weather}

@tool
def convert_currency(amount: float, from_cur: str, to_cur: str) -> dict:
    """환율 정보를 이용해 금액을 변환합니다."""
    # 통화별로 원화(KRW) 1단위 기준 환율만 저장해두고, KRW를 매개로 두 통화 간 교차 환산
    # (예: JPY -> USD 도 KRW를 거쳐 계산되므로 통화쌍이 늘어나도 테이블이 N개면 충분)
    rates_to_krw = {"KRW": 1, "USD": 1529.70, "JPY": 9.4605}  # 1단위당 원화 환율
    if from_cur not in rates_to_krw or to_cur not in rates_to_krw:
        # LLM이 지원 안 하는 통화 코드(예: EUR)를 넣어도 크래시 대신 에러 메시지를 돌려줌
        return {"error": f"지원하지 않는 통화: {from_cur} 또는 {to_cur}"}
    krw = amount * rates_to_krw[from_cur]       # 1) from_cur -> KRW 환산
    result = krw / rates_to_krw[to_cur]          # 2) KRW -> to_cur 환산
    return {"from_cur": from_cur, "to_cur": to_cur, "amount": amount, "result": round(result, 2)}


# 도구 레지스트리
workplace_tools = [get_weather, convert_currency]
# print(f"정의된 업무 도구: {len(workplace_tools)}개")
# for t in workplace_tools:
    # print(f"  - {t.name}: {t.description.splitlines()[0]}")

# # 모든 도구를 한 번에 연결
all_tools = workplace_tools + [web_search]
llm_with_tools = llm.bind_tools(all_tools)

tool_map = {t.name: t for t in all_tools}

def run_tool_loop(user_query: str, max_iter: int = 5, verbose: bool = False):
    messages = [HumanMessage(content=user_query)]

    for step in range(max_iter):
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)

        # 더 이상 도구 호출이 없으면 종료
        if not ai.tool_calls:
            if verbose:
                print(f"\n[step {step+1}] FINAL ANSWER")
            return ai.content

        # 도구 호출 실행
        for tc in ai.tool_calls:
            name = tc["name"]
            args = tc["args"]
            if verbose:
                print(f"[step {step+1}] 🔧 {name}({args})")
            try:
                result = tool_map[name].invoke(args)
            except Exception as e:
                result = f"Error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            # print(messages)

    return "(max iterations reached)"


# text = "서울 날씨 알려주고, 100달러는 한화로 얼마인지 계산해줘. 그리고 내가 오늘 나가서 환전을 해도 괜찮을지 알려줘"
text = "일본 날씨 알려주고, 220엔은 한화로 얼마인지 계산해줘. 그리고 내가 오늘 나가서 환전을 해도 괜찮을지 알려줘"

print("■ 질문:")
print(text)
answer = run_tool_loop(text)
print("\n■ 답변:")
print(answer)

