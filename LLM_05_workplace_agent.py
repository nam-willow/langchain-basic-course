# .env 파일에 있는 GOOGLE_API_KEY, TAVILY_API_KEY 등을 환경변수로 로드
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent               # ReAct 에이전트를 한 줄로 만들어주는 팩토리 함수
from langchain.tools import tool                          # 함수를 LLM이 호출 가능한 도구로 변환하는 데코레이터
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver       # 대화 이력을 thread_id 별로 저장해 멀티턴 메모리를 구현하는 체크포인터
import os, json
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt    # 매 호출마다 system_prompt를 동적으로 생성하는 미들웨어 데코레이터
# ⚠️ ModelRequest 는 일부 버전에서 .types 서브모듈에 위치 — 두 경로 중 동작하는 것 사용
try:
    from langchain.agents.middleware import ModelRequest
except ImportError:
    from langchain.agents.middleware.types import ModelRequest
from langchain_tavily import TavilySearch

# TAVILY_API_KEY 존재 여부(현재는 로그용으로만 사용, 실제 분기 처리는 안 함)
TAVILY_OK = bool(os.getenv("TAVILY_API_KEY"))

# ✅ Option 1: Google Gemini 2.5 Flash-Lite (무료) — thinking_budget=0 으로 추론 토큰 비활성화
llm = init_chat_model("google_genai:gemini-3.1-flash-lite", temperature=0, model_kwargs={"thinking_budget": 0})

print(f"✅ ready (Tavily: {'on' if TAVILY_OK else 'fallback'})")


@tool
def meeting_room(date_str: str, room: str, time: int) -> dict:
    """회의실 예약 가능 여부를 조회합니다.

    date_str: 'YYYY.MM.DD HH:MM' 형식 (예: '2026.07.19 11:00')
    room: 'a회의실' 또는 'b회의실' (대소문자 무관)
    time: 사용 시간(시간 단위)
    """
    room = room.lower()  # "A회의실"처럼 LLM이 대문자로 넘겨도 매칭되도록 정규화
    print(f"date_str: {date_str}, room: {room}, time: {time}")

    # 회의실별 기존 예약 목록(하드코딩된 더미 데이터)
    meeting_room_db = {
            "a회의실": [
                {"date": datetime.strptime("2026.07.19 11:00", "%Y.%m.%d %H:%M"), "duration": 2, "team": "기획팀"},
                {"date": datetime.strptime("2026.07.19 13:00", "%Y.%m.%d %H:%M"), "duration": 1, "team": "홍보팀"},
            ],
            "b회의실": [
                {"date": datetime.strptime("2026.07.19 10:00", "%Y.%m.%d %H:%M"), "duration": 3, "team": "ai개발팀"},
            ],
        }
    if room not in meeting_room_db :
        return {"error": f"지원하지 않는 회의실: {room}"}
    try:
        req_start = datetime.strptime(date_str, "%Y.%m.%d %H:%M")
    except ValueError:
        # LLM이 형식에 안 맞는 날짜를 넣어도 크래시 대신 에러 메시지를 돌려줌
        return {"error": f"날짜 형식 오류: {date_str}"}
    req_end = req_start + timedelta(hours=time)

    # 요청 시간대(req_start~req_end)가 기존 예약 시간대와 겹치는지 하나씩 검사
    for reservation in meeting_room_db[room]:
        existing_start = reservation["date"]
        existing_end = existing_start + timedelta(hours=reservation["duration"])
        if req_start < existing_end and existing_start < req_end:  # 두 구간이 겹치는 조건
            return {
                "available": False,
                "conflict": {
                    "team": reservation["team"],
                    "start": existing_start.strftime("%Y.%m.%d %H:%M"),
                    "end": existing_end.strftime("%Y.%m.%d %H:%M"),
                },
            }

    return {"available": True}


@tool
def check_supply_stock(item_name: str) -> dict:
    """사무용품 재고 조회"""
    supply_db = {
        "A4용지": {"stock": 12, "unit": "박스"},
        "볼펜": {"stock": 3, "unit": "박스"},
        "마우스": {"stock": 0, "unit": "개"},
    }
    return supply_db.get(item_name, {"error": "등록되지 않은 품목"})

@tool
def find_employee_seat(employee_name: str) -> dict:
    """직원 이름으로 좌석 위치와 내선 번호를 조회합니다."""
    employee_db = {
        "김부장": {"seat": "3층 A구역", "extension": "1234"},
        "고길동": {"seat": "3층 B구역", "extension": "5678"},
    }
    return employee_db.get(employee_name, {"error": "등록되지 않은 직원"})


# Tavily 실시간 웹 검색 도구. LangChain 자체 제공 Tool이라 @tool 데코레이터 없이도 바로 tool_calling에 사용 가능
web_search = TavilySearch(max_results=3)

# 에이전트가 사용할 도구 목록 (LLM이 상황에 맞게 알아서 골라 호출)
tools = [meeting_room, check_supply_stock, find_employee_seat,  web_search]

# @dynamic_prompt 미들웨어에 전달되는 요청 컨텍스트(호출자 정보)
# create_agent(..., context_schema=UserContext)로 등록하고, invoke(..., context=UserContext(...))로 매 호출마다 주입
@dataclass
class UserContext:
    user_level: str = "user"       # 'user' 또는 'admin' — 권한에 따라 답변 형식/공개 범위가 달라짐
    user_name: str = ""
    employee_id: str = ""           # 도구 호출에 바로 사용

# @dynamic_prompt: 유저 컨택스트를 확인해서 프롬프트를 리턴해준다.(미들웨어)
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    ctx: UserContext = request.runtime.context
    base = "당신은 사내 업무 어시스턴트입니다."

    if ctx.user_level == "admin":
        return (
            f"{base}\n"
            f"[호출자] {ctx.user_name} / 권한: **관리자** / 사번: {ctx.employee_id}\n"
            "[답변 규칙]\n"
            "- 반드시 '[관리자 리포트]' 로 시작합니다.\n"
            "- 다음 항목을 **모두** 글머리표(•)로 포함합니다:\n"
            "   • 사번(raw 그대로)\n"
            "   • 잔여 연차(소수점 포함 정확한 숫자)\n"
            "   • 내부 정책 수치: 연차 1일 = 근무 8시간, 연간 부여 15일, 이월 한도 5일\n"
            "   • 타 직원 대비 상대 수준 (추정 코멘트 1줄)\n"
            "- 경어체는 유지하되 수치/PII 를 가리지 말고 투명하게 공개합니다.\n"
            "- 도구 호출 시 employee_id 는 호출자 사번을 그대로 사용하세요."
        )
    else:
        return (
            f"{base}\n"
            f"[호출자] {ctx.user_name} / 권한: 일반 직원 / 사번: {ctx.employee_id}\n"
            "[답변 규칙]\n"
            "- 반드시 '[안내]' 로 시작하며 **한두 문장** 으로만 답합니다.\n"
            "- 사번·정확한 소수점·타 직원 정보·내부 정책 수치는 **절대 공개 금지**.\n"
            "- 팀 평균/타 직원 비교 요청은 정중히 거절하세요.\n"
            "- 도구 호출 시 employee_id 는 호출자 사번을 그대로 사용하세요.\n"
            "- 회의실 예약 여부를 물어볼때 예약이 불가능하다면 가능한 시간대의 회의실을 추천해주세요."
        )


# create_agent 한 줄로 ReAct 에이전트 생성
# - system_prompt 인자 대신 middleware=[personalized_prompt]가 매 호출마다 동적으로 프롬프트를 만들어줌
# - context_schema를 등록해야 invoke(..., context=UserContext(...))로 넘긴 값을 미들웨어에서 request.runtime.context로 읽을 수 있음
memory = MemorySaver()
agent_dyn = create_agent(
    model=llm,
    tools=tools,
    middleware=[personalized_prompt],     # system_prompt 는 미들웨어가 제공
    context_schema=UserContext,
    checkpointer=memory,
)

# 같은 질문 — "비품 + 직원내선번호 + 회의실 예약" 를 한 번에 요청 → 권한별 응답이 확연히 갈림
# (재고/내선/회의실/웹검색까지 최소 3개 이상의 도구를 한 질문 안에서 연쇄 호출하도록 유도)
question = (
    "A4용지가 부족해서 받고싶은데 내가 1박스를 받아올수 있어? "
    "그리고 김부장님의 내선 번호좀 알려줘"
    "그리고 7월 19일 11시에 a회의실 예약가능한지 알아봐줘 5시간 사용할거야"
    "그리고 추가로 지금 인터넷에서 가장 핫한 경제뉴스가 뭔지 3줄로 정리해줘"
)

# thread 별로 관리
cfg = {"configurable" : {"thread_id": "session-001"}}
# 일반 사용자 권한으로 호출 → personalized_prompt가 "일반 직원" 분기를 태워 간략/비공개 답변을 생성
r_user = agent_dyn.invoke(
    {"messages": [{"role": "user", "content": question}]},
    context=UserContext(user_level="user", user_name="고길동", employee_id="E67890"),
    config=cfg
)
print("[USER]")
# 응답 메시지 리스트의 마지막(최종 답변) AIMessage에서 텍스트만 추출
# content가 문자열이 아니라 콘텐츠 블록 리스트로 올 수도 있어 .text 프로퍼티로 안전하게 꺼냄
print(r_user["messages"][-1].text)
print()
print("-" * 60)
print()

# thread 별로 관리
cfg2 = {"configurable" : {"thread_id": "session-002"}}
# 관리자 권한으로 동일 질문 호출 → "관리자" 분기가 태워져 사번/정책 수치까지 상세 공개
r_admin = agent_dyn.invoke(
    {"messages": [{"role": "user", "content": question}]},
    context=UserContext(user_level="admin", user_name="김부장", employee_id="E12345"),
    config=cfg2
)
print("[ADMIN]")
print(r_admin["messages"][-1].text)