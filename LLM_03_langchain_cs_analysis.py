from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
import re
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


# Groq Qwen3 32B (무료) — reasoning_effort="none" 으로 <think> 토큰 비활성화
llm = init_chat_model("groq:qwen/qwen3-32b", temperature=0, reasoning_effort="none")

def strip_think(text: str) -> str:
    """<think>...</think> 추론 토큰 블록을 제거합니다. (안전 장치로 설정)"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# StrOutputParser → AIMessage.content 추출 후 <think> 블록 제거
parser = StrOutputParser() | RunnableLambda(strip_think)

# 스키마
class TicketClassification(BaseModel):
    """리뷰 분석 스키마"""
    keywords: list[str] = Field(description="핵심 키워드 (최대 3개, 3개를 넘지 마세요)")
    sentiment: Literal["긍정", "중립", "부정"]  = Field(description="감정 분석")
    urgency: int = Field(description="처리 긴급도 1(낮음)~5(높음) 정수")
    action_suggestion: str = Field(description="20자 이내 정확한 처리행동 제안")

# 출력 스키마
# 한 줄을 적용하면 모델 응답이 곧바로 Pydantic 객체로 반환
structured_llm = llm.with_structured_output(TicketClassification)

# 리뷰 리스트
reviews = [
    "배송 진짜 느리네요. 친구 결혼식에 가려고 구매했는데 아직도 안와서 다른곳에서 구매해서 입고갔습니다. 다음엔 여기서 구매 안합니다.",
    "배송은 느리지만 디자인 진짜 예뻐요!! 친구한테 선물했는데 너무 좋아하네요 ㅎㅎ",
    "안경이 너무 무거워요. 표기는 L인데 실제로는 M 같은 느낌. 환불 요청드립니다",
    "제품이 잘못왓어요 저는 노란색을 시켰는데 검은색이 왔습니다. 환불이나 교환요청을 하기에는 귀찮기도 하고 생각보다 괜찮아서 그냥 쓸게요 다음에는 신경써주세요",
]

# # -------------------- invoke --------------------
# for r in reviews : 
#     ticket = structured_llm.invoke(r)
#     print("----------")
#     print("keywords     : ", ticket.keywords)
#     print("감정         : ", ticket.sentiment)
#     print("처리긴급도   : ", ticket.urgency)
#     print("행동제안     : ", ticket.action_suggestion)

# -------------------- batch --------------------
tickets = structured_llm.batch(reviews, config={"max_concurrenoy":4}, return_exceptions=True)
for t in tickets:
    # print(ticket)
    print("----------")
    print("keywords     : ", t.keywords)
    print("감정         : ", t.sentiment)
    print("처리긴급도   : ", t.urgency)
    print("행동제안     : ", t.action_suggestion)