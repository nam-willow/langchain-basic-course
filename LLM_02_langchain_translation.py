from dotenv import load_dotenv
import warnings
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate # 변수기반 프롬프트
from langchain_core.output_parsers import StrOutputParser # AIMessage 객체에서 문자열 content만 추출
from langchain_core.output_parsers import JsonOutputParser # AIMessage 객체에서 Json content 만 추출

load_dotenv()

print(f"사용 API Key name - { "gemini" if os.getenv("GOOGLE_API_KEY") else "-"}")

# 사용 모델 호출
llm = init_chat_model("google_genai:gemini-3.1-flash-lite", temperature=0.3)

# ----------------- option 1. StrOutputParser -----------------
# AIMessage 객체에서 문자열 content만 추출
parser = StrOutputParser()
# 기본 프롬프트 세팅
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", "입력된 한국어 문장을 영어로 번역한 뒤, 지정된 톤 {tone}으로 다듬으세요. 표현 설명은 제외하고 다듬은 문장만 답변주세요."),
    ("human", "{prompt}")
])

# #----------------- option 1. JsonOutputParser -----------------
# # AIMessage 객체에서 json content만 추출
# parser = JsonOutputParser()
# # 기본 프롬프트 세팅
# translation_prompt = ChatPromptTemplate.from_messages([
#     ("system", "입력된 한국어 문장을 영어로 번역한 뒤, 지정된 톤 {tone}으로 다듬으세요. 표현 설명은 제외하고 다듬은 문장만 답변주세요." 
#      "다음 JSON 형식으로만 답변하세요. 다른 설명, 마크다운 코드블록은 포함하지 마세요.\n"
#      '{{"tone": "적용된 톤", "translated": "번역 및 톤 조정된 문장"}}'),
#     ("human", "{prompt}")
# ])


# LCEL: LangChain 방식 "|" 을 사용해 컴포넌트 연결, 체인 정의
chain = translation_prompt | llm | parser

text = "안녕하세요 저는 홍길동입니다. 저는 ai engineer가 되고싶습니다. 그래서 안써본 기술을 열심히 사용해보면서 경험을 만들어 나가는 개발자 입니다. 잘부탁드립니다."
# text = input("번역할 내용을 작성해주세요: ")

# # ----------------- option 2. invoke -----------------
# result = chain.invoke({"tone": "formal", "news": text})
# print(result)

# ----------------- option 2. batch -----------------
result = chain.batch([
        {"tone": "formal", "prompt": text},
        {"tone": "casual", "prompt": text},
        {"tone": "polite", "prompt": text},
    ],
    # config={"max_concurrency": 2}  # 동시에 최대 실행 개수를 지정할수 있음(무료로 사용할경우)
    )

for i, r in enumerate(result):
    print(f"[{i}] - {r}")

