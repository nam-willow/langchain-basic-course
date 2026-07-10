"""
2026.07.10
미니 챗봇 만들기

1. 스크립트를 실행하여 질문하고싶은 메시지를 입력합니다.
2. quit를 입력하기전까지 대화는 계속 반복됩니다. 종료를 원하면 quit를 입력해주세요.
"""

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

# 사용 가능한 키 자동 감지
HAS_GEMINI = bool(os.getenv("GOOGLE_API_KEY"))
HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

if not (HAS_GEMINI or HAS_OPENAI):
    raise RuntimeError("[error] 환경변수 오류")

USE_PROVIDER = "gemini" if HAS_GEMINI else "openai"

## 사용가능한 모델 리스트 확인
# key = os.getenv("GOOGLE_API_KEY")
# r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
# for m in r.json().get("models", []):
#     if "generateContent" in m.get("supportedGenerationMethods", []):
#         print(m["name"])

# 모델명 설정
client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-3.1-flash-lite"

# 대화 함수
def chat(history):
    # 호출
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        temperature=0.3,
    )

    # print(f"[{MODEL}]")
    print("------------------------------")
    reply = response.choices[0].message.content
    history.append({"role": "assistant" , "content": reply})
    print(reply)

    # 응답 전체 구조 확인
    print("")
    print("------------------------------")
    print(f"모델: {response.model} / 입력token: {response.usage.prompt_tokens} / 출력token: {response.usage.completion_tokens} / 합계token: {response.usage.total_tokens}")
    

history = [{"role": "system", 
            "content": "당신은 친절하고 정확한 한국어 사내 AI 지원 도우미 입니다."},]

while True:
    print("------------------------------")
    promft = input("메시지를 입력하세요: ")
    
    if promft.lower() == 'quit':
        print("감사합니다. 챗봇을 종료합니다.")
        break
    history.append({"role": "user" , "content": promft})
    print(chat(history))

    