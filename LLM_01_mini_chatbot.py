"""
2026.07.10
미니 챗봇 만들기

1. 스크립트를 실행하여 질문하고싶은 메시지를 입력합니다.
2. quit를 입력하기전까지 대화는 계속 반복됩니다. 종료를 원하면 quit를 입력해주세요.
"""


# 환경 변수 로드 + 자동 분기 (Gemini 우선, 없으면 OpenAI)
from dotenv import load_dotenv
import os
from openai import OpenAI


load_dotenv()

# 사용 가능한 키 자동 감지
HAS_GEMINI = bool(os.getenv("GOOGLE_API_KEY"))
HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

if not (HAS_GEMINI or HAS_OPENAI):
    raise RuntimeError(
        "[error] GOOGLE_API_KEY 또는 OPENAI_API_KEY 둘 중 하나가 .env 에 있어야 합니다.\n"
        "   → SETUP.md 의 4번 섹션을 참고해서 키를 발급받으세요. (Gemini는 무료!)"
    )

# 우선순위: 무료 Gemini → 유료 OpenAI
USE_PROVIDER = "gemini" if HAS_GEMINI else "openai"
# print(f"[success] 환경 변수 로드 완료. 사용 제공자: {USE_PROVIDER}")



## 사용가능한 모델 리스트 확인
# import requests, os

# key = os.getenv("GOOGLE_API_KEY")
# r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
# for m in r.json().get("models", []):
#     if "generateContent" in m.get("supportedGenerationMethods", []):
#         print(m["name"])

##################

# 사용 제공자에 따라 클라이언트와 모델명을 자동 설정
if USE_PROVIDER == "gemini":
    # 무료: Google Gemini (OpenAI 호환 엔드포인트)
    client = OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    # MODEL = "gemini-2.5-flash-lite"
    MODEL = "gemini-3.1-flash-lite"
else:
    # 유료: OpenAI
    client = OpenAI()  # OPENAI_API_KEY 자동 감지
    MODEL = "gpt-4.1-mini"

def chat(history):
    # 첫 호출
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
    print("모델:", response.model)
    print("완료 이유:", response.choices[0].finish_reason)
    print("토큰 사용량:")
    print(f"  - 입력(prompt): {response.usage.prompt_tokens}")
    print(f"  - 출력(completion): {response.usage.completion_tokens}")
    print(f"  - 합계(total): {response.usage.total_tokens}")
    

history = [{"role": "system", 
            "content": "당신은 친절하고 정확한 한국어 사내 AI 지원 도우미 입니다."},]

while True:
    print("------------------------------")
    promft = input("메시지를 입력하세요: ")
    
    if promft.strip().lower() == 'quit':
        print("감사합니다. 챗봇을 종료합니다.")
        break
    history.append({"role": "user" , "content": promft})
    print(chat(history))

    