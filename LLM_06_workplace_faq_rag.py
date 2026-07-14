from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
import os

llm = init_chat_model("google_genai:gemini-3.1-flash-lite", temperature=0, model_kwargs={"thinking_budget": 0})

# 임베딩 — 사전 준비: `ollama pull bge-m3`
embeddings = OllamaEmbeddings(model="bge-m3")  # 1024차원

DATA_DIR = Path("./data")
INDEX_DIR = Path("./faiss_index")
print("✓ ready (LLM + Ollama bge-m3 임베딩)")

# ======================================================
# 문서 로더 - Markdown / Text 로더
# ======================================================
from langchain_community.document_loaders import TextLoader
faq_docs = TextLoader(str(DATA_DIR / "faq.txt"), encoding='utf-8').load()


# ======================================================
# RecursiveCharacterTextSplitter (표준)
# ======================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # 너무 크면 분활을 한다.
    chunk_overlap=50, # 너무 작으면 결합을 한다.
    separators=["\n\n", "\n", ". ", " ", ""],  # 우선 순위 별 구분자 리스트
)

all_raw = faq_docs # + hr_docs + it_docs
chunks = splitter.split_documents(all_raw)

print(f"분할 후 청크: {len(chunks)}개")
print("\n메타데이터:", chunks[0].metadata)
print("======================================================")


# ======================================================
# FAISS 벡터 스토어
# FAISS(Facebook AI Similarity Search) 는 빠르고 가벼운 벡터 인덱스 라이브러리로, 로컬 파일 시스템에 저장 가능합니다. 
# 수백만 규모의 벡터를 밀리초 단위로 검색할 수 있어 프로토타이핑 및 중소 규모 운영 환경에 적합합니다.
# ======================================================
from langchain_community.vectorstores import FAISS

# 청크들을 임베딩하고 FAISS 인덱스에 저장
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

# 로컬에 저장
INDEX_DIR.mkdir(exist_ok=True)
vectorstore.save_local(str(INDEX_DIR), index_name="company_manual")
print(f"✓ FAISS 인덱스 저장 완료: {INDEX_DIR}/company_manual.*")


# ======================================================
# 저장된 인덱스 로드 
# ======================================================
vectorstore = FAISS.load_local(
    str(INDEX_DIR),
    embeddings,
    index_name="company_manual",
    allow_dangerous_deserialization=True,   # pickle 신뢰 플래그
)
print("✓ 인덱스 로드 완료")
print("======================================================")


# ======================================================
# Retriever로 변환
# ======================================================
retriever = vectorstore.as_retriever( # vectorstore를 체인에서 쓰려면 직접 쓸수 없고 러너블 인터페이스로 변경해서 사용해야함
    search_type="similarity",
    search_kwargs={"k": 4}, # 검색된 문서의 개수
)

# ======================================================
# RAG 체인 — LCEL 기반 구성
# ======================================================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 사내 업무 faq 어시스턴트입니다. "
     "주어진 컨텍스트만 근거로 답하세요. "
     "컨텍스트에 없는 내용은 '찾을 수 없는 정보입니다.' 라고 답하세요.\n\n"
     "[컨텍스트]\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}   # 검색이 먼지 이루어지도록 retriever를 앞쪽에 위치 시킴
    | rag_prompt
    | llm
    | StrOutputParser()
)

q1 = "직원 주차는 어디에 무료로 할수 있나요?"
faq_1 = rag_chain.invoke(q1)
print(faq_1)
print(f"{len(retriever.invoke(q1))}개 문서 반환")
print("======================================================")

# 문서에 없는 정보 — 정직한 '모름' 응답이 나와야 함
q2 = "부자가 되려면 어떻게 해야하나요?"
faq_2 = rag_chain.invoke(q2)
print(faq_2)
print(f"{len(retriever.invoke(q2))}개 문서 반환")
print("======================================================")

q3 = "우리 회사의 점심시간은 몇시야? 그리고 사내 동호회가 있다는데 관련 내용 알려줘"
faq_3 = rag_chain.invoke(q3)
print(faq_3)
print(f"{len(retriever.invoke(q3))}개 문서 반환")
print("======================================================")