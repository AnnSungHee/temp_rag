import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --------------------------------------------
# 0) 환경변수 / 의존성 준비
# --------------------------------------------
# - .env 파일에 저장된 환경 변수(GROQ_API_KEY 등)를 OS 환경변수로 로드
# - Groq: LLM 호출용 클라이언트
# - sentence-transformers: 문장을 벡터로 바꾸는 임베딩 모델
# - chromadb: 벡터DB(유사도 검색) 저장 및 조회
load_dotenv()

# --------------------------------------------
# 2) ChromaDB 영속 클라이언트 생성
# --------------------------------------------
# - PersistentClient: 로컬 디스크(./chroma_db)에 데이터가 저장되어 재실행해도 유지됩니다.
# - path 아래에 컬렉션/임베딩 인덱스가 저장됩니다.
client = chromadb.PersistentClient(path="./chroma_db")

# --------------------------------------------
# 3) 컬렉션 생성 + 임베딩 함수 지정
# --------------------------------------------
# - Chroma 컬렉션은 "문서 + 임베딩 벡터"를 함께 저장하는 단위입니다.
# - embedding_function을 지정하면:
#   1) add() 시 문서를 임베딩해서 저장하고
#   2) query() 시 질문도 임베딩해서 유사도 검색합니다.
#
# - SentenceTransformerEmbeddingFunction은 model_name을 받아 내부에서 모델을 로드합니다.
# - 같은 모델을 사용해야 문서 임베딩과 질문 임베딩이 동일한 벡터 공간에 존재합니다.
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="my_documents",
    embedding_function=sentence_transformer_ef
)

# --------------------------------------------
# 4-1) 컬렉션에 데이터가 없으면 초기 데이터 적재
# --------------------------------------------
# - count() == 0이면 첫 실행으로 보고 문서를 넣습니다.
# - 이미 데이터가 있으면 그대로 재사용합니다(중복 적재 방지).
#
# 주의: 여기서 documents를 다시 3개짜리로 재정의합니다.
#      위에서 5개 준비했지만 실제로는 3개만 들어가게 됩니다.
if collection.count() == 0:
    documents = [
        "파이썬은 데이터 분석에 매우 유용한 프로그래밍 언어입니다.",
        "머신러닝 모델을 학습시키기 위해서는 많은 데이터가 필요합니다.",
        "자연어 처리는 인공지능의 중요한 분야 중 하나입니다.",
        "벡터 데이터베이스는 유사도 검색에 최적화되어 있습니다.",
        "한국어와 영어를 모두 지원하는 다국어 모델입니다."
    ]

    # - ids는 문서 식별자. 컬렉션 내에서 유일해야 합니다.
    # - documents와 ids의 길이는 동일해야 합니다.
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    print("새로운 데이터 추가 완료")
else:
    print(f"기존 컬렉션 사용 (문서 수: {collection.count()})")


# --------------------------------------------
# 5) Groq 클라이언트 초기화
# --------------------------------------------
# - GROQ_API_KEY는 .env에서 로드된 환경 변수에 있어야 합니다.
# - api_key가 None이면 호출 시 인증 에러가 납니다.
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# --------------------------------------------
# 6) 사용자 질문 입력
# --------------------------------------------
# - CLI에서 질문을 입력받습니다.
user_question = input("질문을 입력하세요: ")


# --------------------------------------------
# 7) 관련 문서(Top-K) 검색
# --------------------------------------------
# - query_texts: 질문(문자열)을 임베딩 → 컬렉션 내부 벡터들과 유사도 계산
# - n_results=3: 가장 가까운 문서 3개를 반환
#
# results 구조는 대략:
# {
#   'ids': [[...]],
#   'documents': [[...]],
#   'distances': [[...]],
#   ...
# }
results = collection.query(query_texts=[user_question], n_results=3)

# - results['documents']는 2차원 리스트 형태(질문이 여러 개일 수 있어서)
# - 지금은 질문 1개만 넣었으므로 [0]을 취해 문서 리스트로 만듭니다.
retrieved_docs = results['documents'][0]


# --------------------------------------------
# 8) 컨텍스트 구성
# --------------------------------------------
# - LLM에게 제공할 "근거 문맥"을 만들기 위해 검색된 문서를 bullet 형태로 연결합니다.
context = "\n".join([f"- {doc}" for doc in retrieved_docs])


# --------------------------------------------
# 9) 프롬프트 구성 및 LLM 응답 생성
# --------------------------------------------
# - system_prompt: 모델의 역할/규칙을 고정하는 지침
# - user_prompt: 실제 질문 + 검색된 문맥(context)을 전달
#
# 핵심: “문맥에 있는 정보만 답하라”고 강제해 환각(hallucination)을 줄이는 구조입니다.
system_prompt = """당신은 제공된 문맥을 기반으로 정확하게 답변하는 AI 어시스턴트입니다.
다음 규칙을 반드시 따르세요:
1. 제공된 문맥에 있는 정보만 사용하여 답변하세요.
2. 문맥에 답변이 없다면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요.
3. 답변은 명확하고 간결하게 작성하세요.
4. 한국어로 답변하세요."""

user_prompt = f"""문맥:
{context}

질문: {user_question}

위 문맥을 바탕으로 질문에 답변해주세요."""

# - model: Groq에서 제공하는 Llama 계열 모델 지정
# - temperature=0.3: 낮게 해서 답변 일관성/보수성 강화(문맥 기반 QA에 유리)
# - max_tokens: 응답 길이 상한
chat_completion = groq_client.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024
)


# --------------------------------------------
# 10) 결과 출력
# --------------------------------------------
# - 검색된 문서와 거리(distance)를 출력하고
# - 최종 AI 답변을 출력합니다.
#
# 주의: Chroma의 distance는 설정된 distance metric에 따라 의미가 달라집니다.
#      흔히 코사인 거리라면 0에 가까울수록 유사합니다.
#      아래처럼 (1 - distance)를 유사도처럼 출력하는 건 “코사인 거리”일 때만 직관적입니다.
print("\n" + "="*50)
print("검색된 관련 문서:")
print("="*50)

for i, (doc, distance) in enumerate(zip(retrieved_docs, results['distances'][0])):
    print(f"\n{i+1}. 유사도: {distance:.4f}")
    print(f"   문서: {doc}")

print("\n" + "="*50)
print("AI 답변:")
print("="*50)

# - Groq 응답의 choices[0]에 첫 번째 답변이 들어있습니다.
print(chat_completion.choices[0].message.content)
