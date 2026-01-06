from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# 1. 모델 로드
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 2. 샘플 문서 준비
documents = [
    "파이썬은 데이터 분석에 매우 유용한 프로그래밍 언어입니다.",
    "머신러닝 모델을 학습시키기 위해서는 많은 데이터가 필요합니다.",
    "자연어 처리는 인공지능의 중요한 분야 중 하나입니다.",
    "벡터 데이터베이스는 유사도 검색에 최적화되어 있습니다.",
    "한국어와 영어를 모두 지원하는 다국어 모델입니다."
]

# 3. ChromaDB 클라이언트 생성
client = chromadb.PersistentClient(path="./chroma_db")

# 4. 컬렉션 생성 (임베딩 함수 지정)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="my_documents",
    embedding_function=sentence_transformer_ef
)

if collection.count() == 0:
    documents = [
        "파이썬은 데이터 분석에 매우 유용한 프로그래밍 언어입니다.",
        "머신러닝 모델을 학습시키기 위해서는 많은 데이터가 필요합니다.",
        "자연어 처리는 인공지능의 중요한 분야 중 하나입니다.",
    ]
    
    collection.add(
        documents=documents,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    print("새로운 데이터 추가 완료")
else:
    print(f"기존 컬렉션 사용 (문서 수: {collection.count()})")

# 5. 검색 수행
query = "자연어 처리 기술"
results = collection.query(query_texts=[query], n_results=2)
print(results['documents'][0])

# 7. 결과 출력
print(f"쿼리: {query}\n")
print("유사한 문서들:")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"\n{i+1}. 유사도: {1 - distance:.4f}")
    print(f"   문서: {doc}")
