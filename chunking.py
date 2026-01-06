from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 최대 시퀀스 길이 확인
print(f"Max sequence length: {model.max_seq_length}")  # 128

# 긴 문서 예시
long_document = "매우 긴 문서 내용..." * 100

# 방법 1: 고정 크기 청킹 (단순하지만 문맥 손실 가능)
def chunk_by_tokens(text, tokenizer, max_tokens=100, overlap=20):
    """토큰 기반 청킹"""
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

# 방법 2: 문장 단위 청킹 (의미 보존)
def chunk_by_sentences(text, model, max_tokens=100):
    """문장 단위로 청킹하되 토큰 제한 준수"""
    import re
    
    # 문장 분리 (한국어 고려)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # 문장 토큰 수 계산
        sentence_tokens = len(model.tokenizer.encode(sentence))
        
        if current_length + sentence_tokens > max_tokens:
            # 현재 청크 저장
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # 마지막 청크
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# 사용 예시
chunks = chunk_by_sentences(long_document, model, max_tokens=100)
print(f"총 청크 수: {len(chunks)}")
