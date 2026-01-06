import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re

class SmartDocumentStore:
    def __init__(self, db_path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="smart_documents",
            embedding_function=self.embedding_function
        )
        
        print(f"최대 시퀀스 길이: {self.model.max_seq_length} 토큰")
    
    def smart_chunk(self, text, max_tokens=100, overlap_tokens=20):
        """문장 경계를 고려한 스마트 청킹"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_tokens = []
        current_count = 0
        
        for sentence in sentences:
            tokens = self.model.tokenizer.encode(sentence, add_special_tokens=False)
            token_count = len(tokens)
            
            if current_count + token_count > max_tokens and current_tokens:
                # 청크 완성
                chunk_text = self.model.tokenizer.decode(current_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
                
                # 오버랩 적용
                current_tokens = current_tokens[-overlap_tokens:] if len(current_tokens) > overlap_tokens else []
                current_count = len(current_tokens)
            
            current_tokens.extend(tokens)
            current_count += token_count
        
        # 마지막 청크
        if current_tokens:
            chunk_text = self.model.tokenizer.decode(current_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        return chunks
    
    def add_file(self, file_path):
        """파일을 토큰 기반으로 청킹하여 저장"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self.smart_chunk(content, max_tokens=100, overlap_tokens=20)
        
        file_name = Path(file_path).stem
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        
        # 각 청크의 실제 토큰 수 확인
        metadatas = [
            {
                "source": file_path,
                "chunk_index": i,
                "tokens": len(self.model.tokenizer.encode(chunk))
            }
            for i, chunk in enumerate(chunks)
        ]
        
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        
        token_counts = [m["tokens"] for m in metadatas]
        print(f"파일 '{file_name}': {len(chunks)}개 청크 (평균 {sum(token_counts)/len(token_counts):.1f} 토큰)")
        
        return chunks

# 사용
store = SmartDocumentStore()
store.add_file("python-3.10.19-docs-text\\c-api\\abstract.txt")
results = store.collection.query(query_texts=["추상객체란 추상객체이다."], n_results=3)
print(results)