# TEMP_RAG
> 스터디에서 진행하는 AI 프로젝트 진행하기 전에 수행하는 간단한 프로젝트

## 목적
RAG의 기본적인 문서 검색 방식을 습득한다.

## Action Item
- [ ] RAG에 사용할 데이터 셋을 준비한다.
- [ ] 임베딩 모델을 사용해서 Vector DB에 저장한다.
- [ ] 쿼리를 사용해서 가장 유사한 문장을 검색한다.

# 환경 구성 명령어

### Conda 환경 리스트 검색
```bash
conda env list
```

### Conda 환경 생성
```bash
conda create -n temp_rag python=3.10
```

### Conda 환경 활성화
```bash
conda activate env list
```

### 의존성 설치
```bash
pip install -r requirements.txt 
```
### 현재 환경의 패키지 목록 저장
```bash
pip freeze > requirements.txt
```