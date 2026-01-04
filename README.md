# temp_rag# hAIr (Hair Artificial intelligence Individual Recommend) - Python Project

## 환경 구성 명령어
``` MarkDown
### Conda 환경 생성
conda create -n iam_proj python=3.10

### Conda 환경 활성화
conda activate iam_proj

### PyTorch 및 관련 패키지 설치
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

### MKL 설치
conda install mkl

### 의존성 설치
pip install -r requirements.txt 
```

## 폴더 경로
```
📁HAIR_PYTHON
    📄
```

## AI Model

## 주요 기능

### 사용자 인터페이스 및 챗봇 (프론트엔드)

1. 랭체인 기반 미용사 챗봇과 대화
    - **기술 스택**: LangChain, RAG, ChromaDB
2. 사용자 얼굴 이미지 업로드 또는 실시간 촬영
3. 얼굴형 분석 진행
    - **기술 스택**: Python, FastAPI, OpenCV, Dlib
4. 선호하는 헤어스타일 입력
    - **기술 스택**: Python, FastAPI, ChromaDB, LangChain, RAG
5. 추천된 스타일 목록 제공
6. 원하는 이미지 두 가지 선택 (헤어스타일 이미지, 헤어 색상 이미지)

### 데이터 계획

- **데이터 수집**: 한국인 헤어스타일 이미지 데이터, CelebA 데이터셋 등 오픈 데이터와 크롤링, 미용실과의 협력
- **데이터 가공 및 라벨링**: 이미지 전처리, 얼굴 검출 및 특징 추출, 스타일 및 퍼스널 컬러 라벨링
- **데이터 저장**: S3, Google Cloud Storage에 이미지 데이터 저장, ChromaDB에 메타데이터 저장