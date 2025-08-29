# AI 유실물 관리 시스템 (Capstone Design 2025-1)

경찰청 유실물 포털 Open API로 유실물 데이터를 수집·정제하고,  
수집한 데이터셋으로 CLIP을 파인튜닝하여 `입력 이미지 → 유사한 물건 이름/설명`을 추론합니다.  
또한 DINOv2로 추출한 이미지 임베딩을 **MongoDB(벡터 인덱스)** 에 저장해 Top-K 이미지 매칭 & 알림을 제공합니다.  
백엔드는 FastAPI, 벡터 검색은 MongoDB Atlas Vector Search(또는 로컬 MongoDB + 벡터 인덱스) 기반입니다.  

## 🏗️ 아키텍처 개요
<img width="2112" height="1048" alt="image" src="https://github.com/user-attachments/assets/d9548218-bdbd-4020-aae4-e5dd66687f59" />


## ✨ 주요 기능

- 데이터 수집: 경찰청 유실물 포털 Open API에서 카테고리/기간 기반 배치 수집

- 데이터 정제: 중복 제거, 텍스트 정규화, 카테고리 매핑, 이미지 품질 필터링

- 학습(파인튜닝): CLIP(이미지–텍스트) 파인튜닝으로 물건명/설명 매칭 성능 향상

- 임베딩 추출: DINOv2(b / l 등 선택)로 이미지 벡터화

- 벡터 저장/검색: MongoDB에 벡터 및 메타데이터 저장 → Top-K 검색

- 알림: 사용자가 등록한 분실물 조건과 유사도 임계치 충족 시 웹훅/이메일 알림

## 📂 디렉터리 구조

```bash
ai-server/
├─ app/
│  ├─ api/
│  │  ├─ describe.py            # 이미지→물건 이름/설명 추론 REST API
│  │  └─ matching.py            # Top-K 이미지 매칭/검색 REST API
│  ├─ config/
│  │  └─ settings.py            # 환경변수/설정(Pydantic Settings 등)
│  ├─ core/
│  │  ├─ category_classifier.py  # 카테고리 분류기(옵션)
│  │  ├─ clip_model.py          # CLIP 로딩/파인튜닝 관련 유틸
│  │  ├─ dino_extracter.py      # DINOv2 임베딩 추출
│  │  ├─ init.py                # 모델/클라이언트 초기화 훅
│  │  └─ mongo_client.py        # MongoDB 연결 및 벡터 검색 유틸
│  ├─ dataset/
│  │  ├─ category_to_products.json
│  │  ├─ final_dataset.csv
│  │  ├─ final_dataset_updated.csv
│  │  └─ product_mapping.json
│  └─ utils/
│     ├─ download_image.py      # 이미지 다운로드 유틸
│     └─ __init__.py
├─ app/main.py                  # FastAPI 엔트리 포인트 (app 객체)
├─ .env                         # 환경 변수(로컬)
├─ .example                     # 예시 파일 (옵션)
├─ requirements.txt             # pip 의존성
└─ .gitignore

```

## 실행 방법

```bash
# conda 환경
conda create -n lostfound-ai python=3.10 -y
conda activate lostfound-ai
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```


