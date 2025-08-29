# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.init import initialize_models
from app.api import describe, matching

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 초기화
models = initialize_models()

# 모델 정보 라우터에 전달
describe.set_model_context(models)

app.include_router(describe.router, prefix="/api/v1")
app.include_router(matching.router, prefix="/api/v1")
