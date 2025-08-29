from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import io
from PIL import Image
import torch
import json
from app.core.clip_model import load_model, load_word_dict, predict_top_n
from app.core.category_classifier import CATEGORY_TO_LABEL, LABEL_TO_CATEGORY
from pathlib import Path 

router = APIRouter()

model = preprocess = device = None
classifier_model = classifier_preprocess = classifier_device = None
word_dict = None
mapping_dict={}

def set_model_context(models):
    global model, preprocess, device
    global classifier_model, classifier_preprocess, classifier_device
    global word_dict, mapping_dict

    model = models["clip"]["model"]
    preprocess = models["clip"]["preprocess"]
    device = models["clip"]["device"]

    classifier_model = models["classifier"]["model"]
    classifier_preprocess = models["classifier"]["preprocess"]
    classifier_device = models["classifier"]["device"]

    word_dict = models["word_dict"]

    BASE_DIR = Path(__file__).resolve().parent  # 현재 파일인 describe.py의 위치 (app/api)
    mapping_path = BASE_DIR.parent / "dataset" / "product_mapping.json"

    with open(mapping_path, encoding="utf-8") as f:
        mapping_dict = json.load(f)

class ImageRequest(BaseModel):
    image_urls: List[str]

class ImageDescription(BaseModel):
    name: str
    description: str
    category: str

from app.core.category_classifier import predict_category

@router.post("/describe-item", response_model=ImageDescription)
async def describe_image(request: ImageRequest):
    if not request.image_urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")

    image_url = request.image_urls[0]

    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image")

        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes).convert("RGB")

        # 카테고리 예측 (전역에서 로드한 MLP 모델 사용)
        category, _ = predict_category(image, classifier_model, classifier_preprocess, classifier_device)

        if category not in word_dict:
            raise HTTPException(status_code=400, detail=f"Invalid category '{category}'")

        # 상품 이름 예측 (기존 CLIP 모델)
        word_list = word_dict[category]
        results = predict_top_n(image, word_list, model, preprocess, device, top_n=1)
        top_name, score = results[0]

        korean_name = mapping_dict.get(top_name, {}).get("korean_name", top_name)

        return {
            "name": korean_name,  # 한국어 이름 반환
            "description": f"카테고리: {category}(으)로 분류된 {top_name}",
            "category": category
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))