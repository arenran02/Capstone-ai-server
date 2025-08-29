from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.core.dino_extractor import extract_vectors_from_urls
from app.core.mongo_client import save_item_vector, find_matched_items

router = APIRouter()

class MatchRequest(BaseModel):
    image_urls: List[str]
    item_id: int
    category: str
    type: str
    threshold: Optional[float] = 0.3
    top_k: Optional[int] = 3

@router.post("/match-items")
async def match_items(request: MatchRequest):
    try:
        vectors, item_vector = extract_vectors_from_urls(request.image_urls)

        save_item_vector(
            item_id=request.item_id,
            vector=item_vector.tolist(),
            category=request.category,
            item_type=request.type
        )

        matched_items = find_matched_items(
            query_vector=item_vector,
            category=request.category,
            item_type=request.type,
            threshold=request.threshold,
            top_k=request.top_k
        )

        matched_items = [item for item in matched_items if item["item_id"] != request.item_id]

        return {
            "item_id": request.item_id,
            "matched_items": matched_items
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
