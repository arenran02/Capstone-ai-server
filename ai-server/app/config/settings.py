from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # MongoDB
    mongo_user: str
    mongo_pass: str
    mongo_host: str = "imagevector.wonp5c6.mongodb.net"
    mongo_db: str = "dino_db"
    mongo_collection: str = "item_vectors"

    # 모델 경로
    clip_model_name: str = "ViT-B/32"
    finetuned_clip_path: str = "app/core/clip_finetuned.pth"
    category_json_path: str = "dataset/category_to_products.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings():
    return Settings()
