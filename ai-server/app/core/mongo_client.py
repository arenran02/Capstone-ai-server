from pymongo import MongoClient
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
import numpy as np

load_dotenv()

username = os.getenv("MONGO_USER")
password = quote_plus(os.getenv("MONGO_PASS"))
uri = f"mongodb+srv://{username}:{password}@imagevector.wonp5c6.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)
collection = client["dino_db"]["item_vectors"]

def save_item_vector(item_id: str, vector: list[float], category: str, item_type: str):
    collection.update_one(
        {"item_id": item_id},
        {"$set": {"vector": vector, "category": category, "type": item_type}},
        upsert=True
    )

def find_matched_items(query_vector: np.ndarray, category: str, item_type: str, threshold=0.3, top_k=3):
    opposite_type = "습득물" if item_type == "분실물" else "분실물"
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "vector",
                "queryVector": query_vector.tolist(),
                "numCandidates": 100,
                "limit": top_k
            }
        },
        {
            "$match": {
                "type": opposite_type
            }
        },
        {
            "$project": {
                "item_id": 1,
                "similarity": {"$meta": "vectorSearchScore"},
                "_id": 0
            }
        }
    ]

    results = list(collection.aggregate(pipeline))

    return [
        {"item_id": doc["item_id"], "similarity": doc.get("similarity", 0.0)}
        for doc in results if doc.get("similarity", 0.0) >= threshold
    ]
