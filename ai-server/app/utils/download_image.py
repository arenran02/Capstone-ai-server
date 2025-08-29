import requests
from PIL import Image
import io

def download_image(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except:
        return None
