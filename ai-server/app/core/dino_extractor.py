import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
import torch
from app.utils.download_image import download_image
import io

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = Dinov2Model.from_pretrained("facebook/dinov2-base")
model.eval()

def extract_vector(pil_img: Image.Image) -> np.ndarray:
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return vec

def extract_vectors_from_urls(urls):
    vectors = []
    for url in urls:
        img = download_image(url)
        if img:
            vec = extract_vector(img)
            vectors.append(vec)

    if not vectors:
        raise ValueError("No valid vectors extracted.")

    vectors = np.array(vectors)
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sim_matrix = normed @ normed.T
    attention_scores = np.mean(sim_matrix, axis=1)
    attention_weights = attention_scores / np.sum(attention_scores)
    weighted_vector = np.sum(vectors * attention_weights[:, np.newaxis], axis=0)

    return vectors, weighted_vector
