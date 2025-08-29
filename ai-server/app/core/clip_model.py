import torch
import clip
from PIL import Image
import os
import json

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    checkpoint_path = os.path.join(os.path.dirname(__file__), "clip_finetuned.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, preprocess, device

def load_word_dict(relative_path):
    base_dir = os.path.dirname(os.path.dirname(__file__))  # app/core → app
    abs_path = os.path.join(base_dir, relative_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_top_n(image, word_list, model, preprocess, device, top_n=3):
    text_inputs = clip.tokenize(word_list).to(device)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities = (image_features @ text_features.T).squeeze(0)
    top_probs, top_labels = similarities.topk(top_n)

    results = [(word_list[idx], top_probs[i].item()) for i, idx in enumerate(top_labels)]
    return results
