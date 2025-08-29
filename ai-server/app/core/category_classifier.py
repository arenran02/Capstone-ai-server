# app/core/category_classifier.py

import torch
import torch.nn as nn
import clip
import os
from PIL import Image

CATEGORY_TO_LABEL = {
    "전자기기": 0,
    "지갑": 1,
    "카드": 2,
    "악세서리": 3,
    "의류": 4,
    "가방": 5,
    "책": 6,
    "텀블러": 7,
    "기타": 8
}
LABEL_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_LABEL.items()}

class CLIPCategoryClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model  # CLIP backbone
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, image):
        with torch.no_grad():
            features = self.clip_model.encode_image(image)
        return self.mlp(features)

def load_classifier_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.visual  # 인코더만 사용

    class MLPClassifier(nn.Module):
        def __init__(self, input_dim=512, num_classes=9):  # 클래스 수 9로 수정
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            return self.classifier(x)

    class CLIPWithMLP(nn.Module):
        def __init__(self, encoder, classifier):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, image):
            with torch.no_grad():
                x = self.encoder(image)
            return self.classifier(x)

    classifier = MLPClassifier(num_classes=len(CATEGORY_TO_LABEL)).to(device)
    model = CLIPWithMLP(clip_model, classifier).to(device)

    model_path = os.path.join(os.path.dirname(__file__), "clip_with_mlp_9.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    return model, preprocess, device


def predict_category(image, classifier_model, preprocess, device):
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier_model(image_tensor)
        predicted_label = torch.argmax(logits, dim=1).item()

    category = LABEL_TO_CATEGORY.get(predicted_label, "기타")
    return category, predicted_label