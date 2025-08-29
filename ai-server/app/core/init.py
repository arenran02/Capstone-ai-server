# app/core/init.py
from app.core.clip_model import load_model as load_clip_model, load_word_dict
from app.core.category_classifier import load_classifier_model

def initialize_models():
    clip_model, clip_preprocess, clip_device = load_clip_model()
    classifier_model, classifier_preprocess, classifier_device = load_classifier_model()
    word_dict = load_word_dict("./dataset/category_to_products.json")

    return {
        "clip": {
            "model": clip_model,
            "preprocess": clip_preprocess,
            "device": clip_device,
        },
        "classifier": {
            "model": classifier_model,
            "preprocess": classifier_preprocess,
            "device": classifier_device,
        },
        "word_dict": word_dict,
    }
