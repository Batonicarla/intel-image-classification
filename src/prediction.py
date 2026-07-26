import numpy as np
from src.preprocessing import preprocess_single_image, CLASSES
from src.model import load_trained_model

_cached_model = None
_cached_model_path = None


def get_model(model_path):
    """Simple in-memory cache so the API doesn't reload the model from disk on every request."""
    global _cached_model, _cached_model_path
    if _cached_model is None or _cached_model_path != model_path:
        _cached_model = load_trained_model(model_path)
        _cached_model_path = model_path
    return _cached_model


def predict_image(img_path, model_path, class_indices=None):
    """
    Predicts the class of a single image.
    class_indices: dict like {'buildings': 0, 'forest': 1, ...} loaded from class_indices.json
    """
    model = get_model(model_path)
    img_array = preprocess_single_image(img_path)

    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if class_indices:
        idx_to_class = {v: k for k, v in class_indices.items()}
        label = idx_to_class[class_idx]
    else:
        label = CLASSES[class_idx]

    all_probs = {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}

    return {
        "predicted_class": label,
        "confidence": confidence,
        "all_probabilities": all_probs
    }
