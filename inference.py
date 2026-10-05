"""Inference contract shared by the dashboard and its tests."""
import numpy as np

CLASS_NAMES = ("normal", "benign", "malignant")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def preprocess_image(content):
    """Return preview and batch matching the notebook's unscaled BGR input."""
    import cv2

    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose an image between 1 byte and 10 MB.")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("This file could not be read as an image. Try PNG or JPEG.")
    resized = cv2.resize(image, (128, 128))
    return image, np.expand_dims(resized, axis=0)


def validate_model(model):
    if model.input_shape != (None, 128, 128, 3):
        raise ValueError("The model must accept images with shape (128, 128, 3).")
    if model.output_shape != (None, 3):
        raise ValueError("The model must output three class probabilities.")
    if getattr(getattr(model.layers[-1], "activation", None), "__name__", None) != "softmax":
        raise ValueError("The model must have the notebook's final softmax layer.")


def validate_probabilities(output):
    probabilities = np.asarray(output)
    if probabilities.shape != (1, 3):
        raise ValueError("Expected one prediction containing three probabilities.")
    probabilities = probabilities[0]
    if (not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-5, rtol=0)):
        raise ValueError("The model returned invalid softmax probabilities.")
    return probabilities
