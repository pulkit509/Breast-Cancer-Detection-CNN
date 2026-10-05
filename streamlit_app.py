"""Streamlit interface for the trained BUSI ultrasound CNN."""
import os
from pathlib import Path

# Restrict CPU thread use for small hosting instances before TensorFlow loads.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import pandas as pd
import streamlit as st

from inference import CLASS_NAMES, MAX_UPLOAD_BYTES, preprocess_image, validate_model, validate_probabilities

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ROOT / "breast_cancer_cnn.keras")))

st.set_page_config(page_title="Ultrasound CNN | Image explorer", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner=False)
def load_cnn(path, modified_time):
    # Only the repository owner's model is loaded, never a visitor-uploaded model.
    from tensorflow.keras.models import load_model
    model = load_model(path, compile=False, safe_mode=True)
    validate_model(model)
    return model


with st.sidebar:
    st.title("Ultrasound CNN")
    st.caption("BREAST IMAGE CLASSIFICATION")
    st.divider()
    st.markdown("**Three classes**\n\nNormal · Benign · Malignant")
    st.markdown("**How to use**\n\n1. Upload an ultrasound image.\n2. Select Predict.\n3. Explore the three probabilities.")
    st.divider()
    st.caption("Research and learning demo. Not a medical diagnosis.")

st.caption("IMAGE EXPLORER / CNN DEMO")
st.title("Explore an ultrasound image")
st.write("Upload one image to view your trained CNN’s prediction and its probability distribution.")
st.info("Use de-identified ultrasound images only. Predictions on new uploads have not been clinically validated.")

prediction_tab, details_tab = st.tabs(["Image prediction", "About the model"])

with details_tab:
    st.subheader("The same CNN, the same inputs")
    st.write("This dashboard loads the model exported from the project notebook. It does not retrain or change its architecture.")
    st.dataframe(pd.DataFrame({
        "Setting": ["Image decoding", "Input size", "Pixel values", "Class order", "Output"],
        "Value": ["OpenCV BGR, 3 channels", "128 × 128", "0–255, no normalization", "normal, benign, malignant", "3 softmax probabilities"],
    }), hide_index=True, use_container_width=True)
    st.caption("The preview is converted to RGB for display only. The model receives BGR pixels.")
    st.subheader("How to read the results")
    st.write("The predicted class has the largest softmax value. These values describe the model’s output; they are not calibrated clinical risk estimates.")
    st.write("No deployment accuracy is reported. A single uploaded image without a verified label cannot measure accuracy.")

with prediction_tab:
    ready = MODEL_PATH.is_file()
    if not ready:
        st.warning("Model not configured yet. Image preview is available; predictions are disabled.")
        st.caption("Owner setup: add breast_cancer_cnn.keras beside streamlit_app.py. See DEPLOYMENT.md in the repository.")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("1. Your image")
        upload = st.file_uploader("Upload an ultrasound image", type=["png", "jpg", "jpeg", "bmp"], help="One image, up to 10 MB. Do not upload a segmentation mask.")
        batch = None
        if upload is not None:
            try:
                if upload.size > MAX_UPLOAD_BYTES:
                    raise ValueError("Choose an image no larger than 10 MB.")
                preview, batch = preprocess_image(upload.getvalue())
                st.image(preview[:, :, ::-1], caption="Uploaded image · preview only", use_container_width=True)
                st.caption(f"Original size: {preview.shape[1]} × {preview.shape[0]} pixels")
            except ValueError as error:
                st.error(str(error))
        else:
            st.info("Your image preview will appear here.")

    with right:
        st.subheader("2. Model result")
        run = st.button("Predict image", type="primary", disabled=batch is None or not ready, use_container_width=True)
        if run:
            try:
                with st.spinner("Running your CNN…"):
                    model = load_cnn(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
                    probabilities = validate_probabilities(model.predict(batch, verbose=0))
                predicted = CLASS_NAMES[int(np.argmax(probabilities))]
                st.metric("Predicted class", predicted.title())
                st.caption("Softmax probabilities · fixed training class order")
                for name, probability in zip(CLASS_NAMES, probabilities):
                    st.write(f"**{name.title()}** — {probability:.2%}")
                    st.progress(float(probability))
                results = pd.DataFrame({"class": CLASS_NAMES, "softmax_probability": probabilities})
                st.download_button("Download probabilities (CSV)", results.to_csv(index=False), "prediction.csv", "text/csv")
                st.caption("This output is a model prediction, not a diagnosis.")
            except Exception:
                st.error("Prediction could not be completed. The owner should verify the saved model and its TensorFlow/Keras version. No result was produced.")
        elif batch is not None and ready:
            st.info("Image ready. Select Predict image to run the saved CNN.")
        else:
            st.caption("A prediction will appear after a model and valid image are available.")
