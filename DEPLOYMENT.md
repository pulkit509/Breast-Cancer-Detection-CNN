# Streamlit dashboard

The app displays one ultrasound image, the predicted class, three softmax probabilities, and a downloadable CSV. It preserves the notebook's OpenCV BGR decoding, default 128×128 resize, and 0–255 scale. It does not change or retrain the CNN.

## Export your trained model

After `model.fit(...)` in Colab, run:

```python
model.save("breast_cancer_cnn.keras")
from google.colab import files
files.download("breast_cancer_cnn.keras")
```

Add that downloaded file beside `streamlit_app.py` on the branch you deploy. Only use the model trained by this repository with class order `normal`, `benign`, `malignant`. The dashboard checks shape and softmax output but cannot infer class semantics from the weights. Do not upload images, dataset archives, or credentials to GitHub.

If GitHub's browser upload limit rejects the model, use Git LFS from your local checkout; Streamlit Community Cloud supports LFS:

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes breast_cancer_cnn.keras
git commit -m "Add trained ultrasound CNN"
git push
```

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ with your GitHub account.
2. Create an app and select repository `pulkit509/Breast-Cancer-Detection-CNN`.
3. Select branch `streamlit-dashboard` (or `main` after merging this branch).
4. Set the entrypoint to `streamlit_app.py` and choose Python **3.11** in Advanced settings.
5. Deploy. Wait for the build to finish, then open the URL Streamlit assigns.

You can launch the interface before adding the model, but prediction remains disabled. No live deployment or real-model prediction has been verified yet. The account owner must complete the hosting sign-in; no hosting credentials are included in this repository.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Local run and checks

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
streamlit run streamlit_app.py
```

The tests compare uploaded-image decoding with `cv2.imread` on the same bytes and check preprocessing, invalid inputs, and output validation. Tests requiring missing packages are explicitly skipped. They do not measure model accuracy.

Before sharing the app, load your actual exported model and compare its predictions for the same image in Colab and the dashboard. The dependency ranges are not a lockfile; if loading fails, match TensorFlow/Keras to the export environment and pin the verified versions. An administrator may alternatively set the `MODEL_PATH` environment variable to a trusted `.keras` file on the server.

## Interpretation and data handling

This is a research/learning demonstration, not a diagnostic tool. Softmax probabilities are not calibrated clinical risk estimates. Deployment performance has not been measured.

Images are sent to the hosting server for processing. The app does not explicitly save uploaded images or cache predictions. Use de-identified images only and consider the host's data policies before sharing a public app. The model is administrator-provided; visitors cannot upload executable model files.
