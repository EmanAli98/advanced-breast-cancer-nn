import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_nn.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Breast Cancer NN Classifier", layout="wide")
st.title("Breast Cancer Classification with Neural Network")
st.warning("Educational demo only. Not a medical diagnostic tool.")

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, feature_names, metadata

if not MODEL_PATH.exists():
    st.error("Model files not found. Run `python -m src.train` first.")
    st.stop()

model, scaler, feature_names, metadata = load_artifacts()
threshold = float(metadata.get("best_threshold", 0.5))

st.sidebar.header("Input features")
st.sidebar.write("Use the sliders to enter the 30 numeric features.")

values = []
for feature in feature_names:
    values.append(st.sidebar.number_input(feature, value=0.0, format="%.5f"))

if st.button("Predict"):
    X = pd.DataFrame([values], columns=feature_names)
    X_scaled = scaler.transform(X)
    probability_benign = float(model.predict(X_scaled).ravel()[0])
    predicted_label = int(probability_benign >= threshold)

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability benign", f"{probability_benign:.3f}")
    col2.metric("Threshold", f"{threshold:.3f}")
    col3.metric("Prediction", "Benign" if predicted_label == 1 else "Malignant")

st.subheader("Model metadata")
st.json(metadata)
