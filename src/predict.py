import json
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import FEATURE_NAMES_PATH, METADATA_PATH, MODEL_PATH, SCALER_PATH


def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, feature_names, metadata


def predict_single(input_values):
    model, scaler, feature_names, metadata = load_artifacts()

    if len(input_values) != len(feature_names):
        raise ValueError(f"Expected {len(feature_names)} features, got {len(input_values)}")

    X = pd.DataFrame([input_values], columns=feature_names)
    X_scaled = scaler.transform(X)
    probability_benign = float(model.predict(X_scaled).ravel()[0])
    threshold = float(metadata.get("best_threshold", 0.5))
    predicted_label = int(probability_benign >= threshold)

    label_name = "Benign" if predicted_label == 1 else "Malignant"

    return {
        "probability_benign": probability_benign,
        "threshold": threshold,
        "predicted_label": predicted_label,
        "prediction": label_name,
    }


if __name__ == "__main__":
    # Example from the original notebook
    example_input = [
        11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115,
        0.1495, 0.05888, 0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758,
        0.01168, 0.007445, 0.02406, 0.001769, 12.98, 25.72, 82.98, 516.5,
        0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563,
    ]

    result = predict_single(example_input)
    print(json.dumps(result, indent=2))
