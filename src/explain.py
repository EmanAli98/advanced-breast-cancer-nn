"""
SHAP explainability for the trained neural network.

Produces:
- A global summary plot: which of the 30 features matter most overall,
  and whether high/low values push predictions toward malignant or benign.
- A per-prediction waterfall plot: why the model predicted what it did for
  one specific test sample. In a medical-adjacent context, "the model said
  so" isn't an acceptable answer on its own — this shows the reasoning.

Run after training:
    python -m src.explain
"""
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
from tensorflow import keras

from src.config import FEATURE_NAMES_PATH, MODEL_PATH, REPORT_DIR, SCALER_PATH
from src.data import load_dataset, split_and_scale


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `python -m src.train` first.")

    model = keras.models.load_model(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    # Recreate the same test split used during training (same RANDOM_STATE
    # in config.py), so the explanations line up with the metrics in
    # metadata.json / the evaluation report.
    df = load_dataset()
    splits = split_and_scale(df)
    X_test_scaled = splits["X_test_scaled"]

    # A small background sample keeps SHAP fast — KernelExplainer/
    # PermutationExplainer both scale poorly with a large background set.
    rng = np.random.default_rng(seed=42)
    background = X_test_scaled[rng.choice(X_test_scaled.shape[0], size=50, replace=False)]

    explainer = shap.Explainer(model, background, feature_names=feature_names)

    sample = X_test_scaled[:100]
    shap_values = explainer(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    summary_path = REPORT_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {summary_path}")

    # Explain a single prediction (first test sample). Rebuilding the
    # Explanation object by hand avoids a shape mismatch that SHAP's own
    # slicing hits for single-sigmoid-output models.
    raw_explanation = explainer(X_test_scaled[:1])
    single_explanation = shap.Explanation(
        values=raw_explanation.values[0].flatten(),
        base_values=float(raw_explanation.base_values.flatten()[0]),
        data=raw_explanation.data[0],
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(single_explanation, show=False)
    plt.tight_layout()
    single_path = REPORT_DIR / "shap_single_prediction.png"
    plt.savefig(single_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved single-prediction SHAP plot -> {single_path}")


if __name__ == "__main__":
    main()
