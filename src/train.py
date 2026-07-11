import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.config import FEATURE_NAMES_PATH, METADATA_PATH, MODEL_PATH, SCALER_PATH
from src.data import load_dataset, split_and_scale
from src.evaluate import evaluate_and_plot, find_best_threshold
from src.model import build_model


def train():
    df = load_dataset()
    splits = split_and_scale(df)

    # Baseline model for comparison
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(splits["X_train_scaled"], splits["y_train"])
    baseline_prob = baseline.predict_proba(splits["X_test_scaled"])[:, 1]
    baseline_pred = (baseline_prob >= 0.5).astype(int)

    baseline_metrics = {
        "accuracy": accuracy_score(splits["y_test"], baseline_pred),
        "precision": precision_score(splits["y_test"], baseline_pred),
        "recall": recall_score(splits["y_test"], baseline_pred),
        "f1": f1_score(splits["y_test"], baseline_pred),
        "roc_auc": roc_auc_score(splits["y_test"], baseline_prob),
    }

    # Class weights help when classes are not perfectly balanced
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(splits["y_train"]),
        y=splits["y_train"],
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

    model = build_model(input_dim=splits["X_train_scaled"].shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=20,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        splits["X_train_scaled"],
        splits["y_train"],
        validation_data=(splits["X_val_scaled"], splits["y_val"]),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    val_prob = model.predict(splits["X_val_scaled"]).ravel()
    best_threshold, best_f1 = find_best_threshold(splits["y_val"], val_prob)

    test_prob = model.predict(splits["X_test_scaled"]).ravel()
    nn_metrics = evaluate_and_plot(
        splits["y_test"],
        test_prob,
        threshold=best_threshold,
        prefix="test",
    )

    metadata = {
        "dataset_rows": int(df.shape[0]),
        "dataset_features": int(len(splits["feature_names"])),
        "target_mapping": {"0": "malignant", "1": "benign"},
        "best_threshold": best_threshold,
        "best_validation_f1": best_f1,
        "baseline_logistic_regression": baseline_metrics,
        "neural_network": nn_metrics,
        "epochs_trained": len(history.history["loss"]),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(splits["scaler"], SCALER_PATH)
    joblib.dump(splits["feature_names"], FEATURE_NAMES_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Training complete.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train()
