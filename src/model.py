import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.config import RANDOM_STATE


def build_model(input_dim: int, learning_rate: float = 0.001) -> keras.Model:
    """Build a regularized neural network for binary classification."""
    tf.random.set_seed(RANDOM_STATE)

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.30),

            layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(0.20),

            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model
