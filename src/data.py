import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE


def load_dataset() -> pd.DataFrame:
    """Load the Scikit-learn Breast Cancer dataset as a DataFrame."""
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["label"] = dataset.target
    return df


def split_and_scale(df: pd.DataFrame):
    """Create stratified train/validation/test splits and scale features."""
    X = df.drop(columns="label")
    y = df["label"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_relative_size = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_relative_size,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }
