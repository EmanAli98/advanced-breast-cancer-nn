import numpy as np
from src.data import load_dataset, split_and_scale


def test_load_dataset_shape():
    df = load_dataset()
    assert df.shape == (569, 31)  # 30 features + label
    assert "label" in df.columns


def test_split_and_scale_sizes():
    df = load_dataset()
    splits = split_and_scale(df)

    n_total = len(df)
    n_train = len(splits["y_train"])
    n_val = len(splits["y_val"])
    n_test = len(splits["y_test"])

    assert n_train + n_val + n_test == n_total
    # test_size=0.15, validation_size=0.15 (from config.py)
    assert abs(n_test / n_total - 0.15) < 0.02
    assert abs(n_val / n_total - 0.15) < 0.02


def test_scaling_is_standardized():
    df = load_dataset()
    splits = split_and_scale(df)
    X_train_scaled = splits["X_train_scaled"]
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-1)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-1)


def test_stratified_split_preserves_class_balance():
    df = load_dataset()
    splits = split_and_scale(df)

    full_ratio = df["label"].mean()
    train_ratio = splits["y_train"].mean()
    val_ratio = splits["y_val"].mean()
    test_ratio = splits["y_test"].mean()

    assert abs(train_ratio - full_ratio) < 0.05
    assert abs(val_ratio - full_ratio) < 0.05
    assert abs(test_ratio - full_ratio) < 0.05


def test_feature_names_match_columns():
    df = load_dataset()
    splits = split_and_scale(df)
    expected = [c for c in df.columns if c != "label"]
    assert splits["feature_names"] == expected
