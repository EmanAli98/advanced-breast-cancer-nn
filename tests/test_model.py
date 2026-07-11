import numpy as np
from src.model import build_model


def test_build_model_output_shape():
    model = build_model(input_dim=30)
    dummy_input = np.random.rand(5, 30)
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (5, 1)
    assert np.all((output >= 0) & (output <= 1))  # valid sigmoid probabilities


def test_build_model_is_compiled_correctly():
    model = build_model(input_dim=30)
    assert model.loss == "binary_crossentropy"
    # Keras 3 nests the named metrics (accuracy, precision, ...) inside a
    # CompileMetrics wrapper rather than exposing them directly on
    # model.metrics, so unwrap it.
    dummy_X = np.random.rand(2, 30)
    dummy_y = np.array([0, 1])
    model.evaluate(dummy_X, dummy_y, verbose=0)

    metric_names = set()
    for m in model.metrics:
        if hasattr(m, "metrics"):  # CompileMetrics wrapper
            metric_names.update(sub.name for sub in m.metrics)
        else:
            metric_names.add(m.name)

    assert {"accuracy", "precision", "recall", "auc"}.issubset(metric_names)


def test_build_model_respects_input_dim():
    model = build_model(input_dim=30)
    assert model.input_shape == (None, 30)
