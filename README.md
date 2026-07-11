# Breast Cancer Classification with a Regularized Neural Network

> Educational machine learning project built on the Scikit-learn Breast Cancer Wisconsin dataset.
> **Note:** this is a learning/portfolio project, not a medical diagnostic tool.

## About this project

I started with a simple tutorial-style neural network — one hidden layer, no regularization,
no baseline to compare against — and rebuilt it into a proper end-to-end classification
project. The goal was to go beyond "train a model, check accuracy" and actually practice the
things that matter in a real ML workflow: a correct validation strategy, comparing against a
baseline, tuning the decision threshold without leaking test data, explaining the model's
predictions, and testing the code.

The workflow covers:

- Exploratory Data Analysis (EDA)
- Stratified train/validation/test split
- Feature scaling with `StandardScaler`
- A Logistic Regression baseline to compare the neural network against
- A regularized Keras neural network (Batch Normalization, Dropout, L2)
- Early stopping and best-model checkpointing
- Evaluation with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC and
  Precision-Recall curves
- Threshold tuning — chosen on the validation set, not the test set, to avoid leaking
  test information into the decision
- SHAP explainability: global feature importance plus a per-prediction waterfall plot
- Saved model, scaler, feature names, and run metadata
- A Streamlit app for interactive prediction
- Automated tests (`pytest`) covering data integrity and model behavior

## Dataset

Built-in `load_breast_cancer()` dataset from scikit-learn.

Classes:
- `0` = Malignant
- `1` = Benign

## Project structure

```
breast-cancer-classification/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── README.md
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   ├── 00_original_breast_cancer_nn.ipynb
│   └── 01_advanced_breast_cancer_nn.ipynb
│
├── reports/
│   └── figures/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
│
├── tests/
│   ├── test_data.py
│   └── test_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## How to run locally

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Train the model

```powershell
python -m src.train
```

This creates:

```
models/breast_cancer_nn.keras
models/scaler.joblib
models/feature_names.joblib
models/metadata.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
reports/figures/precision_recall_curve.png
```

### 4. Make one prediction from the command line

```powershell
python -m src.predict
```

### 5. Generate SHAP explainability plots

```powershell
python -m src.explain
```

This creates:

```
reports/figures/shap_summary.png
reports/figures/shap_single_prediction.png
```

### 6. Run the tests

```powershell
pytest tests/
```

### 7. Run the Streamlit app

```powershell
streamlit run app/streamlit_app.py
```

## What I focused on

The original notebook this started from trains a bare neural network and checks accuracy.
Rebuilding it, I focused on:

- A clean, modular project structure instead of one long notebook
- A correct validation strategy — proper train/val/test split, with the decision threshold
  tuned on validation data and the test set touched only once, at the end
- A baseline model, so the neural network's performance means something relative to a
  simpler alternative
- More complete evaluation than accuracy alone (precision, recall, F1, ROC-AUC, PR curve)
- SHAP explainability, since "the model said so" isn't good enough reasoning on its own,
  especially for a health-related dataset
- Tests, so the data pipeline and model behavior are actually verified, not just assumed
- A small Streamlit app so the model is something you can interact with, not just numbers
  in a notebook

## Disclaimer

This is a learning project built on a public educational dataset. It is not intended for
real medical decision-making.
