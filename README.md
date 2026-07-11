# Advanced Breast Cancer Classification with Neural Networks

> Educational machine learning project using the Scikit-learn Breast Cancer Wisconsin dataset.  
> **Important:** This project is for learning and portfolio purposes only. It is **not** a medical diagnostic tool.

## Project overview

This project improves a simple neural-network notebook into a more GitHub-ready deep learning project.

The workflow includes:

- Exploratory Data Analysis (EDA)
- Stratified train/validation/test split
- Feature scaling with `StandardScaler`
- Baseline model using Logistic Regression
- Advanced Keras neural network with Batch Normalization, Dropout, and L2 regularization
- Early stopping and best-model checkpointing
- Evaluation with Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curve, and Precision-Recall curve
- Threshold tuning for safer classification analysis
- SHAP explainability: global feature importance + per-prediction waterfall plots
- Saved model, scaler, feature names, and metadata
- Streamlit app for interactive prediction
- Automated tests (`pytest`) covering data integrity and model behavior

## Dataset

The project uses the built-in `load_breast_cancer()` dataset from Scikit-learn.

Classes:

- `0` = Malignant
- `1` = Benign

## Project structure

```text
breast_cancer_nn_advanced_github/
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

This will create:

```text
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

This will create:

```text
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

## What makes this version more advanced?

The original notebook trains a simple neural network and checks accuracy. This advanced version adds:

- Clean project structure
- Reusable Python modules
- Better validation strategy (proper train/val/test split, threshold tuned on validation, not test)
- Baseline comparison
- More serious evaluation metrics
- Threshold tuning
- SHAP explainability (global + per-prediction)
- Model/scaler saving
- A small app for prediction
- Automated tests
- A clear README for GitHub

## Portfolio explanation

You can describe this project on your GitHub like this:

> Built an advanced binary classification pipeline for breast cancer tumor classification using TensorFlow/Keras. The project includes EDA, preprocessing, a baseline model, a regularized neural network, threshold tuning, model evaluation with multiple metrics, model persistence, and a Streamlit interface for interactive predictions.

## Disclaimer

This is a learning project using a public educational dataset. It should not be used for real medical decision-making.
