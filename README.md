# Breast Cancer Classification: Evaluation, Explainability & Interactive Demo

An end-to-end machine learning portfolio project comparing a regularized Keras neural network with Logistic Regression on scikit-learn's Breast Cancer Wisconsin (Diagnostic) dataset.

**Core contribution:** extending a notebook experiment into a modular workflow with stratified evaluation, validation-only threshold selection, SHAP explanations, reusable inference artifacts, a Streamlit interface, and automated tests.

**Educational portfolio project only; not a medical diagnostic tool.** “Advanced” in the repository name refers to the workflow, not advanced-stage cancer. The task is benign/malignant classification, not cancer staging or prognosis.

## Technical contributions

- **Evaluation design:** stratified train/validation/test splits of approximately 70%/15%/15% (397/86/86 samples), with `StandardScaler` fitted only on training features.
- **Baseline comparison:** Logistic Regression and the neural network use the same training data, scaled features, and held-out test split.
- **Regularized modeling:** a 64 → 32 → 16 → 1 network with Batch Normalization, Dropout, L2 regularization, balanced training class weights, early stopping, best-model checkpointing, and learning-rate reduction.
- **Decision policy:** select the neural network threshold on validation data by maximizing F1 over 181 thresholds from 0.05 to 0.95. Test labels are not used for threshold selection.
- **Explainability:** a separate SHAP script generates global feature-attribution and single-prediction waterfall plots.
- **Software engineering:** separate data, model, training, evaluation, explanation, and prediction modules; saved model/scaler/feature order/metadata; a Streamlit demo; pytest checks for data and model contracts.

See [the original experiment](notebooks/00_original_breast_cancer_nn.ipynb) and [the extended notebook](notebooks/01_advanced_breast_cancer_nn.ipynb). The module commands below are the main reproducible workflow.

## Dataset and label semantics

The built-in `load_breast_cancer()` dataset contains 569 samples and 30 numeric features derived from breast-mass cell nuclei. No separate CSV download is required.

| Label | Class | Interpretation |
| --- | --- | --- |
| 0 | Malignant | Model output below the selected threshold |
| 1 | Benign | Model output at or above the selected threshold |

The sigmoid output estimates **P(benign)**. The threshold objective and default binary precision, recall, F1, ROC and precision–recall curves treat **benign as the positive class**. They must not be described as malignant detection sensitivity. The neural network classification report also provides malignant-class precision, recall and F1.

## Comparison and results

**Results status:** the current repository does not include trained artifacts, `models/metadata.json`, or executed outputs in the extended notebook. No numerical advantage of the neural network over Logistic Regression is established by the checked-in evidence. Outputs from the original notebook are not results for this extended pipeline.

After running `python -m src.train`, inspect `models/metadata.json`:

| Test-set result | Logistic Regression | Neural network |
| --- | --- | --- |
| Decision threshold | Fixed at 0.5 | Selected on validation F1 |
| Accuracy | `baseline_logistic_regression.accuracy` | `neural_network.classification_report.accuracy` |
| Benign precision / recall / F1 | `baseline_logistic_regression.precision`, `recall`, `f1` | `neural_network.classification_report.benign` |
| ROC-AUC | `baseline_logistic_regression.roc_auc` | `neural_network.roc_auc` |
| Confusion matrix | Not saved by the current baseline code | `neural_network.confusion_matrix` |

Confusion-matrix rows are true labels and columns are predicted labels, both ordered malignant then benign. The malignant → benign cell represents a missed malignant case.

**How to interpret the comparison:** threshold-dependent scores use different decision policies, and only the neural network uses balanced class weights. This is a useful baseline comparison, but it does not isolate the effect of model architecture. ROC-AUC compares ranking without choosing a threshold. Report the actual outcome, including a baseline win or tie; greater complexity alone is not evidence of improvement.

## Run locally

Run all commands from the repository root. Use a Python version supported by TensorFlow; Python 3.11 is a suggested starting point, not a verified compatibility guarantee. Dependencies currently specify minimum versions rather than a locked environment.

```bash
git clone https://github.com/EmanAli98/advanced-breast-cancer-nn.git
cd advanced-breast-cancer-nn
python -m venv .venv
```

Activate the environment:

- macOS / Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install dependencies and train:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.train
```

Training writes:

- `models/breast_cancer_nn.keras`, `scaler.joblib`, `feature_names.joblib`, and `metadata.json`
- `reports/figures/test_confusion_matrix.png`
- `reports/figures/test_roc_curve.png`
- `reports/figures/test_precision_recall_curve.png`

### Streamlit demo

Train first: the app requires all four artifacts in `models/`.

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`). Enter the 30 features using the sidebar numeric inputs, then select **Predict**. The app displays P(benign), the saved validation threshold, the predicted class, and run metadata. Its initial zero values are placeholders, not a representative sample. SHAP plots are generated separately and are not embedded in the app.

### Tests

```bash
python -m pytest tests/ -q
```

Training and saved model files are not required for these tests; TensorFlow must still be installed. The suite contains eight tests covering dataset shape, split proportions, training-feature standardization, class balance, feature order, model input/output shapes, sigmoid output range, loss, and compiled metrics. These are unit checks; they do not establish model quality or end-to-end app reliability.

### Prediction and explanations

After training:

```bash
python -m src.predict
python -m src.explain
```

The prediction command uses a hard-coded 30-feature example. SHAP writes `reports/figures/shap_summary.png` and `shap_single_prediction.png`; generation can be slower than ordinary prediction.

## Repository guide

| Location | Purpose |
| --- | --- |
| `src/data.py`, `src/config.py` | Dataset loading, stratified splitting, scaling, paths and split seed |
| `src/model.py`, `src/train.py` | Keras architecture, baseline, callbacks, threshold selection and artifact saving |
| `src/evaluate.py` | Threshold search, classification report and evaluation plots |
| `src/explain.py`, `src/predict.py` | SHAP attribution and saved-artifact inference |
| `app/streamlit_app.py` | Interactive prediction demo |
| `tests/test_data.py`, `tests/test_model.py` | Data-pipeline and model unit checks |
| `notebooks/` | Original experiment and extended walkthrough |
| `models/`, `reports/figures/` | Generated artifacts and plots; initially placeholders |

## Limitations and lessons

- **Small, single benchmark:** 86 test samples and one split provide limited evidence of generalization. There is no external validation, clinical validation, subgroup analysis, or uncertainty interval.
- **Objective choice matters:** maximizing benign F1 does not directly minimize missed malignant cases. A malignancy-focused policy would require a separately specified validation objective.
- **Validation reuse:** early stopping and threshold selection share the same validation split. This avoids test-based tuning but can still overfit validation choices. Test results should not guide repeated revisions.
- **Reproducibility is partial:** the split seed and TensorFlow seed are set, but dependencies are not locked and full deterministic training is not configured.
- **SHAP explains model behavior, not causality:** the current script uses 50 held-out test samples as its background and recreates the scaler from the split rather than loading the saved scaler. This does not tune the threshold, but makes explanations dependent on that test distribution and matching preprocessing.
- **Demo scope:** inputs lack domain-range validation, displayed probabilities have not been calibrated, and tests do not yet cover artifact loading, threshold selection, or the Streamlit workflow.

## Small next improvements

1. **Publish evidence:** attach a dated run summary with commit, environment versions, both models' metrics, malignant recall, thresholds, and confusion matrices. Add the actual comparison table and one evaluation figure here after verification.
2. **Make the demo immediate:** add one clearly labeled dataset example, a screenshot or short GIF, and a demo link when available; replace zero defaults and validate input ranges.
3. **Strengthen repeatability:** record a tested environment and add GitHub Actions for pytest. Add focused checks for split disjointness, train-only scaler fitting, threshold selection, and saved-artifact inference.
4. **Align explanations and inference:** load the saved scaler in SHAP and use a training-only background. Compare threshold policies consistently on validation data before any new final test evaluation.

For a CV or LinkedIn project entry, a supportable description is:

> Built a modular breast-cancer classification pipeline comparing a regularized Keras network with Logistic Regression, with stratified evaluation, validation-only threshold tuning, SHAP explanations, a Streamlit demo, and eight pytest unit checks.

Add measured results only after publishing a reproducible run. The strongest current portfolio evidence is the evaluation design, explainability workflow, and code organization.
