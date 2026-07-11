from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_PATH = MODEL_DIR / "breast_cancer_nn.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
