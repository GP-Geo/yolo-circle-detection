from pathlib import Path

# Project root (this file lives at repo root)
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
RUNS_DIR = MODELS_DIR / "runs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INFERENCE_DIR = OUTPUTS_DIR / "inference"
OVERLAYS_DIR = OUTPUTS_DIR / "overlays"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

REPORTS_DIR = PROJECT_ROOT / "reports"
QA_DIR = REPORTS_DIR / "qa"
