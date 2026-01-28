from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = ROOT / "02_processed" / "yolo_dataset_ms8_v1" / "data.yaml"
PROJECT_DIR = ROOT / "03_models" / "runs"
EXP_NAME = "yolo11n_wv3_120e_v1"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        imgsz=512,
        epochs=150,
        patience=40,   
        batch=4,
        device=device,
        project=str(PROJECT_DIR),
        name=EXP_NAME,
        pretrained=True,

        # Keep eval fast on Mac (prevents late-epoch slowdown)
        conf=0.8,
        iou=0.50,
        plots=True,
        cache=True,
        workers=0,     # stable default on macOS
    )

if __name__ == "__main__":
    main()