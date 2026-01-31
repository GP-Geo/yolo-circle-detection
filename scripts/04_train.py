from pathlib import Path
import sys
import json
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from paths import PROCESSED_DIR, RUNS_DIR, WEIGHTS_DIR
from utils import setup_logger, load_config

# Set up logging
logger = setup_logger(__name__, log_file=RUNS_DIR / "logs" / "training.log")


def validate_dataset(data_yaml: Path, meta_path: Path) -> dict:
    """
    Validate that dataset files exist and are properly formatted.

    Args:
        data_yaml: Path to YOLO data.yaml file
        meta_path: Path to dataset metadata JSON

    Returns:
        Dictionary containing dataset metadata

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If metadata is invalid
    """
    if not data_yaml.exists():
        logger.error(f"YOLO data.yaml not found: {data_yaml}")
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")

    if not meta_path.exists():
        logger.error(f"Dataset metadata not found: {meta_path}")
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.debug(f"Loaded dataset metadata from {meta_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse dataset metadata: {e}")
        raise ValueError(f"Invalid JSON in metadata file: {e}") from e

    # Validate tile_size exists and is reasonable
    tile_size = meta.get("tile_size")
    if tile_size is None:
        logger.warning("tile_size not found in metadata, using default 512")
        tile_size = 512
    elif not (64 <= tile_size <= 2048):
        logger.error(f"Invalid tile_size in metadata: {tile_size}")
        raise ValueError(f"tile_size must be between 64 and 2048, got {tile_size}")

    return meta


def main():
    """Main training function with comprehensive logging and error handling."""
    logger.info("=" * 80)
    logger.info("Starting YOLO11 Circle Detection Training")
    logger.info("=" * 80)

    # Load configuration
    cfg_path = ROOT / "configs" / "pipeline.yaml"
    logger.info(f"Loading configuration from: {cfg_path}")

    try:
        cfg = load_config(cfg_path)
        run_id = cfg.get("run_id", "run")
        logger.info(f"Run ID: {run_id}")
    except Exception as e:
        logger.critical(f"Failed to load configuration: {e}", exc_info=True)
        return 1

    # Validate dataset
    data_root = PROCESSED_DIR / f"yolo_dataset_{run_id}"
    data_yaml = data_root / "data.yaml"
    meta_path = data_root / "dataset_meta.json"

    logger.info(f"Dataset directory: {data_root}")

    try:
        meta = validate_dataset(data_yaml, meta_path)
        tile_size_px = int(meta.get("tile_size", 512))
        logger.info(f"Tile size: {tile_size_px}px")
    except Exception as e:
        logger.critical(f"Dataset validation failed: {e}", exc_info=True)
        return 1

    # Set up device
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("No GPU acceleration available, using CPU (training will be slow)")

    # Load base model
    # Options: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large)
    model_name = "yolo11s.pt"  # Using small model for v2 (better performance than nano)
    weights_path = WEIGHTS_DIR / model_name

    # Ensure weights directory exists
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if weights_path.exists():
        logger.info(f"Loading base model from: {weights_path}")
        try:
            model = YOLO(str(weights_path))
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return 1
    else:
        # Auto-download weights from ultralytics
        logger.info("Base model weights not found locally")
        logger.info(f"Downloading {model_name} weights from Ultralytics...")
        try:
            model = YOLO(model_name)  # This will auto-download
            logger.info("Weights downloaded successfully")
            # Save to local weights directory for future use
            logger.info(f"Saving weights to: {weights_path}")
            model.save(str(weights_path))
        except Exception as e:
            logger.error(f"Failed to download model: {e}", exc_info=True)
            logger.error("Please check your internet connection")
            return 1

    # Training configuration
    model_size = model_name.replace(".pt", "").replace("yolo11", "")  # Extract: n, s, m, l
    output_dir = RUNS_DIR / "training_runs" / f"yolo11{model_size}_{run_id}"
    logger.info(f"Output directory: {output_dir}")

    training_params = {
        'data': str(data_yaml),
        'imgsz': tile_size_px,
        'epochs': 1200,
        'patience': 200,
        'batch': 8,
        'device': device,
        'project': str(RUNS_DIR / "training_runs"),
        'name': f"yolo11{model_size}_{run_id}",
        'pretrained': True,
        'conf': 0.1,
        'iou': 0.50,
        'augment': True,
        'plots': True,
        'cache': True,
        'workers': 0,
        'degrees': 10,
        'translate': 0.05,
        'scale': 0.2,
        'shear': 0.0,
        'fliplr': 0.5,
        'flipud': 0.5,
        'mosaic': 1.0,
        'close_mosaic': 10,
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'lr0': 0.003,
        'cos_lr': True,
    }

    logger.info("Training parameters:")
    logger.info(f"  Image size: {tile_size_px}px")
    logger.info(f"  Epochs: {training_params['epochs']} (patience: {training_params['patience']})")
    logger.info(f"  Batch size: {training_params['batch']}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Data augmentation: enabled")
    logger.info(f"  HSV augmentation: disabled (multispectral data)")

    # Start training
    logger.info("Starting training...")
    try:
        results = model.train(**training_params)
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        return 0
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        return 130
    except Exception as e:
        logger.error("Training failed with error:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if isinstance(exit_code, int) else 0)
