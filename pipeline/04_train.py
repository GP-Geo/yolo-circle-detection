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
    import argparse
    parser = argparse.ArgumentParser(description="Train or fine-tune YOLO circle detection model.")
    parser.add_argument(
        "--finetune",
        default="",
        help=(
            "Path to trained weights to fine-tune from (e.g. models/runs/.../best.pt). "
            "Uses lower LR, fewer epochs, and saves under a versioned run name."
        ),
    )
    parser.add_argument(
        "--version-suffix",
        dest="version_suffix",
        default="",
        help="Suffix appended to the run name (e.g. 'ft1'). Auto-set to 'ft1','ft2',... when --finetune is used.",
    )
    parser.add_argument(
        "--model",
        default="yolo11s",
        help="Base YOLO model variant for full training (e.g. yolo11n, yolo11s, yolo11m). Ignored when --finetune is used.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default="",
        help="Override the output run directory name (e.g. yolo11n_s2_v1_15032026). Overrides auto-generated name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Override number of training epochs (default: 1200 for full training, 300 for fine-tune).",
    )
    args = parser.parse_args()

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

    # Load model — fine-tune from existing weights or train from pretrained backbone
    finetune_path = Path(args.finetune) if args.finetune else None
    if finetune_path and not finetune_path.is_absolute():
        finetune_path = ROOT / finetune_path

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if finetune_path:
        if not finetune_path.exists():
            logger.error(f"Fine-tune weights not found: {finetune_path}")
            return 1
        logger.info(f"Fine-tuning from: {finetune_path}")
        model = YOLO(str(finetune_path))
        model_size = "s"  # infer from filename if needed
        for part in finetune_path.stem.split("_"):
            if part.startswith("yolo11"):
                model_size = part.replace("yolo11", "")
        is_finetune = True
    else:
        base_model = args.model if args.model.endswith(".pt") else args.model + ".pt"
        model_size = base_model.replace(".pt", "").replace("yolo11", "")
        model_name = base_model
        weights_path = WEIGHTS_DIR / model_name
        if weights_path.exists():
            logger.info(f"Loading base model from: {weights_path}")
            model = YOLO(str(weights_path))
        else:
            logger.info(f"Downloading {model_name} from Ultralytics...")
            model = YOLO(model_name)
            model.save(str(weights_path))
        model_size = model_name.replace(".pt", "").replace("yolo11", "")
        is_finetune = False

    logger.info("Model loaded successfully")

    # Determine run name and suffix
    if args.run_name:
        run_name = args.run_name
    else:
        version_suffix = args.version_suffix
        if not version_suffix and is_finetune:
            # Auto-increment ft suffix based on existing run dirs
            existing = list((RUNS_DIR / "training_runs").glob(f"yolo11{model_size}_{run_id}_ft*"))
            version_suffix = f"ft{len(existing) + 1}"
        run_name = f"yolo11{model_size}_{run_id}" + (f"_{version_suffix}" if version_suffix else "")
    output_dir = RUNS_DIR / "training_runs" / run_name
    logger.info(f"Output directory: {output_dir}")

    # Training configuration — fine-tune uses lower LR and fewer epochs
    training_params = {
        'data': str(data_yaml),
        'imgsz': tile_size_px,
        'epochs': args.epochs if args.epochs > 0 else (300 if is_finetune else 1200),
        'patience': min(50, args.epochs // 3) if args.epochs > 0 else (50 if is_finetune else 200),
        'batch': 8,
        'device': device,
        'project': str(RUNS_DIR / "training_runs"),
        'name': run_name,
        'pretrained': True,
        'optimizer': 'AdamW',        # explicit — prevents auto overriding lr0
        'conf': 0.35,                # pre-filters boxes before NMS during val
        'iou': 0.50,
        'augment': True,
        'plots': True,
        'cache': 'disk',             # deterministic; was True (RAM)
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
        'lr0': 0.0003 if is_finetune else 0.003,  # 10x lower for fine-tuning
        'cos_lr': True,
    }

    logger.info("Training parameters:")
    logger.info(f"  Mode: {'Fine-tune' if is_finetune else 'Full training'}")
    logger.info(f"  Image size: {tile_size_px}px")
    logger.info(f"  Epochs: {training_params['epochs']} (patience: {training_params['patience']})")
    logger.info(f"  Batch size: {training_params['batch']}")
    logger.info(f"  Learning rate: {training_params['lr0']}")
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
        logger.warning("Training interrupted by user (Ctrl+C) — Ultralytics will save weights and run final validation")
        raise
    except Exception as e:
        logger.error("Training failed with error:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if isinstance(exit_code, int) else 0)
