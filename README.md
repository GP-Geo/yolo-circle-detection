# YOLO Circle Detection Pipeline

End-to-end pipeline for detecting circular features in geospatial satellite imagery using YOLO object detection. The pipeline handles dataset creation from vector labels, model training, large-scale tiled inference, post-processing, evaluation, and export to GIS formats.

## Project Overview

This pipeline is designed to detect circular features (e.g., archaeological structures, agricultural features) from satellite imagery. It supports multiple sensors including Sentinel-2 and WorldView-3.

**Key features:**
- Tile-based processing for large imagery (handles images larger than memory)
- Automatic dataset creation from GeoPackage vector labels
- Support for multispectral imagery (4-8 bands)
- Sliding window inference with NMS post-processing
- Comprehensive evaluation metrics
- Export to GIS-ready formats (GeoPackage)
- Structured logging and error handling across all pipeline scripts
- Configuration validation with Pydantic schemas
- Comprehensive test suite (27 tests)

**Repository structure:**
- Code, configs, and documentation are version-controlled
- Data, models, outputs, and reports are local-only (ignored via `.gitignore`)

## Directory Layout

```
YOLO_detections/
├── scripts/              # Pipeline scripts (00-10)
├── configs/              # YAML configuration files
│   └── pipeline.yaml     # Main config (tile size, bands, splits, etc.)
├── utils/                # Shared utilities (logging, config, paths, validation)
├── tests/                # Unit tests (27 tests, all passing)
├── data/                 # Local only - not versioned
│   ├── raw/              # Original input data
│   ├── external/         # Active data for training
│   │   └── active/       # Current training images and labels
│   ├── interim/          # Intermediate processing outputs
│   └── processed/        # YOLO-formatted datasets
├── models/               # Local only - not versioned
│   ├── weights/          # Base YOLO weights (yolo11n.pt)
│   └── runs/
│       └── training_runs/  # Trained model checkpoints
├── outputs/              # Local only - not versioned
│   ├── inference/        # Tiled images for inference
│   ├── overlays/         # Visualization overlays
│   └── predictions/      # Detection results
├── reports/              # Local only - not versioned
│   ├── evaluation/       # Model evaluation reports
│   └── logs/             # Execution logs
├── archive/              # Archived models and datasets
├── notebooks/            # Jupyter notebooks for exploration
├── README.md             # This file
├── environment.yml       # Conda environment specification
├── requirements.txt      # Python dependencies
└── paths.py              # Path configuration
```

## Prerequisites

- **Python**: 3.10+ recommended
- **Conda**: Miniforge, Miniconda, or Anaconda
- **GDAL**: Installed via conda-forge (handled by `environment.yml`)
- **GPU**: Optional but recommended for training (MPS for Apple Silicon, CUDA for NVIDIA)

## Installation

### 1. Create Conda Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate yolo-circle-detection
```

### 2. Install Additional Dependencies (Optional)

For configuration validation (recommended):
```bash
pip install pydantic>=2.0.0
```

For development and testing:
```bash
pip install -r requirements-dev.txt
```

### 3. Verify Installation

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import rasterio; print(rasterio.__version__)"
python -c "import pydantic; print('Config validation available')"
pytest -v  # Should show 27 tests passing
```

### 4. Download Base Weights

Download YOLO11n base weights:

```bash
mkdir -p models/weights
# Download from ultralytics or use auto-download during first training
```

## Configuration

All pipeline settings are controlled via `configs/pipeline.yaml`:

```yaml
run_id: s2_v1_31012026          # Unique identifier for this run

inputs:                          # Input rasters and their labels
  - id: image1
    raster: data/external/active/s2_image1.tif
    masks_gpkg: data/external/active/s2_image1.gpkg
    masks_layer: image1_sent2

bands:                           # Band selection
  ms: [1, 2, 3, 4]              # Multispectral bands (R,G,B,NIR)
  rgb: [4, 3, 2]                # RGB for visualization

tiling:
  tile_size_px: 384             # Tile size (must be divisible by 32)
  stride_frac: 0.5              # Overlap fraction (0.5 = 50% overlap)

labels:
  class_name: circle            # Object class name
  class_id: 0                   # YOLO class ID
  min_bbox_px: 6                # Minimum bounding box size

split:
  ratios: {train: 0.7, val: 0.3, test: 0.0}
  seed: 42
```

### Configuration Validation

The pipeline includes optional Pydantic-based configuration validation to catch errors early:

```python
from utils import load_config

# Load with validation (recommended)
cfg = load_config(Path("configs/pipeline.yaml"), validate=True)
```

**Validation checks:**
- ✅ `tile_size_px` must be divisible by 32 (YOLO architecture requirement)
- ✅ `overlap_fraction` must be between 0.0 and 1.0
- ✅ `split ratios` must sum to 1.0
- ✅ All required fields present
- ✅ Correct data types

Install pydantic for validation support:
```bash
pip install pydantic>=2.0.0
```

## Code Quality & Testing

### Testing

Run the test suite to verify everything works:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=utils --cov=scripts

# Expected: 27 tests passing
```

### Logging

All pipeline scripts use structured logging:
- Color-coded console output
- File logging with rotation
- Timestamps and log levels
- Easy debugging and monitoring

Logs are saved to:
- Training: `models/runs/logs/training.log`
- General: `reports/logs/`

## Pipeline Scripts

All scripts assume you're in the repository root with the conda environment activated.

### 00 - Inspect GeoPackage

Inspect and validate GeoPackage files containing vector labels.

```bash
# Inspect all GPKGs in active directory (default)
python scripts/00_inspect_gpkg.py

# Inspect specific GeoPackage files
python scripts/00_inspect_gpkg.py data/external/active/s2_image1.gpkg data/external/active/s2_image2.gpkg

# Inspect all GPKGs in a directory
python scripts/00_inspect_gpkg.py data/raw/vectors/

# Quiet mode (less verbose)
python scripts/00_inspect_gpkg.py --quiet
```

**What it does:**
- Lists all layers in each GeoPackage
- Shows geometry types and feature counts per layer
- Displays CRS (coordinate reference system)
- Reports invalid or empty geometries
- Shows spatial bounds
- Lists all attribute columns

**Purpose:**
- Validate label data before dataset creation
- Verify layer names for pipeline configuration
- Check data quality and completeness

---

### 01 - Build YOLO Dataset

Creates a YOLO-formatted dataset from satellite imagery and vector labels. This is the **main dataset creation script**.

```bash
python scripts/01_build_yolo_dataset.py
```

**What it does:**
- Reads configuration from `configs/pipeline.yaml`
- Tiles input rasters into overlapping patches (e.g., 384×384 pixels)
- Converts vector labels (GeoPackage) to YOLO format (normalized bounding boxes)
- Splits data into train/val/test sets
- Creates `data.yaml` for YOLO training
- Outputs to `data/processed/yolo_dataset_{run_id}/`

**Outputs:**
```
data/processed/yolo_dataset_{run_id}/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── images_rgb/              # RGB previews
├── data.yaml                # YOLO dataset config
└── dataset_meta.json        # Metadata (tile size, splits, etc.)
```

---

### 02 / 03 - Vector to Labels & Split Dataset

These are legacy scripts. Their functionality is **now integrated into script 01**.

```bash
# Not needed - use script 01 instead
python scripts/02_vectors_to_yolo_labels.py
python scripts/03_split_dataset.py
```

---

### 04 - Train YOLO Model

Trains a YOLO11n model on the prepared dataset.

```bash
python scripts/04_train.py
```

**What it does:**
- Loads the dataset created by script 01
- Initializes YOLO11n with pretrained weights
- Trains with data augmentation and early stopping
- Saves checkpoints to `models/runs/training_runs/yolo11n_{run_id}/`

**Training parameters:**
- Epochs: 1000 (with patience: 150 for early stopping)
- Batch size: 8
- Image size: Matches tile_size_px from config
- Device: Auto-detected (MPS/CUDA/CPU)
- Augmentation: Rotation, translation, scaling, flips, mosaic

**Outputs:**
```
models/runs/training_runs/yolo11n_{run_id}/
├── weights/
│   ├── best.pt              # Best model (by validation mAP)
│   └── last.pt              # Last epoch checkpoint
├── results.csv              # Training metrics per epoch
├── confusion_matrix.png
├── results.png              # Loss and metric curves
└── args.yaml                # Training configuration
```

**Monitoring:**
- Training logs: `models/runs/logs/training.log`
- Watch metrics: `tail -f models/runs/logs/training.log`

---

### 05 - Make Inference Tiles

Tiles a large raster for inference (can be different from training images).

```bash
python scripts/05_make_inference_tiles.py \
  --config configs/pipeline.yaml \
  --out outputs/inference/tiles_full
```

**What it does:**
- Uses the tiling parameters from `configs/pipeline.yaml`
- Tiles large imagery into patches matching the training tile size
- Creates both multispectral TIFFs (for model) and RGB PNGs (for visualization)

**Outputs:**
```
outputs/inference/tiles_full/
├── images_ms4/              # 4-band TIFFs for model input
├── images_rgb/              # RGB PNGs for visualization
└── tiles_meta.json          # Tiling metadata (geotransform, tile IDs)
```

---

### 06 - Run Inference

Runs the trained model on tiled imagery to detect circles.

```bash
# For Apple Silicon (M1/M2/M3) - use MPS fallback for NMS operation
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/06_infer.py \
  --model models/runs/training_runs/yolo11n_s2_v1_31012026/weights/best.pt \
  --source outputs/inference/tiles_full/images_ms \
  --meta outputs/inference/tiles_full/tiles_meta.json \
  --out outputs/predictions/tiles

# For other systems (or to force CPU)
python scripts/06_infer.py \
  --model models/runs/training_runs/yolo11n_s2_v1_31012026/weights/best.pt \
  --source outputs/inference/tiles_full/images_ms \
  --meta outputs/inference/tiles_full/tiles_meta.json \
  --out outputs/predictions/tiles \
  --device cpu
```

**Parameters:**
- `--model`: Path to trained weights (best.pt or last.pt)
- `--source`: Directory of tiled images
- `--meta`: Tile metadata (for coordinate mapping)
- `--out`: Output directory for predictions
- `--device`: Optional - specify device (cpu/cuda/mps, defaults to auto-detect)

**What it does:**
- Loads the trained model
- Runs inference on each tile
- Saves detections in tile coordinate space
- Outputs predictions CSV with tile IDs and bounding boxes

**Outputs:**
```
outputs/predictions/tiles/
└── predictions_tiles.csv    # Detections: tile_id, class, conf, x1, y1, x2, y2
```

---

### 07 - Merge + NMS

Merges overlapping tile predictions and applies Non-Maximum Suppression (NMS) in global pixel space.

```bash
# Processes all images automatically
python scripts/07_merge_nms.py \
  --pred_csv outputs/predictions/tiles/predictions_tiles.csv \
  --meta data/processed/yolo_dataset_s2_v1_31012026/dataset_meta.json \
  --out outputs/predictions/merged/yolo11n_s2_v1_full

# Or process a specific image only
python scripts/07_merge_nms.py \
  --pred_csv outputs/predictions/tiles/predictions_tiles.csv \
  --meta data/processed/yolo_dataset_s2_v1_31012026/dataset_meta.json \
  --out outputs/predictions/merged/yolo11n_s2_v1_full \
  --input_id image1
```

**What it does:**
- Automatically detects and processes all input images from the predictions CSV
- Converts tile-local predictions to global pixel coordinates
- Applies NMS to remove duplicate detections from overlapping tiles
- Applies nested suppression (prefers larger/outer boxes)
- Creates separate output files for each image (e.g., `detections_merged_image1.gpkg`)

**Outputs:**
```
outputs/predictions/merged/yolo11n_s2_v1_full/
├── detections_merged_image1.csv    # Global detections for image1
├── detections_merged_image1.gpkg   # Georeferenced detections for image1
├── detections_merged_image2.csv    # Global detections for image2
└── detections_merged_image2.gpkg   # Georeferenced detections for image2
```

---

### 08 - Render Predictions

Visualizes predictions overlaid on RGB tiles for quality assessment.

```bash
python scripts/08_render_predictions.py \
  --pred outputs/predictions/tiles/predictions_tiles.csv \
  --rgb_dir outputs/inference/tiles_full/images_rgb \
  --out outputs/overlays/tiles \
  --stride 192
```

**Parameters:**
- `--stride`: Tile stride in pixels (from dataset_meta.json: tile_size_px * stride_frac)
  - Example: 384 × 0.5 = 192

**What it does:**
- Draws bounding boxes on RGB tile images
- Color-codes by confidence score
- Useful for visual inspection and debugging

**Outputs:**
```
outputs/overlays/tiles/
└── *.png                    # RGB tiles with bounding box overlays
```

---

### 09 - Export to GeoPackage

Exports predictions to a GIS-compatible GeoPackage format with proper georeferencing.

```bash
# Processes all images automatically
python scripts/09_export_predictions_gpkg.py \
  --pred outputs/predictions/tiles/predictions_tiles.csv \
  --tiles_meta outputs/inference/tiles_full/tiles_meta.json \
  --out_gpkg outputs/predictions/full/detections_merged.gpkg \
  --layer detections \
  --dedup \
  --dedup_by_class

# Or process a specific image only
python scripts/09_export_predictions_gpkg.py \
  --pred outputs/predictions/tiles/predictions_tiles.csv \
  --tiles_meta outputs/inference/tiles_full/tiles_meta.json \
  --out_gpkg outputs/predictions/full/detections_merged.gpkg \
  --layer detections \
  --dedup \
  --dedup_by_class \
  --input_id image1
```

**Parameters:**
- `--dedup`: Remove duplicate detections based on IOU
- `--dedup_by_class`: Deduplicate separately per class
- `--input_id`: Optional - specify a single image to process (e.g., image1)

**What it does:**
- Automatically detects and processes all input images from the predictions CSV
- Converts pixel coordinates to geographic coordinates (using tiles_meta.json)
- Creates polygon geometries from bounding boxes
- Adds attributes: confidence, class, detection ID
- Creates separate output files for each image (e.g., `detections_merged_image1.gpkg`)
- Exports to GeoPackage (readable by QGIS, ArcGIS, etc.)

**Outputs:**
```
outputs/predictions/full/
└── detections_merged.gpkg   # Georeferenced detections (layer: detections)
```

---

### 10 - Evaluate Model

Computes comprehensive evaluation metrics for the trained model.

```bash
python scripts/10_evaluate_model.py \
  --model models/runs/training_runs/yolo11s_s2_v2_01022026/weights/best.pt \
  --conf_threshold 0.2 \
  --iou_threshold 0.5 \
  --split val
```

**Parameters:**
- `--model`: Path to trained model weights
- `--conf_threshold`: Confidence threshold for predictions (default: 0.25)
- `--iou_threshold`: IOU threshold for evaluation (default: 0.5)
- `--split`: Dataset split to evaluate on - 'train', 'val', or 'test' (default: val)

**What it does:**
- Runs validation on the specified dataset split
- Computes precision, recall, F1-score per class
- Calculates mAP (mean Average Precision) at IOU thresholds [0.5:0.95]
- Generates confusion matrices
- Analyzes performance by object size (small/medium/large)

**Outputs:**
```
reports/evaluation/yolo11n_{run_id}/
├── metrics.json             # JSON summary of all metrics
├── confusion_matrix.png     # Confusion matrix visualization
├── pr_curve.png             # Precision-Recall curve
└── evaluation_report.txt    # Human-readable report
```

**Metrics logged:**
- Precision, Recall, F1-Score
- mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
- Per-class performance breakdown
- Inference speed (FPS)

---

## Complete Workflow Example

Here's a typical workflow from raw data to final GeoPackage:

```bash
# 1. Prepare your data
# Place your imagery and labels in data/external/active/
# Edit configs/pipeline.yaml with your run_id and input paths

# 2. Build dataset
python scripts/01_build_yolo_dataset.py

# 3. Train model
python scripts/04_train.py

# 4. Evaluate model (optional)
python scripts/10_evaluate_model.py \
  --model models/runs/training_runs/yolo11n_s2_v1_31012026/weights/best.pt

# 5. Prepare inference tiles (for a new large image)
python scripts/05_make_inference_tiles.py \
  --config configs/pipeline.yaml \
  --out outputs/inference/tiles_full

# 6. Run inference
python scripts/06_infer.py \
  --model models/runs/training_runs/yolo11n_s2_v1_31012026/weights/best.pt \
  --source outputs/inference/tiles_full/images_ms4 \
  --meta outputs/inference/tiles_full/tiles_meta.json \
  --out outputs/predictions/tiles

# 7. Merge predictions with NMS
python scripts/07_merge_nms.py \
  --pred_csv outputs/predictions/tiles/predictions_tiles.csv \
  --meta data/processed/yolo_dataset_s2_v2_01022026/dataset_meta.json \
  --out outputs/predictions/merged/full

# 8. Export to GeoPackage
python scripts/09_export_predictions_gpkg.py \
  --pred outputs/predictions/tiles/predictions_tiles.csv \
  --tiles_meta outputs/inference/tiles_full/tiles_meta.json \
  --out_gpkg outputs/predictions/full/detections.gpkg \
  --layer circles \
  --dedup \
  --dedup_by_class

# 9. (Optional) Visualize predictions
python scripts/08_render_predictions.py \
  --pred outputs/predictions/tiles/predictions_tiles.csv \
  --rgb_dir outputs/inference/tiles_full/images_rgb \
  --out outputs/overlays/tiles \
  --stride 64
```

## Model Naming Convention

All models and datasets follow the simplified naming format: `{sensor}_v{version}_{DDMMYYYY}`

**Components:**
- **sensor**: Data source (s2=Sentinel-2, wv3=WorldView-3, ps=PlanetScope, etc.)
- **version**: Sequential version number (v1, v2, v3, etc.)
- **date**: Creation date in DDMMYYYY format

**Examples:**
- Dataset: `yolo_dataset_s2_v1_31012026` (Sentinel-2, version 1, created Jan 31, 2026)
- Model: `yolo11n_s2_v1_31012026` (YOLO11n trained on the above dataset)
- Next version: `yolo_dataset_s2_v2_15022026` (Sentinel-2, version 2, created Feb 15, 2026)

**Version increments:**
- Use v1, v2, v3... for sequential versions with the same sensor
- Increment version when creating a new training run
- Date reflects when the dataset/model was created
- Update `run_id` in `configs/pipeline.yaml` for each new version

## Technical Notes

### Tile Size Requirements
- Tile sizes must be **divisible by 32** (required for YOLO's downsampling architecture)
- Valid sizes: 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, etc.
- Recommended: 256, 384, or 512 pixels
- Larger tiles = more context but fewer training samples
- Smaller tiles = more samples but less context per detection

### Overlap Strategy
- `stride_frac: 0.5` means 50% overlap between tiles
- Overlap reduces edge effects during inference
- Overlap increases computational cost but improves detection at tile boundaries
- NMS (script 07) handles duplicate detections from overlapping tiles

### GPU Support
- **Apple Silicon (M1/M2/M3)**: Uses MPS (Metal Performance Shaders) automatically
- **NVIDIA GPUs**: Uses CUDA automatically if available
- **CPU**: Falls back to CPU if no GPU detected (slower training/inference)

### Data Organization
- **Training data**: `data/external/active/` - Your current training images and labels
- **Raw data**: `data/raw/` - Original unprocessed data
- **Processed datasets**: `data/processed/yolo_dataset_{run_id}/` - YOLO-formatted datasets
- **Models**: `models/runs/training_runs/yolo11n_{run_id}/` - Trained weights and logs
- **Archive**: `archive/` - Old models and datasets (kept for reference)

### Configuration Management
- All pipeline settings are in `configs/pipeline.yaml` (single source of truth)
- Dataset configuration is stored in `dataset_meta.json` (auto-generated by script 01)
- Training parameters are saved in model's `args.yaml` (auto-generated by script 04)

## Troubleshooting

### Common Issues

**1. "tile_size_px must be divisible by 32" error**
- Edit `configs/pipeline.yaml` and set `tile_size_px` to a valid value (256, 384, 512, etc.)

**2. Out of memory during training**
- Reduce batch size in `scripts/04_train.py` (line 145: `'batch': 8` → `'batch': 4`)
- Or reduce tile size in `configs/pipeline.yaml`

**3. No GPU detected**
- Check CUDA installation: `nvidia-smi` (NVIDIA) or `system_profiler SPDisplaysDataType | grep Metal` (Apple)
- Verify PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"`

**4. GeoPackage layer not found**
- Use script 00 to inspect available layers: `python scripts/00_inspect_gpkg.py`
- Update `masks_layer` in `configs/pipeline.yaml` to match the actual layer name

**5. Tile coordinates mismatch in export**
- Ensure you're using the correct `tiles_meta.json` file (from script 05)
- Verify the tile stride matches: `tile_size_px * stride_frac`

## Advanced Usage

### Custom Data Augmentation
Edit `scripts/04_train.py` lines 156-162 to adjust augmentation parameters:
```python
'degrees': 10,        # Rotation range (±10°)
'translate': 0.05,    # Translation (5% of image)
'scale': 0.2,         # Scaling (±20%)
'fliplr': 0.5,        # Horizontal flip probability
'flipud': 0.5,        # Vertical flip probability
'mosaic': 1.0,        # Mosaic augmentation
```

### Multi-Sensor Training
To combine data from multiple sensors:
1. Add multiple entries in `configs/pipeline.yaml` under `inputs:`
2. Ensure all inputs use the same tile size and overlap
3. Run script 01 - it will automatically merge all inputs

### Hyperparameter Tuning
Key parameters to tune in `scripts/04_train.py`:
- Learning rate: `'lr0': 0.003`
- Batch size: `'batch': 8`
- Epochs: `'epochs': 1000`
- Patience: `'patience': 150` (early stopping)
- Confidence threshold: `'conf': 0.3`
- IOU threshold: `'iou': 0.50`

## HITL Iterative Refinement Workflow

The pipeline supports a human-in-the-loop (HITL) workflow for iteratively improving detection quality on large tiles (~100×50 km Sentinel-2 scenes). This is especially useful when ground truth labels are sparse — you bootstrap with a rough model, review its predictions in QGIS, correct them, and retrain.

### Workflow Overview

```
1. Train initial model    python scripts/04_train.py
2. Run inference          python scripts/12_run_iteration.py --model best.pt --raster large_tile.tif
3. Review in QGIS         Open output GeoPackage + raster. Delete FPs, add FNs, adjust bboxes. Save.
4. Ingest corrections     python scripts/11_ingest_corrections.py --corrections_gpkg corrected.gpkg --raster large_tile.tif
5. Retrain                python scripts/04_train.py
6. Repeat from step 2 until satisfied
```

### Step-by-step Instructions

#### Step 1 — Initial training

Build the base dataset and train a first model:

```bash
python scripts/01_build_yolo_dataset.py
python scripts/04_train.py
```

Note the path to `best.pt` from the training output (e.g. `models/runs/training_runs/yolo11s_s2_v2/weights/best.pt`).

#### Step 2 — Run an inference iteration

```bash
python scripts/12_run_iteration.py \
    --model  models/runs/training_runs/yolo11s_s2_v2/weights/best.pt \
    --raster data/external/active/s2_large_tile.tif \
    --output-dir outputs/iterations/round1 \
    --input-id large_tile
```

This chains scripts 05 → 06 → 07 → 09 and writes a GeoPackage to
`outputs/iterations/round1/gpkg/detections_merged.gpkg`.

After it completes, the script prints detailed next-step instructions.

#### Step 3 — Review detections in QGIS

1. Open QGIS and load the raster and the output GeoPackage as layers.
2. Review each detection polygon:
   - **Delete false positives**: select the feature → `Delete` key
   - **Add missed circles**: use the polygon digitize tool to draw new bboxes
   - **Adjust inaccurate bboxes**: node editing tool
3. Save the corrected layer to a new GeoPackage, e.g.:
   `data/corrections/round1_corrected.gpkg`

#### Step 4 — Ingest corrections

```bash
python scripts/11_ingest_corrections.py \
    --corrections_gpkg data/corrections/round1_corrected.gpkg \
    --raster data/external/active/s2_large_tile.tif \
    --round_id round1
```

This tiles the corrected GeoPackage to match the existing dataset grid, writes
new YOLO `.tif` + `.txt` pairs, and replaces any tiles already present
(`merge_strategy: replace`). A `corrections_manifest.json` is written to the
dataset directory for traceability.

#### Step 5 — Retrain and repeat

```bash
python scripts/04_train.py
# Then run 12_run_iteration.py again with the new best.pt
```

### Configuration

HITL settings live in `configs/pipeline.yaml`:

```yaml
hitl:
  corrections_dir: data/corrections/   # where corrected GeoPackages are stored
  merge_strategy: replace              # replace duplicate tiles with corrections

iterations:
  current: 1
  history:
    - round: 1
      model: models/runs/.../best.pt
      corrections_gpkg: data/corrections/round1_corrected.gpkg
      date: 2026-03-14
```

Update `iterations.history` manually after each round to keep a record.

### Large-tile support

Scripts 05 and 06 use rasterio windowed reads — the full raster is never loaded
into memory at once. A 10,000×5,000 px tile (100×50 km at 10 m resolution)
produces ~11,400 tiles at the default stride; all stages handle this via
streaming reads and chunked processing. `tqdm` progress bars are shown for
long-running loops.

### New scripts

| Script | Purpose |
|--------|---------|
| `11_ingest_corrections.py` | HITL bridge: converts corrected GeoPackage → YOLO tiles + labels |
| `12_run_iteration.py` | Orchestrator: chains 05→06→07→09 for one iteration cycle |

---

## Recent Improvements

### Phase 2 (February 2026) ✅
- **Code Refactoring**: Eliminated duplicate functions across scripts (config loading, path resolution)
- **Logging Standardization**: Added structured logging to 4 additional scripts (05, 06, 08, 09)
- **Configuration Validation**: Implemented Pydantic schemas for config validation
- **Test Coverage**: Added 15 new tests (27 total, all passing)
- See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for complete details

### Phase 1 (January 2026) ✅
- **Centralized Logging**: Color-coded console output with file rotation
- **Model Evaluation**: Script 10 with mAP, precision, recall, F1-score metrics
- **Testing Framework**: pytest with 18 NMS algorithm tests
- See [PHASE1_IMPROVEMENTS.md](PHASE1_IMPROVEMENTS.md) for details

**Test Status**: 27/27 tests passing ✅

## References

- **YOLO**: [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- **Rasterio**: [Rasterio Documentation](https://rasterio.readthedocs.io/)
- **GeoPackage**: [OGC GeoPackage Specification](https://www.geopackage.org/)

## License

[Add your license information here]

## Contact

[Add contact information or project maintainers here]
