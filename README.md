# YOLO Circle Detection

End-to-end YOLO-based pipeline for detecting circular features in large geospatial imagery: dataset build from vectors, training, tiled inference, post-processing, visualization, and export back to GIS formats.

The repository is intentionally lean:
- Code, configs, and documentation are versioned
- Data, models, outputs, and reports are local-only and ignored via `.gitignore`

## Repository layout

- `scripts/` Pipeline scripts (00 to 09)
- `configs/` YAML configs for dataset and training
- `notebooks/` Optional exploration notebooks
- `data/` Local only: `raw/`, `interim/`, `processed/`, `external/`
- `models/` Local only: `weights/`, `runs/`
- `outputs/` Local only: `inference/`, `overlays/`, `predictions/`
- `reports/` Local only: `qa/`

## Prerequisites

- Python 3.10 recommended
- Conda (Miniforge/Miniconda/Anaconda)
- GDAL dependencies via conda-forge (handled by environment.yml)

## Install (conda)

From the repo root (`yolo-circle-detection/`):

```bash
conda env create -f environment.yml
conda activate yolo-circle-detection
```

If you prefer running without activating:

```bash
conda run -n yolo-circle-detection python scripts/00_inspect_gpkg.py
```

## Pipeline commands

All commands below assume you are in the repo root and the conda env is active.

### 00 - Inspect GPKG

```bash
python scripts/00_inspect_gpkg.py
```

### 01 - Build YOLO dataset (tiles + labels)

```bash
python scripts/01_build_yolo_dataset.py
```

### 02 / 03 - Placeholder steps

These are currently implemented inside `01_build_yolo_dataset.py`.

```bash
python scripts/02_vectors_to_yolo_labels.py
python scripts/03_split_dataset.py
```

### 04 - Train YOLO

```bash
python scripts/04_train.py
```

### 05 - Make inference tiles

```bash
python scripts/05_make_inference_tiles.py \
  --meta data/processed/yolo_dataset_ms8_v1/dataset_meta.json \
  --out outputs/inference/tiles_full
```

### 06 - Run inference on tiles

```bash
python scripts/06_infer.py \
  --model models/runs/training_runs/yolo11n_wv3_120e_v1/weights/best.pt \
  --source outputs/inference/tiles_full/images_ms8 \
  --meta outputs/inference/tiles_full/tiles_meta.json \
  --out outputs/predictions/tiles
```

### 07 - Merge + NMS in global pixel space

```bash
python scripts/07_merge_nms.py \
  --pred_csv outputs/predictions/tiles/predictions_tiles.csv \
  --meta data/processed/yolo_dataset_ms8_v1/dataset_meta.json \
  --out outputs/predictions/merged/yolo11n_wv3_v1_full
```

### 08 - Render predictions onto RGB tiles

Set `--stride` to the value used during tiling (from `dataset_meta.json` or `tiles_meta.json`).

```bash
python scripts/08_render_predictions.py \
  --pred outputs/predictions/tiles/predictions_tiles.csv \
  --rgb_dir outputs/inference/tiles_full/images_rgb \
  --out outputs/overlays/tiles \
  --stride 384
```

### 09 - Export predictions to GeoPackage

```bash
python scripts/09_export_predictions_gpkg.py \
  --pred outputs/predictions/tiles/predictions_tiles.csv \
  --tiles_meta outputs/inference/tiles_full/tiles_meta.json \
  --out_gpkg outputs/predictions/full/detections_merged.gpkg \
  --layer detections \
  --dedup \
  --dedup_by_class
```

## Notes

- Default raw inputs are expected under:
  - `data/raw/raster/wv3.tif`
  - `data/raw/vectors/labeling_WV3.gpkg`
- The training script uses base weights from `models/weights/yolo11n.pt`.
