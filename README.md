# yolo-circle-detection
# YOLO Circle Detection

# YOLO Circle Detection Pipeline

End-to-end YOLO-based object detection pipeline for identifying circular features in large-scale geospatial imagery.  
The pipeline covers dataset construction from GIS vectors, training, tiled inference, post-processing with NMS, visualization, and export back to GIS formats.

The repository is intentionally lean:
- Code, configs, and documentation are versioned
- Data, weights, runs, and outputs are local-only and ignored via `.gitignore`



## Repository layout

- `scripts/` Pipeline scripts (00 to 09)
- `configs/` YAML configs for dataset and training
- `notebooks/` Optional exploration notebooks
- `data/` Local only (ignored by git)
- `weights/`, `models/`, `runs/`, `outputs/` Local only (ignored by git)

---

## Prerequisites

- Python 3.10+ recommended
- `pip` or `conda`
- GDAL dependencies may be required depending on your stack (for `rasterio`, `fiona`, `geopandas`)

---

## Install

### Option A: venv + pip (recommended for simplicity)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

