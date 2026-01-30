# scripts/01_build_yolo_dataset.py
from __future__ import annotations

from pathlib import Path
import sys
import json
import math
import random

import numpy as np
import geopandas as gpd
import fiona
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import cv2

# NEW: robust multi-page TIFF writer
try:
    import tifffile as tiff
except ImportError as e:
    raise ImportError(
        "Missing dependency 'tifffile'. Install with: pip install tifffile"
    ) from e


# -------------------
# Config
# -------------------
TILE_SIZE = 512
STRIDE = 384  # overlap since STRIDE < TILE_SIZE

MIN_AREA_FRACTION = 0.25
MIN_BBOX_PX = 6

# WV3 is 8-band here (1-indexed bands for rasterio)
EXPORT_BANDS_1_INDEXED = [1, 2, 3, 4, 5, 6, 7, 8]

# RGB for visualization only
RGB_BANDS_1_INDEXED = [5, 3, 2]

CLASS_NAME = "circle"
CLASS_ID = 0

BLOCK_TILES = 6
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_SEED = 42


def to_uint8_per_band(img: np.ndarray) -> np.ndarray:
    """
    Percentile stretch each band to 8-bit.
    img: CHW (bands, H, W) -> uint8 CHW
    """
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.uint8)
    for b in range(img.shape[0]):
        band = img[b]
        valid = np.isfinite(band)
        if valid.sum() < 50:
            out[b] = 0
            continue
        v = band[valid]
        lo, hi = np.percentile(v, (2, 98))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
            out[b] = 0
            continue
        scaled = (band - lo) / (hi - lo)
        scaled = np.clip(scaled, 0, 1)
        scaled[~valid] = 0
        out[b] = (scaled * 255).astype(np.uint8)
    return out


def yolo_line(px_min: int, py_min: int, px_max: int, py_max: int, w: int, h: int) -> str:
    bw = (px_max - px_min) / w
    bh = (py_max - py_min) / h
    cx = (px_min + px_max) / 2 / w
    cy = (py_min + py_max) / 2 / h
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def choose_split(tile_row: int, tile_col: int) -> str:
    br = tile_row // BLOCK_TILES
    bc = tile_col // BLOCK_TILES
    key = (br * 1000003) ^ bc
    rng = random.Random(key + SPLIT_SEED)
    r = rng.random()
    t = SPLIT_RATIOS["train"]
    v = SPLIT_RATIOS["val"]
    if r < t:
        return "train"
    if r < t + v:
        return "val"
    return "test"


def write_ms_tiff_multipage(path: Path, chw_uint8: np.ndarray) -> None:
    """
    Write CHW uint8 as multi-page TIFF: C pages, each page is (H,W).
    This avoids PIL "samples per pixel" decode limits.
    """
    if chw_uint8.dtype != np.uint8 or chw_uint8.ndim != 3:
        raise ValueError("Expected uint8 CHW array")
    # tiff expects pages as first axis -> (C,H,W) is perfect
    tiff.imwrite(
        str(path),
        chw_uint8,
        photometric="minisblack",
        metadata=None,
    )


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from paths import RAW_DIR, PROCESSED_DIR


def main() -> None:
    raster_path = RAW_DIR / "raster" / "wv3.tif"
    gpkg_path = RAW_DIR / "vectors" / "labeling_WV3.gpkg"

    out_root = PROCESSED_DIR / "yolo_dataset_ms8_v1"

    # Model input tiles (8-band multi-page TIFF)
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Visualization-only RGB tiles (PNG)
    for split in ["train", "val", "test"]:
        (out_root / "images_rgb" / split).mkdir(parents=True, exist_ok=True)

    layers = fiona.listlayers(gpkg_path)
    layer = layers[0]
    gdf = gpd.read_file(gpkg_path, layer=layer)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.is_valid].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf["orig_area"] = gdf.geometry.area

    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        W, H = src.width, src.height
        tiles_written = {"train": 0, "val": 0, "test": 0}
        labels_written = {"train": 0, "val": 0, "test": 0}
        skipped_empty = 0

        n_cols = math.floor((W - TILE_SIZE) / STRIDE) + 1
        n_rows = math.floor((H - TILE_SIZE) / STRIDE) + 1

        for r in range(n_rows):
            top = r * STRIDE
            for c in range(n_cols):
                left = c * STRIDE
                win = Window(left, top, TILE_SIZE, TILE_SIZE)

                x_min, y_min, x_max, y_max = rasterio.windows.bounds(win, src.transform)
                tile_poly = box(x_min, y_min, x_max, y_max)

                hits = gdf[gdf.intersects(tile_poly)]
                if hits.empty:
                    continue

                split = choose_split(r, c)
                if SPLIT_RATIOS.get(split, 0.0) == 0.0:
                    continue

                label_lines = []
                for _, row in hits.iterrows():
                    geom = row.geometry
                    clipped = geom.intersection(tile_poly)
                    if clipped.is_empty:
                        continue

                    frac = clipped.area / (row.orig_area if row.orig_area > 0 else clipped.area)
                    if frac < MIN_AREA_FRACTION:
                        continue

                    gxmin, gymin, gxmax, gymax = clipped.bounds

                    row_min, col_min = src.index(gxmin, gymax)
                    row_max, col_max = src.index(gxmax, gymin)

                    px_min = int(np.clip(col_min - left, 0, TILE_SIZE - 1))
                    px_max = int(np.clip(col_max - left, 0, TILE_SIZE - 1))
                    py_min = int(np.clip(row_min - top, 0, TILE_SIZE - 1))
                    py_max = int(np.clip(row_max - top, 0, TILE_SIZE - 1))

                    if px_max <= px_min or py_max <= py_min:
                        continue
                    if (px_max - px_min) < MIN_BBOX_PX or (py_max - py_min) < MIN_BBOX_PX:
                        continue

                    label_lines.append(yolo_line(px_min, py_min, px_max, py_max, TILE_SIZE, TILE_SIZE))

                if not label_lines:
                    skipped_empty += 1
                    continue

                img_ms = src.read(EXPORT_BANDS_1_INDEXED, window=win)  # CHW
                img_ms8 = to_uint8_per_band(img_ms)

                img_rgb = src.read(RGB_BANDS_1_INDEXED, window=win)  # CHW
                img_rgb8 = to_uint8_per_band(img_rgb)
                img_rgb_hwc = np.transpose(img_rgb8, (1, 2, 0))

                tile_id = f"r{r:04d}_c{c:04d}"
                ms_name = f"tile_{tile_id}.tif"
                rgb_name = f"tile_{tile_id}.png"
                lbl_name = f"tile_{tile_id}.txt"

                out_ms = out_root / "images" / split / ms_name
                out_rgb = out_root / "images_rgb" / split / rgb_name
                out_lbl = out_root / "labels" / split / lbl_name

                write_ms_tiff_multipage(out_ms, img_ms8)
                cv2.imwrite(str(out_rgb), cv2.cvtColor(img_rgb_hwc, cv2.COLOR_RGB2BGR))
                out_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

                tiles_written[split] += 1
                labels_written[split] += len(label_lines)

        data_yaml = (
            f"path: {out_root.as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"channels: 8\n"
            f"names:\n"
            f"  0: {CLASS_NAME}\n"
        )
        (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

        meta = {
            "raster": str(raster_path),
            "gpkg": str(gpkg_path),
            "gpkg_layer": layer,
            "crs": str(src.crs),
            "raster_size_px": [W, H],
            "bands_count": src.count,
            "export_bands_1_indexed": EXPORT_BANDS_1_INDEXED,
            "rgb_bands_1_indexed": RGB_BANDS_1_INDEXED,
            "pixel_size": [src.transform.a, abs(src.transform.e)],
            "tile_size": TILE_SIZE,
            "stride": STRIDE,
            "min_area_fraction": MIN_AREA_FRACTION,
            "min_bbox_px": MIN_BBOX_PX,
            "split": {"ratios": SPLIT_RATIOS, "seed": SPLIT_SEED, "block_tiles": BLOCK_TILES},
            "stats": {
                "tiles_written": tiles_written,
                "labels_written": labels_written,
                "tiles_skipped_no_labels": skipped_empty,
            },
        }
        (out_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("DONE")
        print("tiles_written:", tiles_written)
        print("labels_written:", labels_written)
        print("tiles_skipped_no_labels:", skipped_empty)
        print("dataset:", out_root)
        print("note: model trains on images/*/*.tif (8-band multipage). RGB previews are in images_rgb/*/*.png")


if __name__ == "__main__":
    main()
