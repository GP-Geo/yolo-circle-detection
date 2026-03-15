# pipeline/01_build_yolo_dataset.py
from __future__ import annotations

from pathlib import Path
import sys
import json
import math
import random

import numpy as np
import geopandas as gpd
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

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import load_config, resolve_path
from paths import PROCESSED_DIR


def is_divisible_by_32(n: int) -> bool:
    """Check if n is divisible by 32 (required for YOLO downsampling layers)."""
    return n > 0 and n % 32 == 0


def snap_to_multiple_of_32(n: float) -> int:
    """Snap to nearest multiple of 32."""
    return max(32, int(round(n / 32) * 32))


def compute_tile_size_px(cfg_tiling: dict, pixel_size_m: float) -> int:
    if cfg_tiling.get("use_ground_size", False):
        tile_size_m = float(cfg_tiling.get("tile_size_m", 0))
        if tile_size_m <= 0:
            raise ValueError("tiling.tile_size_m must be > 0 when use_ground_size is true.")
        px = tile_size_m / pixel_size_m
        return snap_to_multiple_of_32(px)
    tile_size_px = int(cfg_tiling.get("tile_size_px", 0))
    if tile_size_px <= 0:
        raise ValueError("tiling.tile_size_px must be > 0.")
    if not is_divisible_by_32(tile_size_px):
        raise ValueError("tiling.tile_size_px must be divisible by 32 (required for YOLO architecture).")
    return tile_size_px


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


def yolo_line(px_min: int, py_min: int, px_max: int, py_max: int, w: int, h: int, class_id: int) -> str:
    bw = (px_max - px_min) / w
    bh = (py_max - py_min) / h
    cx = (px_min + px_max) / 2 / w
    cy = (py_min + py_max) / 2 / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def choose_split(tile_row: int, tile_col: int, block_tiles: int, split_seed: int, split_ratios: dict) -> str:
    br = tile_row // block_tiles
    bc = tile_col // block_tiles
    key = (br * 1000003) ^ bc
    rng = random.Random(key + split_seed)
    r = rng.random()
    t = split_ratios["train"]
    v = split_ratios["val"]
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


def main() -> None:
    cfg_path = ROOT / "configs" / "pipeline.yaml"
    cfg = load_config(cfg_path)

    run_id = cfg.get("run_id", "run")
    inputs = cfg.get("inputs", [])
    if not inputs:
        raise ValueError("Config must include at least one input under 'inputs'.")

    bands_cfg = cfg.get("bands", {})
    ms_bands = bands_cfg.get("ms", [])
    rgb_bands = bands_cfg.get("rgb", [])
    if not ms_bands:
        raise ValueError("bands.ms must be set (1-indexed band list).")
    if not rgb_bands:
        raise ValueError("bands.rgb must be set (1-indexed band list).")

    labels_cfg = cfg.get("labels", {})
    min_area_fraction = float(labels_cfg.get("min_area_fraction", 0.25))
    min_bbox_px = int(labels_cfg.get("min_bbox_px", 6))
    class_name = str(labels_cfg.get("class_name", "circle"))
    class_id = int(labels_cfg.get("class_id", 0))

    split_cfg = cfg.get("split", {})
    split_ratios = split_cfg.get("ratios", {"train": 0.6, "val": 0.2, "test": 0.2})
    split_seed = int(split_cfg.get("seed", 42))
    block_tiles = int(split_cfg.get("block_tiles", 6))

    tiling_cfg = cfg.get("tiling", {})
    stride_frac = float(tiling_cfg.get("stride_frac", 0.75))
    if not (0 < stride_frac <= 1.0):
        raise ValueError("tiling.stride_frac must be in (0, 1].")

    out_root = PROCESSED_DIR / f"yolo_dataset_{run_id}"

    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "images_rgb" / split).mkdir(parents=True, exist_ok=True)

    tiles_written = {"train": 0, "val": 0, "test": 0}
    labels_written = {"train": 0, "val": 0, "test": 0}
    skipped_empty = 0

    tile_size_px = None
    stride_px = None
    inputs_meta = []

    for inp in inputs:
        input_id = inp.get("id")
        if not input_id:
            raise ValueError("Each input must have an 'id'.")

        raster_path = resolve_path(ROOT, inp["raster"])
        gpkg_path = resolve_path(ROOT, inp["masks_gpkg"])
        layer = inp.get("masks_layer", "")

        if not raster_path.exists():
            raise FileNotFoundError(f"Raster not found: {raster_path}")
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GPKG not found: {gpkg_path}")

        gdf = gpd.read_file(gpkg_path, layer=layer if layer else None)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[gdf.is_valid].copy()
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)
        gdf["orig_area"] = gdf.geometry.area

        with rasterio.open(raster_path) as src:
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            px_size_x = abs(src.transform.a)
            px_size_y = abs(src.transform.e)
            px_size_m = (px_size_x + px_size_y) / 2.0

            this_tile_size_px = compute_tile_size_px(tiling_cfg, px_size_m)
            this_stride_px = int(round(this_tile_size_px * stride_frac))
            if this_stride_px <= 0:
                raise ValueError("Computed stride_px is invalid.")

            if tile_size_px is None:
                tile_size_px = this_tile_size_px
                stride_px = this_stride_px
            else:
                if this_tile_size_px != tile_size_px or this_stride_px != stride_px:
                    raise ValueError(
                        "All inputs must resolve to the same tile_size_px and stride_px. "
                        "Adjust tiling config or use per-resolution datasets."
                    )

            W, H = src.width, src.height
            n_cols = math.floor((W - tile_size_px) / stride_px) + 1
            n_rows = math.floor((H - tile_size_px) / stride_px) + 1
            eff_block_tiles = max(1, min(block_tiles, n_rows, n_cols))

            for r in range(n_rows):
                top = r * stride_px
                for c in range(n_cols):
                    left = c * stride_px
                    win = Window(left, top, tile_size_px, tile_size_px)

                    x_min, y_min, x_max, y_max = rasterio.windows.bounds(win, src.transform)
                    tile_poly = box(x_min, y_min, x_max, y_max)

                    hits = gdf[gdf.intersects(tile_poly)]
                    if hits.empty:
                        continue

                    split = choose_split(r, c, eff_block_tiles, split_seed, split_ratios)
                    if split_ratios.get(split, 0.0) == 0.0:
                        continue

                    label_lines = []
                    for _, row in hits.iterrows():
                        geom = row.geometry
                        clipped = geom.intersection(tile_poly)
                        if clipped.is_empty:
                            continue

                        frac = clipped.area / (row.orig_area if row.orig_area > 0 else clipped.area)
                        if frac < min_area_fraction:
                            continue

                        gxmin, gymin, gxmax, gymax = clipped.bounds

                        row_min, col_min = src.index(gxmin, gymax)
                        row_max, col_max = src.index(gxmax, gymin)

                        px_min = int(np.clip(col_min - left, 0, tile_size_px - 1))
                        px_max = int(np.clip(col_max - left, 0, tile_size_px - 1))
                        py_min = int(np.clip(row_min - top, 0, tile_size_px - 1))
                        py_max = int(np.clip(row_max - top, 0, tile_size_px - 1))

                        if px_max <= px_min or py_max <= py_min:
                            continue
                        if (px_max - px_min) < min_bbox_px or (py_max - py_min) < min_bbox_px:
                            continue

                        label_lines.append(
                            yolo_line(px_min, py_min, px_max, py_max, tile_size_px, tile_size_px, class_id)
                        )

                    if not label_lines:
                        skipped_empty += 1
                        continue

                    img_ms = src.read(ms_bands, window=win)  # CHW
                    img_ms8 = to_uint8_per_band(img_ms)

                    img_rgb = src.read(rgb_bands, window=win)  # CHW
                    img_rgb8 = to_uint8_per_band(img_rgb)
                    img_rgb_hwc = np.transpose(img_rgb8, (1, 2, 0))

                    tile_id = f"{input_id}_r{r:04d}_c{c:04d}"
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

            inputs_meta.append(
                {
                    "id": input_id,
                    "raster": str(raster_path),
                    "gpkg": str(gpkg_path),
                    "gpkg_layer": layer,
                    "crs": str(src.crs),
                    "raster_size_px": [W, H],
                    "bands_count": src.count,
                    "pixel_size": [px_size_x, px_size_y],
                }
            )

    data_yaml = (
        f"path: {out_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"channels: {len(ms_bands)}\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    (out_root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    meta = {
        "run_id": run_id,
        "inputs": inputs_meta,
        "export_bands_1_indexed": ms_bands,
        "rgb_bands_1_indexed": rgb_bands,
        "tile_size": tile_size_px,
        "stride": stride_px,
        "min_area_fraction": min_area_fraction,
        "min_bbox_px": min_bbox_px,
        "split": {"ratios": split_ratios, "seed": split_seed, "block_tiles": block_tiles},
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
    print("note: model trains on images/*/*.tif. RGB previews are in images_rgb/*/*.png")


if __name__ == "__main__":
    main()
