# pipeline/11_ingest_corrections.py
"""Ingest corrected GeoPackage detections into the YOLO training dataset.

This is the HITL (Human-In-The-Loop) bridge script — step 5 of the iterative
refinement workflow. After a user reviews and corrects detections in QGIS, this
script converts the corrected bounding box polygons back into YOLO training tiles
and labels, then merges them into the existing dataset.

Workflow context:
  1. Run 12_run_iteration.py  → produces detections GeoPackage
  2. Open GeoPackage in QGIS, review / add / delete / adjust bboxes, save
  3. Run this script           → ingests corrections into dataset
  4. Run 04_train.py           → retrains model on expanded dataset
  5. Repeat from step 1 until quality is satisfactory

Usage:
    python pipeline/11_ingest_corrections.py \\
        --corrections_gpkg data/corrections/round1_corrected.gpkg \\
        --raster data/external/active/s2_large_tile.tif \\
        --round_id round1

Deduplication: if a tile already exists from a previous round, the corrected
version replaces it (controlled by hitl.merge_strategy in pipeline.yaml).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

try:
    import tifffile as tiff
except ImportError as e:
    raise ImportError(
        "Missing dependency 'tifffile'. Install with: pip install tifffile"
    ) from e

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import setup_logger, load_config, resolve_path

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers (reuse same logic as 01_build_yolo_dataset.py)
# ---------------------------------------------------------------------------


def is_divisible_by_32(n: int) -> bool:
    """Check if n is divisible by 32 (required for YOLO downsampling layers)."""
    return n > 0 and n % 32 == 0


def snap_to_multiple_of_32(n: float) -> int:
    """Snap to nearest multiple of 32."""
    return max(32, int(round(n / 32) * 32))


def compute_tile_size_px(cfg_tiling: dict, pixel_size_m: float) -> int:
    """Derive tile size in pixels from config and raster pixel size."""
    if cfg_tiling.get("use_ground_size", False):
        tile_size_m = float(cfg_tiling.get("tile_size_m", 0))
        if tile_size_m <= 0:
            raise ValueError("tiling.tile_size_m must be > 0 when use_ground_size is true.")
        return snap_to_multiple_of_32(tile_size_m / pixel_size_m)
    tile_size_px = int(cfg_tiling.get("tile_size_px", 0))
    if tile_size_px <= 0:
        raise ValueError("tiling.tile_size_px must be > 0.")
    if not is_divisible_by_32(tile_size_px):
        raise ValueError("tiling.tile_size_px must be divisible by 32 (YOLO requirement).")
    return tile_size_px


def to_uint8_per_band(img: np.ndarray) -> np.ndarray:
    """Percentile-stretch each band to uint8. img: (C,H,W) -> uint8 (C,H,W)."""
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
        scaled = np.clip((band - lo) / (hi - lo), 0, 1)
        scaled[~valid] = 0
        out[b] = (scaled * 255).astype(np.uint8)
    return out


def write_ms_tiff_multipage(path: Path, chw_uint8: np.ndarray) -> None:
    """Write uint8 CHW array as a multi-page TIFF (avoids PIL channel limits)."""
    if chw_uint8.dtype != np.uint8 or chw_uint8.ndim != 3:
        raise ValueError("Expected uint8 CHW array.")
    tiff.imwrite(str(path), chw_uint8, photometric="minisblack", metadata=None)


def yolo_line(
    px_min: int, py_min: int, px_max: int, py_max: int, w: int, h: int, class_id: int
) -> str:
    """Format a bounding box as a YOLO label line (normalized cx cy bw bh)."""
    bw = (px_max - px_min) / w
    bh = (py_max - py_min) / h
    cx = (px_min + px_max) / 2 / w
    cy = (py_min + py_max) / 2 / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def choose_split(
    tile_row: int, tile_col: int, block_tiles: int, split_seed: int, split_ratios: dict
) -> str:
    """Deterministic block-based train/val/test assignment (matches 01_build_yolo_dataset)."""
    br = tile_row // block_tiles
    bc = tile_col // block_tiles
    key = (br * 1000003) ^ bc
    rng = random.Random(key + split_seed)
    r = rng.random()
    t = split_ratios.get("train", 0.7)
    v = split_ratios.get("val", 0.3)
    if r < t:
        return "train"
    if r < t + v:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Core ingestion logic (exposed for testing)
# ---------------------------------------------------------------------------


def ingest_corrections(
    corrections_gpkg: Path,
    raster_path: Path,
    dataset_dir: Path,
    tiling_cfg: dict,
    bands_cfg: dict,
    labels_cfg: dict,
    split_cfg: dict,
    round_id: str,
    input_id: str = "corrections",
    corrections_layer: str | None = None,
    include_negatives: bool = False,
    tile_bounds: tuple | None = None,
) -> dict:
    """
    Ingest a corrected GeoPackage into an existing YOLO dataset.

    Returns a summary dict with tile counts for the manifest.
    """
    ms_bands: list[int] = bands_cfg.get("ms", [1, 2, 3, 4])
    rgb_bands: list[int] = bands_cfg.get("rgb", [4, 3, 2])
    min_area_fraction: float = float(labels_cfg.get("min_area_fraction", 0.25))
    min_bbox_px: int = int(labels_cfg.get("min_bbox_px", 4))
    class_id: int = int(labels_cfg.get("class_id", 0))

    split_ratios: dict = split_cfg.get("ratios", {"train": 0.7, "val": 0.3, "test": 0.0})
    split_seed: int = int(split_cfg.get("seed", 42))
    block_tiles: int = int(split_cfg.get("block_tiles", 6))
    stride_frac: float = float(tiling_cfg.get("stride_frac", 0.25))

    # Load corrections
    layer = corrections_layer or None
    logger.info(f"Loading corrections from: {corrections_gpkg}")
    gdf = gpd.read_file(corrections_gpkg, layer=layer)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.is_valid].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf["orig_area"] = gdf.geometry.area
    logger.info(f"  Loaded {len(gdf)} correction geometries")

    if gdf.empty:
        if not include_negatives or tile_bounds is None:
            logger.warning("No valid geometries — nothing to ingest.")
            return {"tiles_new": 0, "tiles_replaced": 0, "tiles_skipped_no_labels": 0}
        logger.info("No geometries — writing negative tiles within review tile bounds.")

    tiles_written = 0
    tiles_replaced = 0
    tiles_skipped = 0

    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        px_size_x = abs(src.transform.a)
        px_size_y = abs(src.transform.e)
        px_size_m = (px_size_x + px_size_y) / 2.0

        tile_size_px = compute_tile_size_px(tiling_cfg, px_size_m)
        stride_px = int(round(tile_size_px * stride_frac))
        if stride_px <= 0:
            raise ValueError("Computed stride_px is invalid.")

        W, H = src.width, src.height
        n_cols = math.ceil((W - tile_size_px) / stride_px) + 1
        n_rows = math.ceil((H - tile_size_px) / stride_px) + 1
        eff_block_tiles = max(1, min(block_tiles, n_rows, n_cols))

        logger.info(f"Raster: {W}×{H}px | tile: {tile_size_px}px | stride: {stride_px}px")
        logger.info(f"Grid: {n_rows}×{n_cols} = {n_rows * n_cols} tiles")

        try:
            from tqdm import tqdm
            row_iter = tqdm(range(n_rows), desc=f"Ingesting {input_id}", unit="row")
        except ImportError:
            row_iter = range(n_rows)

        # Pre-compute review tile polygon for spatial filtering (negatives case)
        review_poly = box(*tile_bounds) if tile_bounds is not None else None

        for r in row_iter:
            top = r * stride_px
            for c in range(n_cols):
                left = c * stride_px
                win = Window(left, top, tile_size_px, tile_size_px)

                x_min, y_min, x_max, y_max = rasterio.windows.bounds(win, src.transform)
                tile_poly = box(x_min, y_min, x_max, y_max)

                # Skip tiles not fully contained within the review tile bounds.
                # Using within() (not intersects()) ensures no training tile extends
                # outside the annotated chip area, which would introduce unlabeled circles.
                if review_poly is not None and not tile_poly.within(review_poly):
                    continue

                hits = gdf[gdf.intersects(tile_poly)] if not gdf.empty else gdf
                if hits.empty and not include_negatives:
                    continue

                split = choose_split(r, c, eff_block_tiles, split_seed, split_ratios)
                if split_ratios.get(split, 0.0) == 0.0:
                    continue

                label_lines: list[str] = []
                for _, row in hits.iterrows():
                    geom = row.geometry
                    clipped = geom.intersection(tile_poly)
                    if clipped.is_empty:
                        continue

                    orig_area = row.orig_area if row.orig_area > 0 else clipped.area
                    if clipped.area / orig_area < min_area_fraction:
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

                if not label_lines and not include_negatives:
                    tiles_skipped += 1
                    continue

                img_ms = src.read(ms_bands, window=win)
                img_ms8 = to_uint8_per_band(img_ms)

                img_rgb = src.read(rgb_bands, window=win)
                img_rgb8 = to_uint8_per_band(img_rgb)
                img_rgb_hwc = np.transpose(img_rgb8, (1, 2, 0))

                tile_id = f"{input_id}_r{r:04d}_c{c:04d}"
                ms_name = f"tile_{tile_id}.tif"
                rgb_name = f"tile_{tile_id}.png"
                lbl_name = f"tile_{tile_id}.txt"

                out_ms = dataset_dir / "images" / split / ms_name
                out_rgb = dataset_dir / "images_rgb" / split / rgb_name
                out_lbl = dataset_dir / "labels" / split / lbl_name

                existed = out_ms.exists()
                write_ms_tiff_multipage(out_ms, img_ms8)
                cv2.imwrite(str(out_rgb), cv2.cvtColor(img_rgb_hwc, cv2.COLOR_RGB2BGR))
                out_lbl.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

                if existed:
                    tiles_replaced += 1
                else:
                    tiles_written += 1

    return {
        "tiles_new": tiles_written,
        "tiles_replaced": tiles_replaced,
        "tiles_skipped_no_labels": tiles_skipped,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest corrected GeoPackage (from QGIS) into the YOLO training dataset. "
            "Step 5 of the HITL iterative refinement workflow."
        )
    )
    parser.add_argument(
        "--corrections_gpkg",
        default="",
        help="Corrected GeoPackage path (e.g. data/corrections/round1_corrected.gpkg)",
    )
    parser.add_argument(
        "--corrections_dir",
        default="",
        help=(
            "Review tiles directory produced by 13_split_for_review.py. "
            "Ingests all detections_corrected.gpkg files found inside it. "
            "Mutually exclusive with --corrections_gpkg."
        ),
    )
    parser.add_argument(
        "--raster",
        required=True,
        help="Raster the corrections were made on (must match the GeoPackage CRS).",
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Pipeline config YAML (default: configs/pipeline.yaml).",
    )
    parser.add_argument(
        "--dataset_dir",
        default="",
        help="YOLO dataset directory. Defaults to data/processed/yolo_dataset_<run_id>.",
    )
    parser.add_argument(
        "--corrections_layer",
        default="",
        help="Layer name inside the corrections GeoPackage (default: first layer).",
    )
    parser.add_argument(
        "--round_id",
        default="",
        help="Correction round identifier (e.g. round1). Auto-increments if omitted.",
    )
    parser.add_argument(
        "--include_negatives",
        action="store_true",
        help=(
            "Write empty-label tiles for review tiles that have zero corrections. "
            "Teaches the model what a negative area looks like. "
            "Only works with --corrections_dir (requires chip.tif in each tile folder)."
        ),
    )
    parser.add_argument(
        "--input_id",
        default="corrections",
        help="Prefix used in tile file names for this correction batch (default: corrections).",
    )
    args = parser.parse_args()

    if not args.corrections_gpkg and not args.corrections_dir:
        raise ValueError("Provide either --corrections_gpkg or --corrections_dir.")
    if args.corrections_gpkg and args.corrections_dir:
        raise ValueError("--corrections_gpkg and --corrections_dir are mutually exclusive.")

    from paths import PROJECT_ROOT, PROCESSED_DIR

    root = PROJECT_ROOT
    cfg_path = resolve_path(root, args.config)
    cfg = load_config(cfg_path)
    run_id = cfg.get("run_id", "run")

    raster_path = resolve_path(root, args.raster)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    if args.dataset_dir:
        dataset_dir = resolve_path(root, args.dataset_dir)
    else:
        dataset_dir = PROCESSED_DIR / f"yolo_dataset_{run_id}"

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Run 01_build_yolo_dataset.py first to create the base dataset."
        )

    # Build list of (gpkg_path, input_id) pairs to ingest
    if args.corrections_gpkg:
        gpkg_path = resolve_path(root, args.corrections_gpkg)
        if not gpkg_path.exists():
            raise FileNotFoundError(f"Corrections GeoPackage not found: {gpkg_path}")
        gpkg_list = [(gpkg_path, args.input_id or "corrections", None)]
    else:
        corrections_dir = resolve_path(root, args.corrections_dir)
        if not corrections_dir.exists():
            raise FileNotFoundError(f"Corrections directory not found: {corrections_dir}")
        found = sorted(corrections_dir.rglob("detections_corrected.gpkg"))
        if not found:
            raise FileNotFoundError(
                f"No detections_corrected.gpkg files found under: {corrections_dir}"
            )
        logger.info(f"Found {len(found)} corrected tile(s) in {corrections_dir}")
        # Use parent folder name (tile_rXXX_cXXX) as input_id for unique tile naming.
        # chip.tif is required to restrict tiling to the corrected chip area only.
        missing_chips = [p for p in found if not (p.parent / "chip.tif").exists()]
        if missing_chips:
            raise FileNotFoundError(
                f"chip.tif missing for {len(missing_chips)} tile(s) — cannot restrict "
                f"tiling to chip bounds:\n" + "\n".join(f"  {p.parent}" for p in missing_chips)
            )
        gpkg_list = [(p, p.parent.name, p.parent / "chip.tif") for p in found]

    # Determine round_id (auto-increment from manifest)
    manifest_path = dataset_dir / "corrections_manifest.json"
    if manifest_path.exists():
        manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"rounds": []}

    round_id = args.round_id or f"round{len(manifest['rounds']) + 1}"

    logger.info("=" * 80)
    logger.info("YOLO Circle Detection - Ingest Corrections (HITL)")
    logger.info("=" * 80)
    logger.info(f"Round:       {round_id}")
    logger.info(f"Raster:      {raster_path}")
    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"GPKGs to ingest: {len(gpkg_list)}")

    totals = {"tiles_new": 0, "tiles_replaced": 0, "tiles_skipped_no_labels": 0}

    for gpkg_path, input_id, chip_path in gpkg_list:
        logger.info(f"  Ingesting: {gpkg_path.name} (input_id={input_id})")

        # Restrict tiling to the corrected chip area only (chip.tif required).
        tile_bounds = None
        if chip_path is not None:
            with rasterio.open(chip_path) as chip_src:
                tile_bounds = chip_src.bounds

        summary = ingest_corrections(
            corrections_gpkg=gpkg_path,
            raster_path=raster_path,
            dataset_dir=dataset_dir,
            tiling_cfg=cfg.get("tiling", {}),
            bands_cfg=cfg.get("bands", {}),
            labels_cfg=cfg.get("labels", {}),
            split_cfg=cfg.get("split", {}),
            round_id=round_id,
            input_id=input_id,
            corrections_layer=args.corrections_layer or None,
            include_negatives=args.include_negatives,
            tile_bounds=tile_bounds,
        )
        for k in totals:
            totals[k] += summary[k]

    # Persist manifest
    round_entry = {
        "round_id": round_id,
        "date": str(date.today()),
        "n_gpkgs": len(gpkg_list),
        "raster": str(raster_path),
        **totals,
    }
    manifest["rounds"].append(round_entry)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("=" * 80)
    logger.info(f"Corrections ingested successfully (round: {round_id})")
    logger.info(f"  New tiles written:          {totals['tiles_new']}")
    logger.info(f"  Existing tiles replaced:    {totals['tiles_replaced']}")
    logger.info(f"  Tiles skipped (no labels):  {totals['tiles_skipped_no_labels']}")
    logger.info(f"  Manifest:                   {manifest_path}")
    logger.info("=" * 80)
    logger.info("Next step: retrain the model on the updated dataset:")
    logger.info("  python pipeline/04_train.py")


if __name__ == "__main__":
    main()
