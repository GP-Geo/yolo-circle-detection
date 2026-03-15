from __future__ import annotations

import json
from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import torch
from shapely.geometry import box as sbox
from rasterio.transform import xy as txy

try:
    from torchvision.ops import nms as _tv_nms
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

# Add utils to path for logging
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import setup_logger, resolve_path

logger = setup_logger(__name__)


def spatial_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
    cell_size_px: int = 512,
) -> np.ndarray:
    """
    Spatially partitioned NMS.

    Divides the prediction space into a grid of cells, runs NMS independently
    per cell, then merges results. Each box participates in NMS only in its
    home cell (centroid-based assignment) plus a border region for context.
    This reduces complexity from O(N²) global to O((N/k)²)*k ≈ O(N²/k).

    cell_size_px: grid cell size in global pixel coords (default 512).
                  Border overlap = cell_size_px // 2 on each side.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    border_px = cell_size_px // 2

    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    x_max = float(boxes[:, 2].max())
    y_max = float(boxes[:, 3].max())
    n_cols = int(np.ceil(x_max / cell_size_px)) + 1
    n_rows = int(np.ceil(y_max / cell_size_px)) + 1

    kept: set[int] = set()

    for r in range(n_rows):
        for c in range(n_cols):
            # Cell bounds (centroid assignment region)
            cell_x1 = c * cell_size_px
            cell_y1 = r * cell_size_px
            cell_x2 = (c + 1) * cell_size_px
            cell_y2 = (r + 1) * cell_size_px

            # Extended bounds for NMS context (includes border from neighbors)
            ext_x1 = cell_x1 - border_px
            ext_y1 = cell_y1 - border_px
            ext_x2 = cell_x2 + border_px
            ext_y2 = cell_y2 + border_px

            # Boxes whose centroid is in this cell — these are the "home" boxes
            home_mask = (cx >= cell_x1) & (cx < cell_x2) & (cy >= cell_y1) & (cy < cell_y2)
            if not home_mask.any():
                continue

            # All boxes overlapping the extended cell (for NMS context)
            overlap_mask = ~(
                (boxes[:, 2] < ext_x1) | (boxes[:, 0] > ext_x2) |
                (boxes[:, 3] < ext_y1) | (boxes[:, 1] > ext_y2)
            )
            cell_idx = np.where(overlap_mask)[0]
            cell_boxes = boxes[cell_idx]
            cell_scores = scores[cell_idx]

            keep_local = nms_xyxy(cell_boxes, cell_scores, iou_thresh)
            kept_global = set(cell_idx[keep_local].tolist())

            # Only register home boxes — avoids double-counting at boundaries
            home_global = set(np.where(home_mask)[0].tolist())
            kept.update(kept_global & home_global)

    return np.array(sorted(kept), dtype=int)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    NMS in xyxy pixel coords.
    Uses torchvision.ops.nms when available (C++/MPS accelerated, ~100x faster).
    Falls back to pure numpy implementation.
    boxes: (N,4) xyxy global pixel coords
    scores: (N,)
    returns indices to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    if _TORCHVISION_AVAILABLE:
        boxes_t = torch.from_numpy(boxes).float()
        scores_t = torch.from_numpy(scores).float()
        keep = _tv_nms(boxes_t, scores_t, iou_thresh)
        return keep.numpy()

    # Numpy fallback
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def suppress_nested_prefer_outer(
    boxes: np.ndarray,
    scores: np.ndarray,
    coverage_thresh: float = 0.90,
    contain_tol_px: float = 3.0,
    score_margin: float = 0.0,
) -> np.ndarray:
    """
    Prefer outer (larger area) boxes.
    Drop a candidate box if it is mostly covered by an already-kept larger box.

    coverage = intersection(candidate, kept) / area(candidate)

    score_margin:
      Only drop inner box if score_inner <= score_outer + score_margin.
      Set 0.0 to always prefer outer regardless of confidence.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)
    areas = (np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None) + 1e-9).astype(np.float32)
    scores = scores.astype(np.float32)

    # Process big boxes first, tie-breaker: higher confidence first
    # lexsort sorts by last key first, so we provide ( -scores, -areas ) and get areas desc primary.
    order = np.lexsort((-scores, -areas))

    keep: list[int] = []
    for idx in order:
        idx = int(idx)
        drop = False

        for k in keep:
            # Only compare against kept boxes that are larger or equal area
            if areas[k] + 1e-9 < areas[idx]:
                continue

            # Near containment check (fast)
            if (
                x1[idx] >= x1[k] - contain_tol_px
                and y1[idx] >= y1[k] - contain_tol_px
                and x2[idx] <= x2[k] + contain_tol_px
                and y2[idx] <= y2[k] + contain_tol_px
            ):
                if scores[idx] <= scores[k] + score_margin:
                    drop = True
                    break

            # Coverage check (robust)
            xx1 = max(x1[idx], x1[k])
            yy1 = max(y1[idx], y1[k])
            xx2 = min(x2[idx], x2[k])
            yy2 = min(y2[idx], y2[k])

            iw = max(0.0, float(xx2 - xx1))
            ih = max(0.0, float(yy2 - yy1))
            inter = iw * ih

            coverage = inter / float(areas[idx])
            if coverage >= coverage_thresh:
                if scores[idx] <= scores[k] + score_margin:
                    drop = True
                    break

        if not drop:
            keep.append(idx)

    return np.array(keep, dtype=int)


def px_box_to_map_polygon(transform, x1, y1, x2, y2):
    """
    Convert xyxy pixel coords (col=x, row=y) into a map-space polygon using raster transform.
    """
    # Pixel corners -> map coords
    x_left, y_top = txy(transform, y1, x1, offset="ul")
    x_right, y_bottom = txy(transform, y2, x2, offset="lr")

    minx, maxx = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
    miny, maxy = (y_bottom, y_top) if y_bottom <= y_top else (y_top, y_bottom)

    return sbox(minx, miny, maxx, maxy)


def process_single_image(df, raster_path, out_dir, args, logger, input_id=None):
    """Process a single image's predictions with NMS and nested suppression."""
    try:
        from tqdm import tqdm as _tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    original_n = len(df)
    logger.info(f"Processing {original_n} predictions for {input_id or 'image'}")

    if args.min_conf > 0:
        df = df[df["conf"] >= args.min_conf].copy()
        if df.empty:
            logger.warning("All predictions filtered out by min_conf. Skipping.")
            return

    # 1) Spatially partitioned NMS per class_id
    classes = list(df.groupby("class_id").groups.keys())
    class_iter = (
        _tqdm(classes, desc="Spatial NMS per class", unit="class", leave=False)
        if _tqdm_available and len(classes) > 1
        else classes
    )
    kept_after_nms = []
    for class_id in class_iter:
        g = df[df["class_id"] == class_id]
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
        scores = g["conf"].to_numpy(dtype=np.float32)
        keep_idx = spatial_nms(boxes, scores, args.nms_iou, cell_size_px=args.nms_cell_size)
        kept_after_nms.append(g.iloc[keep_idx])

    df_nms = pd.concat(kept_after_nms, ignore_index=True)
    after_nms_n = len(df_nms)

    # 2) Nested suppression per class_id (prefer outer) — optional, off by default
    if not args.skip_nested:
        nms_classes = list(df_nms.groupby("class_id").groups.keys())
        nested_iter = (
            _tqdm(nms_classes, desc="Nested suppression", unit="class", leave=False)
            if _tqdm_available and len(nms_classes) > 1
            else nms_classes
        )
        kept_final = []
        for class_id in nested_iter:
            g = df_nms[df_nms["class_id"] == class_id]
            boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
            scores = g["conf"].to_numpy(dtype=np.float32)

            keep_idx = suppress_nested_prefer_outer(
                boxes,
                scores,
                coverage_thresh=args.coverage_thresh,
                contain_tol_px=args.contain_tol_px,
                score_margin=args.score_margin,
            )
            kept_final.append(g.iloc[keep_idx])

        out_df = pd.concat(kept_final, ignore_index=True).sort_values("conf", ascending=False)
    else:
        logger.info("Nested suppression skipped (use --nested_suppression to enable).")
        out_df = df_nms.sort_values("conf", ascending=False)
    final_n = len(out_df)

    # 3) Export to GPKG in raster CRS
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs

        geoms = [
            px_box_to_map_polygon(transform, row.x1, row.y1, row.x2, row.y2)
            for row in out_df.itertuples(index=False)
        ]

    gdf = gpd.GeoDataFrame(out_df.copy(), geometry=geoms, crs=crs)

    # Add input_id suffix to filename if processing multiple images
    suffix = f"_{input_id}" if input_id else ""
    out_gpkg = out_dir / f"detections_merged{suffix}.gpkg"
    gdf.to_file(out_gpkg, layer="detections", driver="GPKG")

    out_csv = out_dir / f"detections_merged{suffix}.csv"
    out_df.to_csv(out_csv, index=False)

    logger.info("=" * 80)
    logger.info(f"Merge & NMS completed successfully for {input_id or 'image'}!")
    logger.info("=" * 80)
    logger.info(f"Prediction counts:")
    logger.info(f"  Original: {original_n}")
    logger.info(f"  After NMS: {after_nms_n} ({100*(1-after_nms_n/original_n):.1f}% reduction)")
    logger.info(f"  After nested suppression: {final_n} ({100*(1-final_n/original_n):.1f}% total reduction)")
    logger.info(f"Parameters:")
    logger.info(f"  Min confidence: {args.min_conf}")
    logger.info(f"  NMS IOU threshold: {args.nms_iou}")
    logger.info(f"  Coverage threshold: {args.coverage_thresh}")
    logger.info(f"  Containment tolerance: {args.contain_tol_px}px")
    logger.info(f"  Score margin: {args.score_margin}")
    logger.info(f"Output files:")
    logger.info(f"  GeoPackage: {out_gpkg}")
    logger.info(f"  CSV: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge tile predictions with NMS + nested suppression (prefer outer), export GeoPackage."
    )
    parser.add_argument("--pred_csv", required=True, help="predictions_tiles.csv from 06_infer.py")
    parser.add_argument("--meta", default="data/processed/yolo_dataset_v1/dataset_meta.json", help="Dataset metadata json")
    parser.add_argument("--out", required=True, help="Output folder, e.g. outputs/predictions/merged/yolo11n_wv3_v1_full")
    parser.add_argument(
        "--input_id",
        default="",
        help="If meta has multiple inputs, set this to choose the raster (e.g. image1).",
    )

    parser.add_argument("--min_conf", type=float, default=0.3, help="Drop predictions below this confidence before merging")
    parser.add_argument("--nms_cell_size", type=int, default=512, help="Spatial NMS grid cell size in pixels (default: 512)")
    parser.add_argument("--skip_nested", action="store_true", default=True, help="Skip nested suppression (default: True — it removes <1%% of boxes but is very slow)")
    parser.add_argument("--nested_suppression", dest="skip_nested", action="store_false", help="Enable nested suppression (slow O(N²) Python loop)")
    parser.add_argument("--nms_iou", type=float, default=0.30, help="NMS IoU threshold (global pixel space)")
    parser.add_argument("--coverage_thresh", type=float, default=0.85, help="Mostly-contained coverage threshold")
    parser.add_argument("--contain_tol_px", type=float, default=3.0, help="Containment tolerance in pixels")
    parser.add_argument("--score_margin", type=float, default=0.0, help="Keep inner if it is higher than outer by this margin")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.append(str(root))

    from paths import PROJECT_ROOT

    root = PROJECT_ROOT
    pred_csv = resolve_path(root, args.pred_csv)
    meta_path = resolve_path(root, args.meta)
    out_dir = resolve_path(root, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_csv.exists():
        raise FileNotFoundError(f"pred_csv not found: {pred_csv}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta json not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    logger.info("=" * 80)
    logger.info("YOLO Circle Detection - Merge & NMS")
    logger.info("=" * 80)
    logger.info(f"Prediction CSV: {pred_csv}")
    logger.info(f"Metadata: {meta_path}")
    logger.info(f"Output directory: {out_dir}")

    try:
        df = pd.read_csv(pred_csv)
    except Exception as e:
        logger.error(f"Failed to read predictions CSV: {e}", exc_info=True)
        return

    if df.empty:
        logger.warning("No predictions found in CSV. Nothing to merge.")
        return

    original_n = len(df)
    logger.info(f"Loaded {original_n} predictions")

    if "raster" in meta:
        raster_path = Path(meta["raster"])
        if not raster_path.is_absolute():
            raster_path = (root / raster_path).resolve()
        # Process single raster
        process_single_image(df, raster_path, out_dir, args, logger, None)
    else:
        inputs = meta.get("inputs", [])
        if not inputs:
            raise ValueError("Meta json must include 'raster' or 'inputs'.")

        input_id = args.input_id.strip()
        if not input_id:
            # Try to infer from tile names: tile_<id>_r####_c####.tif
            prefixes = set()
            for name in df["tile"].dropna().astype(str).tolist():
                if name.startswith("tile_") and "_r" in name:
                    prefixes.add(name.split("_r")[0].replace("tile_", ""))
            if len(prefixes) == 1:
                input_id = prefixes.pop()
            elif len(prefixes) > 1:
                # Process all input prefixes
                logger.info(f"Found {len(prefixes)} input images: {sorted(prefixes)}")
                logger.info("Processing all images...")
                for prefix in sorted(prefixes):
                    logger.info("=" * 80)
                    logger.info(f"Processing input: {prefix}")
                    logger.info("=" * 80)
                    df_subset = df[df["tile"].astype(str).str.startswith(f"tile_{prefix}_")].copy()
                    if df_subset.empty:
                        logger.warning(f"No predictions found for {prefix}, skipping.")
                        continue

                    match = [i for i in inputs if i.get("id") == prefix]
                    if not match:
                        logger.warning(f"input_id not found in meta inputs: {prefix}, skipping.")
                        continue
                    raster_path = Path(match[0]["raster"])
                    if not raster_path.is_absolute():
                        raster_path = (root / raster_path).resolve()

                    process_single_image(df_subset, raster_path, out_dir, args, logger, prefix)
                return
            else:
                raise ValueError("Could not infer input_id from CSV. Use --input_id.")

        df = df[df["tile"].astype(str).str.startswith(f"tile_{input_id}_")].copy()
        if df.empty:
            print("No predictions found for input_id. Exiting.")
            return

        match = [i for i in inputs if i.get("id") == input_id]
        if not match:
            raise ValueError(f"input_id not found in meta inputs: {input_id}")
        raster_path = Path(match[0]["raster"])
        if not raster_path.is_absolute():
            raster_path = (root / raster_path).resolve()

        process_single_image(df, raster_path, out_dir, args, logger, input_id)


if __name__ == "__main__":
    main()
