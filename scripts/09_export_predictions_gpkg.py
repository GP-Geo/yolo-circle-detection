# scripts/09_export_predictions_gpkg.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as txy

import geopandas as gpd
from shapely.geometry import box as sbox

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import resolve_path, setup_logger

logger = setup_logger(__name__)


def _normalize_boxes_xyxy(df: pd.DataFrame) -> np.ndarray:
    x1 = df["x1"].to_numpy(dtype=np.float32)
    y1 = df["y1"].to_numpy(dtype=np.float32)
    x2 = df["x2"].to_numpy(dtype=np.float32)
    y2 = df["y2"].to_numpy(dtype=np.float32)

    xx1 = np.minimum(x1, x2)
    yy1 = np.minimum(y1, y2)
    xx2 = np.maximum(x1, x2)
    yy2 = np.maximum(y1, y2)
    return np.stack([xx1, yy1, xx2, yy2], axis=1)


def iou_1_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    area_b = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    union = np.maximum(area_a + area_b - inter, 1e-9)
    return inter / union


def coverage_small_by_big(small: np.ndarray, bigs: np.ndarray) -> np.ndarray:
    """
    coverage = intersection_area / area_small
    If coverage is high, the small box is likely a cut box sitting inside a full box.
    """
    x1 = np.maximum(small[0], bigs[:, 0])
    y1 = np.maximum(small[1], bigs[:, 1])
    x2 = np.minimum(small[2], bigs[:, 2])
    y2 = np.minimum(small[3], bigs[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_small = np.maximum((small[2] - small[0]) * (small[3] - small[1]), 1e-9)
    return inter / area_small


def drop_cut_boxes_prefer_outer(
    df: pd.DataFrame,
    contain_thr: float = 0.90,
    contain_tol_px: float = 3.0,
) -> pd.DataFrame:
    """
    Pass A (stride artifact killer):
    - Sort by area desc (then conf desc)
    - Keep big boxes first
    - Drop a candidate if it is basically inside an already kept box
      OR if >= contain_thr of its area is covered by a kept box.
    """
    if df.empty:
        return df

    boxes = _normalize_boxes_xyxy(df)
    scores = df["conf"].to_numpy(dtype=np.float32)
    areas = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)

    # biggest first, then higher conf
    order = np.lexsort((-scores, -areas)).astype(np.int64)

    keep_idx: list[int] = []
    kept_boxes: list[np.ndarray] = []

    for i in order:
        b = boxes[i]

        if not kept_boxes:
            keep_idx.append(i)
            kept_boxes.append(b)
            continue

        kb = np.stack(kept_boxes, axis=0)

        # fast near-containment check with tolerance
        x1, y1, x2, y2 = b
        inside_any = np.any(
            (x1 >= kb[:, 0] - contain_tol_px)
            & (y1 >= kb[:, 1] - contain_tol_px)
            & (x2 <= kb[:, 2] + contain_tol_px)
            & (y2 <= kb[:, 3] + contain_tol_px)
        )
        if inside_any:
            continue

        # robust coverage check
        cov = coverage_small_by_big(b, kb)
        if np.any(cov >= contain_thr):
            continue

        keep_idx.append(i)
        kept_boxes.append(b)

    return df.iloc[keep_idx].copy()


def nms_keep_highest_score(df: pd.DataFrame, iou_thr: float = 0.90) -> pd.DataFrame:
    """
    Pass B (true duplicates killer):
    Standard score-based NMS in global pixel space.
    If IoU >= iou_thr, keep highest conf and suppress the rest.
    """
    if df.empty:
        return df

    boxes = _normalize_boxes_xyxy(df)
    scores = df["conf"].to_numpy(dtype=np.float32)

    order = scores.argsort()[::-1].astype(np.int64)

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]
        ious = iou_1_to_many(boxes[i], boxes[rest])

        # suppress high-overlap boxes
        remain = rest[ious < iou_thr]
        order = remain

    return df.iloc[keep].copy()


def px_box_to_map_polygon(transform, x1, y1, x2, y2):
    """
    Convert xyxy pixel coords (col=x, row=y) into a map-space polygon using raster transform.
    Uses rasterio.transform.xy for correct corner handling.
    """
    x_left, y_top = txy(transform, y1, x1, offset="ul")
    x_right, y_bottom = txy(transform, y2, x2, offset="lr")

    minx, maxx = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
    miny, maxy = (y_bottom, y_top) if y_bottom <= y_top else (y_top, y_bottom)

    return sbox(minx, miny, maxx, maxy)


def process_single_image(df, raster_path, out_gpkg_base, args, logger, input_id=None):
    """Process a single image's predictions with deduplication and export to GeoPackage."""
    root = Path(__file__).resolve().parents[1]

    n0 = len(df)
    df = df[df["conf"] >= float(args.min_conf)].copy()
    n1 = len(df)

    if df.empty:
        logger.warning(f"No detections after min_conf filter for {input_id or 'image'}. Skipping.")
        return

    n2 = n1
    n3 = n1

    if args.dedup:
        if args.dedup_by_class and "class_id" in df.columns:
            parts = []
            for _, sub in df.groupby("class_id", sort=False):
                sub2 = drop_cut_boxes_prefer_outer(
                    sub,
                    contain_thr=float(args.contain_thr),
                    contain_tol_px=float(args.contain_tol_px),
                )
                sub3 = nms_keep_highest_score(sub2, iou_thr=float(args.dup_iou))
                parts.append(sub3)
            df = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
        else:
            df = drop_cut_boxes_prefer_outer(
                df,
                contain_thr=float(args.contain_thr),
                contain_tol_px=float(args.contain_tol_px),
            )
            n2 = len(df)
            df = nms_keep_highest_score(df, iou_thr=float(args.dup_iou))

        n2 = n2 if n2 != n1 else len(drop_cut_boxes_prefer_outer(df, float(args.contain_thr), float(args.contain_tol_px)))
        n3 = len(df)

    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs

    crs_out = args.crs_override.strip() if args.crs_override.strip() else crs
    if crs_out is None:
        raise ValueError("Raster has no CRS. Provide --crs_override like EPSG:xxxx.")

    geoms = [
        px_box_to_map_polygon(transform, row.x1, row.y1, row.x2, row.y2)
        for row in df.itertuples(index=False)
    ]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs=crs_out)

    # Add input_id suffix to filename if processing multiple images
    out_gpkg = Path(out_gpkg_base)
    if input_id:
        # Insert input_id before the .gpkg extension
        out_gpkg = out_gpkg.parent / f"{out_gpkg.stem}_{input_id}{out_gpkg.suffix}"

    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_gpkg, layer=args.layer, driver="GPKG")

    logger.info("=" * 80)
    logger.info(f"Export completed successfully for {input_id or 'image'}!")
    logger.info("=" * 80)
    logger.info(f"Raster: {raster_path}")
    logger.info(f"Output GeoPackage: {out_gpkg}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Min confidence: {args.min_conf} (rows: {n0} -> {n1})")
    if args.dedup:
        logger.info(
            f"Dedup: enabled | by_class: {bool(args.dedup_by_class and 'class_id' in df.columns)} | "
            f"contain_thr: {args.contain_thr} | contain_tol_px: {args.contain_tol_px} | dup_iou: {args.dup_iou}"
        )
        logger.info(f"Rows after dedup: {len(gdf)}")
    else:
        logger.info("Dedup: disabled")
        logger.info(f"Final rows: {len(gdf)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export YOLO predictions CSV to GeoPackage with robust 2-pass dedup.")
    ap.add_argument("--pred", required=True, help="predictions_tiles.csv from 06_infer.py")
    ap.add_argument("--tiles_meta", required=True, help="tiles_meta.json from 05_make_inference_tiles.py")
    ap.add_argument("--out_gpkg", required=True, help="Output GeoPackage path")
    ap.add_argument("--layer", required=True, help="Layer name inside the GeoPackage")
    ap.add_argument(
        "--input_id",
        default="",
        help="If tiles_meta has multiple inputs, set this to choose the raster and filter tiles (e.g. image1).",
    )

    ap.add_argument("--min_conf", type=float, default=0.25, help="Keep detections with conf >= this")

    ap.add_argument("--dedup", action="store_true", help="Enable deduplication")
    ap.add_argument("--dedup_by_class", action="store_true", help="Dedup per class_id (recommended)")

    # Pass A - cut boxes
    ap.add_argument("--contain_thr", type=float, default=0.90, help="Drop small box if covered by >= this fraction")
    ap.add_argument("--contain_tol_px", type=float, default=3.0, help="Tolerance for near-containment in pixels")

    # Pass B - true duplicates
    ap.add_argument("--dup_iou", type=float, default=0.90, help="If IoU >= this, keep higher score only")

    ap.add_argument("--crs_override", default="", help="Override CRS (e.g. EPSG:32737). If empty, use raster CRS.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pred_csv = resolve_path(root, args.pred)
    tiles_meta_path = resolve_path(root, args.tiles_meta)
    out_gpkg = resolve_path(root, args.out_gpkg)

    if not pred_csv.exists():
        raise FileNotFoundError(f"Pred CSV not found: {pred_csv}")
    if not tiles_meta_path.exists():
        raise FileNotFoundError(f"tiles_meta.json not found: {tiles_meta_path}")

    tiles_meta = json.loads(tiles_meta_path.read_text(encoding="utf-8"))

    df = pd.read_csv(pred_csv)
    required = {"x1", "y1", "x2", "y2", "conf"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    raster_path = None
    if "raster" in tiles_meta:
        raster_path = Path(tiles_meta["raster"])
        if not raster_path.is_absolute():
            raster_path = (root / raster_path).resolve()
        # Process single raster
        logger.info(f"Predictions CSV: {pred_csv}")
        logger.info(f"Tiles metadata: {tiles_meta_path}")
        process_single_image(df, raster_path, out_gpkg, args, logger, None)
    else:
        inputs = tiles_meta.get("inputs", [])
        if not inputs:
            raise ValueError("tiles_meta.json must include 'raster' or 'inputs'.")

        input_id = args.input_id.strip()
        if not input_id:
            prefixes = set()
            for name in df["tile"].dropna().astype(str).tolist():
                if name.startswith("tile_") and "_r" in name:
                    prefixes.add(name.split("_r")[0].replace("tile_", ""))
            if len(prefixes) == 1:
                input_id = prefixes.pop()
            elif len(prefixes) > 1:
                # Process all input prefixes
                logger.info(f"Predictions CSV: {pred_csv}")
                logger.info(f"Tiles metadata: {tiles_meta_path}")
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
                        logger.warning(f"input_id not found in tiles_meta inputs: {prefix}, skipping.")
                        continue
                    raster_path = Path(match[0]["raster"])
                    if not raster_path.is_absolute():
                        raster_path = (root / raster_path).resolve()

                    if not raster_path.exists():
                        logger.warning(f"Raster not found: {raster_path}, skipping.")
                        continue

                    process_single_image(df_subset, raster_path, out_gpkg, args, logger, prefix)
                return
            else:
                raise ValueError("Could not infer input_id from CSV. Use --input_id.")

        df = df[df["tile"].astype(str).str.startswith(f"tile_{input_id}_")].copy()
        if df.empty:
            raise ValueError(f"No predictions for input_id '{input_id}' in CSV.")

        match = [i for i in inputs if i.get("id") == input_id]
        if not match:
            raise ValueError(f"input_id not found in tiles_meta inputs: {input_id}")
        raster_path = Path(match[0]["raster"])
        if not raster_path.is_absolute():
            raster_path = (root / raster_path).resolve()

        if not raster_path.exists():
            raise FileNotFoundError(f"Raster referenced in tiles_meta.json not found: {raster_path}")

        logger.info(f"Predictions CSV: {pred_csv}")
        logger.info(f"Tiles metadata: {tiles_meta_path}")
        process_single_image(df, raster_path, out_gpkg, args, logger, input_id)


if __name__ == "__main__":
    main()
