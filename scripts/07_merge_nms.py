from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box as sbox
from rasterio.transform import xy as txy


def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    Standard NMS in xyxy pixel coords.
    boxes: (N,4) xyxy global pixel coords
    scores: (N,)
    returns indices to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge tile predictions with NMS + nested suppression (prefer outer), export GeoPackage."
    )
    parser.add_argument("--pred_csv", required=True, help="predictions_tiles.csv from 06_infer.py")
    parser.add_argument("--meta", default="02_processed/yolo_dataset_v1/dataset_meta.json", help="Dataset metadata json")
    parser.add_argument("--out", required=True, help="Output folder, e.g. 04_inference/merged_predictions/yolo11n_wv3_v1_full")

    parser.add_argument("--min_conf", type=float, default=0.0, help="Drop predictions below this confidence before merging")
    parser.add_argument("--nms_iou", type=float, default=0.30, help="NMS IoU threshold (global pixel space)")
    parser.add_argument("--coverage_thresh", type=float, default=0.85, help="Mostly-contained coverage threshold")
    parser.add_argument("--contain_tol_px", type=float, default=3.0, help="Containment tolerance in pixels")
    parser.add_argument("--score_margin", type=float, default=0.0, help="Keep inner if it is higher than outer by this margin")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    pred_csv = resolve_path(root, args.pred_csv)
    meta_path = resolve_path(root, args.meta)
    out_dir = resolve_path(root, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_csv.exists():
        raise FileNotFoundError(f"pred_csv not found: {pred_csv}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta json not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raster_path = Path(meta["raster"])
    if not raster_path.is_absolute():
        raster_path = (root / raster_path).resolve()

    df = pd.read_csv(pred_csv)
    if df.empty:
        print("No predictions found. Exiting.")
        return

    original_n = len(df)

    if args.min_conf > 0:
        df = df[df["conf"] >= args.min_conf].copy()
        if df.empty:
            print("All predictions filtered out by min_conf. Exiting.")
            return

    # 1) Global NMS per class_id
    kept_after_nms = []
    for class_id, g in df.groupby("class_id"):
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
        scores = g["conf"].to_numpy(dtype=np.float32)
        keep_idx = nms_xyxy(boxes, scores, args.nms_iou)
        kept_after_nms.append(g.iloc[keep_idx])

    df_nms = pd.concat(kept_after_nms, ignore_index=True)
    after_nms_n = len(df_nms)

    # 2) Nested suppression per class_id (prefer outer)
    kept_final = []
    for class_id, g in df_nms.groupby("class_id"):
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

    out_gpkg = out_dir / "detections_merged.gpkg"
    gdf.to_file(out_gpkg, layer="detections", driver="GPKG")

    out_csv = out_dir / "detections_merged.csv"
    out_df.to_csv(out_csv, index=False)

    print("DONE")
    print("pred_csv:", pred_csv)
    print("raster:", raster_path)
    print("min_conf:", args.min_conf)
    print("nms_iou:", args.nms_iou)
    print("coverage_thresh:", args.coverage_thresh)
    print("contain_tol_px:", args.contain_tol_px)
    print("score_margin:", args.score_margin)
    print("counts: original =", original_n, " after_nms =", after_nms_n, " final =", final_n)
    print("gpkg:", out_gpkg)
    print("csv:", out_csv)


if __name__ == "__main__":
    main()