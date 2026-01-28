from __future__ import annotations

import re
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import rasterio
import cv2


TILE_RE = re.compile(
    r"tile_r(?P<r>\d+)_c(?P<c>\d+)\.(tif|tiff|png|jpg|jpeg)$",
    re.IGNORECASE,
)


def parse_tile_rc(filename: str) -> tuple[int, int]:
    m = TILE_RE.search(filename)
    if not m:
        raise ValueError(
            f"Filename does not match expected pattern 'tile_r####_c####.(tif|tiff|png|jpg|jpeg)': {filename}"
        )
    return int(m.group("r")), int(m.group("c"))


def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


def list_tile_files(source_dir: Path) -> list[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # Deterministic order
    files.sort(key=lambda p: p.name)
    return files


def read_tile_as_hwc_uint8(path: Path) -> np.ndarray:
    suf = path.suffix.lower()

    if suf in {".tif", ".tiff"}:
        # Rasterio returns CHW. Convert to HWC.
        with rasterio.open(path) as src:
            arr = src.read()  # (C, H, W)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape {arr.shape} for {path.name}")
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
        if arr.dtype != np.uint8:
            # If your tiles are already uint8, this is a no-op.
            arr = arr.astype(np.uint8, copy=False)
        return np.ascontiguousarray(arr)

    # PNG/JPG via OpenCV is fine
    im_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im_bgr is None:
        raise ValueError(f"OpenCV failed to read image: {path}")

    # Handle grayscale or BGRA
    if im_bgr.ndim == 2:
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_GRAY2RGB)
    elif im_bgr.shape[2] == 4:
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGRA2RGB)
    else:
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    if im_rgb.dtype != np.uint8:
        im_rgb = im_rgb.astype(np.uint8, copy=False)

    return np.ascontiguousarray(im_rgb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on tiles and export global-pixel predictions CSV. Supports 8-band TIFF by reading with rasterio."
    )
    parser.add_argument("--model", required=True, help="Weights path, e.g. 03_models/runs/yolo_wv3_v1/weights/best.pt")
    parser.add_argument("--source", required=True, help="Tiles folder, e.g. 04_inference/tiles_full/images_ms")
    parser.add_argument("--meta", required=True, help="Metadata json with tile_size and stride (dataset_meta.json or tiles_meta.json)")
    parser.add_argument("--out", required=True, help="Output folder")

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold used inside predict()")
    parser.add_argument("--max_det", type=int, default=300, help="Max detections per image")
    parser.add_argument("--device", default="auto", help="auto | mps | cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]  # YOLO_detections
    model_path = resolve_path(root, args.model)
    source_dir = resolve_path(root, args.source)
    meta_path = resolve_path(root, args.meta)
    out_dir = resolve_path(root, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source tiles folder not found: {source_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata json not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tile_size = int(meta["tile_size"])
    stride = int(meta["stride"])

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    model = YOLO(str(model_path))

    tile_files = list_tile_files(source_dir)
    if len(tile_files) == 0:
        raise FileNotFoundError(f"No tiles found in: {source_dir}")

    rows: list[dict] = []
    images_scored = 0
    boxes_saved = 0
    images_failed_read = 0
    images_failed_infer = 0

    for tile_path in tile_files:
        try:
            r, c = parse_tile_rc(tile_path.name)
        except Exception as e:
            # Skip anything not matching naming convention
            print(f"WARNING: skipping file (name pattern): {tile_path.name} ({e})")
            continue

        try:
            im = read_tile_as_hwc_uint8(tile_path)
        except Exception as e:
            images_failed_read += 1
            print(f"WARNING: Image Read Error {tile_path} - {e}")
            continue

        # Global pixel offsets for this tile
        left = c * stride
        top = r * stride

        try:
            # Predict on numpy image to bypass OpenCV file reading (critical for 8-band TIFF).
            pred_list = model.predict(
                source=im,
                imgsz=tile_size,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
                verbose=False,
            )
            if not pred_list:
                images_scored += 1
                continue
            res = pred_list[0]
        except Exception as e:
            images_failed_infer += 1
            print(f"WARNING: Inference Error {tile_path} - {e}")
            continue

        images_scored += 1

        if res.boxes is None or len(res.boxes) == 0:
            continue

        xyxy = res.boxes.xyxy.detach().cpu().numpy()  # tile pixel coords
        conf = res.boxes.conf.detach().cpu().numpy()
        cls = res.boxes.cls.detach().cpu().numpy().astype(int)

        for (x1, y1, x2, y2), p, k in zip(xyxy, conf, cls):
            gx1 = float(x1 + left)
            gy1 = float(y1 + top)
            gx2 = float(x2 + left)
            gy2 = float(y2 + top)

            rows.append(
                {
                    "tile": tile_path.name,
                    "tile_r": r,
                    "tile_c": c,
                    "x1": gx1,
                    "y1": gy1,
                    "x2": gx2,
                    "y2": gy2,
                    "conf": float(p),
                    "class_id": int(k),
                }
            )
            boxes_saved += 1

    df = pd.DataFrame(rows)
    out_csv = out_dir / "predictions_tiles.csv"
    df.to_csv(out_csv, index=False)

    print("DONE")
    print("model:", model_path)
    print("source:", source_dir)
    print("meta:", meta_path)
    print("device:", device)
    print("tile_size:", tile_size, "stride:", stride)
    print("tiles_found:", len(tile_files))
    print("images_scored:", images_scored)
    print("images_failed_read:", images_failed_read)
    print("images_failed_infer:", images_failed_infer)
    print("boxes_saved:", boxes_saved)
    print("out_csv:", out_csv)


if __name__ == "__main__":
    main()