from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import setup_logger

logger = setup_logger(__name__)


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def main() -> None:
    ap = argparse.ArgumentParser(description="Render YOLO predictions (CSV) on RGB tile PNGs.")
    ap.add_argument("--pred", required=True, help="predictions_tiles.csv from 06_infer.py")
    ap.add_argument("--rgb_dir", required=True, help="Folder with RGB tiles (tile_r####_c####.png)")
    ap.add_argument("--out", required=True, help="Output folder for overlay PNGs")
    ap.add_argument("--stride", type=int, default=None, help="Stride used for tiling. If not set, inferred from CSV columns if possible.")
    ap.add_argument("--tile_size", type=int, default=512, help="Tile size (used for clamping)")
    ap.add_argument("--min_conf", type=float, default=0.25, help="Only draw boxes with conf >= this")
    ap.add_argument("--thickness", type=int, default=2, help="Rectangle thickness")
    ap.add_argument("--write_empty", action="store_true", help="Also write overlays for tiles with 0 detections")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pred_csv = (root / args.pred).resolve()
    rgb_dir = (root / args.rgb_dir).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_csv.exists():
        raise FileNotFoundError(f"Pred CSV not found: {pred_csv}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB dir not found: {rgb_dir}")

    df = pd.read_csv(pred_csv)

    needed = {"tile", "tile_r", "tile_c", "x1", "y1", "x2", "y2", "conf"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing columns {sorted(missing)}. "
            f"Expected output from 06_infer.py predictions_tiles.csv."
        )

    # stride is required to convert global coords -> tile-local coords
    stride = args.stride
    if stride is None:
        # Try to infer stride from CSV if it contains left/top, otherwise fail fast.
        # Most robust is: pass --stride explicitly or keep tiles_meta.json around and copy value.
        raise ValueError("You must provide --stride (same stride used in tiling).")

    df = df[df["conf"] >= args.min_conf].copy()

    # Build list of tiles to process
    if args.write_empty:
        rgb_tiles = sorted(rgb_dir.glob("tile_r*_c*.png"))
        tiles = [p.name for p in rgb_tiles]
    else:
        tiles = sorted(df["tile"].unique().tolist())

    rendered = 0
    total_boxes = 0

    for tile_name in tiles:
        rgb_path = rgb_dir / tile_name.replace(".tif", ".png").replace(".tiff", ".png")
        if not rgb_path.exists():
            # If your RGB name is strictly .png already, this is fine.
            rgb_path = rgb_dir / (Path(tile_name).stem + ".png")
        if not rgb_path.exists():
            continue

        img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]

        sub = df[df["tile"] == tile_name]
        if len(sub) == 0 and not args.write_empty:
            continue

        # Draw detections
        for _, row in sub.iterrows():
            r = int(row["tile_r"])
            c = int(row["tile_c"])

            left = c * stride
            top = r * stride

            # global -> local coords
            lx1 = float(row["x1"]) - left
            ly1 = float(row["y1"]) - top
            lx2 = float(row["x2"]) - left
            ly2 = float(row["y2"]) - top

            x1, y1, x2, y2 = clamp_box(lx1, ly1, lx2, ly2, w, h)

            # Draw rectangle + confidence
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), args.thickness)
            label = f"{row['conf']:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            total_boxes += 1

        out_path = out_dir / rgb_path.name
        cv2.imwrite(str(out_path), img)
        rendered += 1

    logger.info("Rendering completed successfully")
    logger.info(f"Predictions CSV: {pred_csv}")
    logger.info(f"RGB directory: {rgb_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Stride: {stride}px")
    logger.info(f"Minimum confidence: {args.min_conf}")
    logger.info(f"Tiles rendered: {rendered}")
    logger.info(f"Boxes drawn: {total_boxes}")


if __name__ == "__main__":
    main()