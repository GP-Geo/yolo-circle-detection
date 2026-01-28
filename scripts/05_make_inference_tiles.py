# scripts/05_make_inference_tiles.py
from __future__ import annotations

from pathlib import Path
import json
import math
import argparse

import numpy as np
import rasterio
from rasterio.windows import Window
import cv2


def to_uint8(img: np.ndarray, nodata_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Percentile-stretch each band to uint8.
    img: (bands, H, W) float/uint
    nodata_mask: (H, W) True where nodata
    """
    img = img.astype(np.float32)

    if nodata_mask is None:
        nodata_mask = np.zeros(img.shape[1:], dtype=bool)

    out = np.zeros_like(img, dtype=np.uint8)

    for b in range(img.shape[0]):
        band = img[b]
        valid = ~nodata_mask & np.isfinite(band)

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


def write_ms_geotiff(path: Path, chw_uint8: np.ndarray, transform, crs) -> None:
    """
    Write uint8 CHW as multi-band GeoTIFF (count=C).
    This is the input YOLO will use for 8-band inference.
    """
    if chw_uint8.dtype != np.uint8 or chw_uint8.ndim != 3:
        raise ValueError("Expected uint8 CHW array for multispectral export.")
    c, h, w = chw_uint8.shape

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": c,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": min(256, w),
        "blockysize": min(256, h),
    }

    with rasterio.open(path, "w", **profile) as dst:
        for i in range(c):
            dst.write(chw_uint8[i], i + 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tile the full raster for inference: writes MS TIFFs for the model + RGB PNGs for visualization."
    )
    parser.add_argument(
        "--meta",
        default="02_processed/yolo_dataset_ms8_v1/dataset_meta.json",
        help="Dataset metadata json (MS8 dataset recommended).",
    )
    parser.add_argument(
        "--out",
        default="04_inference/tiles_full",
        help="Output root folder. Will create images_ms8/ and images_rgb/ inside it.",
    )
    parser.add_argument(
        "--ms_bands",
        default="",
        help="Override multispectral bands like '1,2,3,4,5,6,7,8' (1-indexed). If empty uses meta.export_bands_1_indexed.",
    )
    parser.add_argument(
        "--rgb_bands",
        default="",
        help="Override RGB bands like '5,3,2' (1-indexed). If empty uses meta.rgb_bands_1_indexed or falls back to 1,2,3.",
    )
    parser.add_argument(
        "--skip_empty",
        action="store_true",
        help="If set, do not write tiles that are 100%% nodata (saves disk).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    meta_path = (root / args.meta).resolve()
    out_root = (root / args.out).resolve()

    out_ms8 = out_root / "images_ms8"
    out_rgb = out_root / "images_rgb"
    out_ms8.mkdir(parents=True, exist_ok=True)
    out_rgb.mkdir(parents=True, exist_ok=True)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raster_path = Path(meta["raster"])
    if not raster_path.is_absolute():
        raster_path = (root / raster_path).resolve()

    tile_size = int(meta["tile_size"])
    stride = int(meta["stride"])

    if args.ms_bands.strip():
        ms_bands = [int(x) for x in args.ms_bands.split(",")]
    else:
        ms_bands = [int(x) for x in meta.get("export_bands_1_indexed", [1, 2, 3, 4, 5, 6, 7, 8])]

    if args.rgb_bands.strip():
        rgb_bands = [int(x) for x in args.rgb_bands.split(",")]
    else:
        rgb_bands = [int(x) for x in meta.get("rgb_bands_1_indexed", [1, 2, 3])]

    with rasterio.open(raster_path) as src:
        W, H = src.width, src.height
        nodata = src.nodata

        n_cols = math.ceil((W - tile_size) / stride) + 1
        n_rows = math.ceil((H - tile_size) / stride) + 1

        written = 0
        skipped_all_nodata = 0

        for r in range(n_rows):
            top = r * stride
            for c in range(n_cols):
                left = c * stride
                win = Window(left, top, tile_size, tile_size)

                img_ms = src.read(
                    ms_bands,
                    window=win,
                    boundless=True,
                    fill_value=nodata if nodata is not None else 0,
                )
                img_rgb = src.read(
                    rgb_bands,
                    window=win,
                    boundless=True,
                    fill_value=nodata if nodata is not None else 0,
                )

                # Nodata mask based on MS bands (more robust than RGB-only)
                if nodata is not None:
                    nodata_mask = np.any(img_ms == nodata, axis=0)
                else:
                    # If nodata is not defined, treat non-finite as nodata
                    nodata_mask = ~np.all(np.isfinite(img_ms), axis=0)

                if args.skip_empty and bool(np.all(nodata_mask)):
                    skipped_all_nodata += 1
                    continue

                ms8 = to_uint8(img_ms, nodata_mask=nodata_mask)    # uint8 CHW
                rgb8 = to_uint8(img_rgb, nodata_mask=nodata_mask)  # uint8 CHW

                # Per-tile transform (kept in GeoTIFF for GIS alignment later)
                tile_transform = rasterio.windows.transform(win, src.transform)

                name = f"tile_r{r:04d}_c{c:04d}"
                ms_path = out_ms8 / f"{name}.tif"
                rgb_path = out_rgb / f"{name}.png"

                write_ms_geotiff(ms_path, ms8, transform=tile_transform, crs=src.crs)

                rgb_hwc = np.transpose(rgb8, (1, 2, 0))
                cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR))

                written += 1

    tiles_meta = {
        "raster": str(raster_path),
        "tile_size": tile_size,
        "stride": stride,
        "ms_bands_1_indexed": ms_bands,
        "rgb_bands_1_indexed": rgb_bands,
        "out_images_ms8_dir": str(out_ms8),
        "out_images_rgb_dir": str(out_rgb),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "tiles_written": written,
        "tiles_skipped_all_nodata": skipped_all_nodata,
        "skip_empty": bool(args.skip_empty),
    }
    (out_root / "tiles_meta.json").write_text(json.dumps(tiles_meta, indent=2), encoding="utf-8")

    print("DONE")
    print("raster:", raster_path)
    print("tile_size:", tile_size, "stride:", stride)
    print("ms_bands:", ms_bands)
    print("rgb_bands:", rgb_bands)
    print("grid:", n_rows, "rows x", n_cols, "cols")
    print("tiles_written:", written)
    if args.skip_empty:
        print("tiles_skipped_all_nodata:", skipped_all_nodata)
    print("out_ms8:", out_ms8)
    print("out_rgb:", out_rgb)
    print("tiles_meta:", out_root / "tiles_meta.json")


if __name__ == "__main__":
    main()