# pipeline/05_make_inference_tiles.py
from __future__ import annotations

from pathlib import Path
import sys
import json
import math
import argparse

import numpy as np
import rasterio
from rasterio.windows import Window
import cv2

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import setup_logger

logger = setup_logger(__name__)


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


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("Missing dependency 'pyyaml'. Install with: pip install pyyaml") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tile rasters for inference: writes MS TIFFs for the model + RGB PNGs for visualization."
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Pipeline config yaml.",
    )
    parser.add_argument(
        "--input_id",
        default="",
        help="Only tile this input id (e.g. image1). If empty, tile all inputs.",
    )
    parser.add_argument(
        "--out",
        default="outputs/inference/tiles_full",
        help="Output root folder. Will create images_ms/ and images_rgb/ inside it.",
    )
    parser.add_argument(
        "--ms_bands",
        default="",
        help="Override multispectral bands like '1,2,3,4' (1-indexed). If empty uses config bands.ms.",
    )
    parser.add_argument(
        "--rgb_bands",
        default="",
        help="Override RGB bands like '4,3,2' (1-indexed). If empty uses config bands.rgb.",
    )
    parser.add_argument(
        "--skip_empty",
        action="store_true",
        help="If set, do not write tiles that are 100%% nodata (saves disk).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.append(str(root))

    from paths import PROJECT_ROOT

    root = PROJECT_ROOT
    cfg_path = resolve_path(root, args.config)
    cfg = load_config(cfg_path)

    inputs = cfg.get("inputs", [])
    if not inputs:
        raise ValueError("Config must include at least one input under 'inputs'.")

    if args.input_id:
        inputs = [i for i in inputs if i.get("id") == args.input_id]
        if not inputs:
            raise ValueError(f"input_id not found in config: {args.input_id}")

    bands_cfg = cfg.get("bands", {})
    if args.ms_bands.strip():
        ms_bands = [int(x) for x in args.ms_bands.split(",")]
    else:
        ms_bands = bands_cfg.get("ms", [])
    if args.rgb_bands.strip():
        rgb_bands = [int(x) for x in args.rgb_bands.split(",")]
    else:
        rgb_bands = bands_cfg.get("rgb", [])
    if not ms_bands or not rgb_bands:
        raise ValueError("bands.ms and bands.rgb must be set in config or overridden.")

    tiling_cfg = cfg.get("tiling", {})
    stride_frac = float(tiling_cfg.get("stride_frac", 0.75))
    if not (0 < stride_frac <= 1.0):
        raise ValueError("tiling.stride_frac must be in (0, 1].")

    out_root = resolve_path(root, args.out)
    out_ms = out_root / "images_ms"
    out_rgb = out_root / "images_rgb"
    out_ms.mkdir(parents=True, exist_ok=True)
    out_rgb.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm as _tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    tile_size_px = None
    stride_px = None
    inputs_meta = []

    for inp in inputs:
        input_id = inp.get("id")
        raster_path = resolve_path(root, inp["raster"])
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster not found: {raster_path}")

        with rasterio.open(raster_path) as src:
            W, H = src.width, src.height
            nodata = src.nodata

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
                        "Run per-input if you need different sizes."
                    )

            n_cols = math.ceil((W - tile_size_px) / stride_px) + 1
            n_rows = math.ceil((H - tile_size_px) / stride_px) + 1

            written = 0
            skipped_all_nodata = 0
            total_tiles = n_rows * n_cols
            logger.info(f"  {input_id}: {n_rows}×{n_cols} = {total_tiles} tiles")

            row_iter = (
                _tqdm(range(n_rows), desc=f"Tiling {input_id}", unit="row", leave=False)
                if _tqdm_available
                else range(n_rows)
            )

            for r in row_iter:
                top = r * stride_px
                for c in range(n_cols):
                    left = c * stride_px
                    win = Window(left, top, tile_size_px, tile_size_px)

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

                    if nodata is not None:
                        nodata_mask = np.any(img_ms == nodata, axis=0)
                    else:
                        nodata_mask = ~np.all(np.isfinite(img_ms), axis=0)

                    if args.skip_empty and bool(np.all(nodata_mask)):
                        skipped_all_nodata += 1
                        continue

                    ms8 = to_uint8(img_ms, nodata_mask=nodata_mask)    # uint8 CHW
                    rgb8 = to_uint8(img_rgb, nodata_mask=nodata_mask)  # uint8 CHW

                    tile_transform = rasterio.windows.transform(win, src.transform)

                    name = f"tile_{input_id}_r{r:04d}_c{c:04d}"
                    ms_path = out_ms / f"{name}.tif"
                    rgb_path = out_rgb / f"{name}.png"

                    write_ms_geotiff(ms_path, ms8, transform=tile_transform, crs=src.crs)

                    rgb_hwc = np.transpose(rgb8, (1, 2, 0))
                    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR))

                    written += 1

            inputs_meta.append(
                {
                    "id": input_id,
                    "raster": str(raster_path),
                    "crs": str(src.crs),
                    "raster_size_px": [W, H],
                    "pixel_size": [px_size_x, px_size_y],
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "tiles_written": written,
                    "tiles_skipped_all_nodata": skipped_all_nodata,
                }
            )

    tiles_meta = {
        "inputs": inputs_meta,
        "tile_size": tile_size_px,
        "stride": stride_px,
        "ms_bands_1_indexed": ms_bands,
        "rgb_bands_1_indexed": rgb_bands,
        "out_images_ms_dir": str(out_ms),
        "out_images_rgb_dir": str(out_rgb),
        "skip_empty": bool(args.skip_empty),
    }
    if len(inputs_meta) == 1:
        tiles_meta["input_id"] = inputs_meta[0]["id"]
        tiles_meta["raster"] = inputs_meta[0]["raster"]
    (out_root / "tiles_meta.json").write_text(json.dumps(tiles_meta, indent=2), encoding="utf-8")

    logger.info("Tiling completed successfully")
    logger.info(f"Inputs: {[i['id'] for i in inputs_meta]}")
    logger.info(f"Tile size: {tile_size_px}px, Stride: {stride_px}px")
    logger.info(f"Multispectral bands: {ms_bands}")
    logger.info(f"RGB bands: {rgb_bands}")
    logger.info(f"Output MS directory: {out_ms}")
    logger.info(f"Output RGB directory: {out_rgb}")
    logger.info(f"Tiles metadata: {out_root / 'tiles_meta.json'}")


if __name__ == "__main__":
    main()
