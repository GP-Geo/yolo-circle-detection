# pipeline/13_split_for_review.py
"""Split merged detections into spatial review tiles for QGIS-based correction.

Creates a non-overlapping grid of review tiles. Each tile gets its own folder:
    review_tiles/
        tile_r000_c000/
            chip.tif              ← raster crop (original uint16 data)
            detections.gpkg       ← detections whose centroid falls in this tile
            detections_corrected.gpkg  ← you create this in QGIS after editing

Detections are assigned to tiles by centroid, so no detection appears in two tiles.
Tiles with zero detections are skipped by default (use --include_empty to keep them).

Workflow:
    1. Run this script after 07_merge_nms.py (done automatically by 12_run_iteration.py)
    2. Browse review_tiles/ — each folder is one spatial tile
    3. Open chip.tif + detections.gpkg in QGIS for tiles you want to correct
    4. Edit, then save as detections_corrected.gpkg in the same folder
    5. Run 11_ingest_corrections.py --corrections_dir <review_tiles_dir>

Usage:
    python pipeline/13_split_for_review.py \\
        --merged_gpkg outputs/iterations/round1_aoi/merged/detections_merged.gpkg \\
        --raster data/raw/Sent_10m_smaller_AOI.tif \\
        --out_dir outputs/iterations/round1_aoi/review_tiles \\
        --tile_size_px 512
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio.windows import Window
from shapely.geometry import box as sbox

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Core logic (exposed for testing)
# ---------------------------------------------------------------------------


def split_for_review(
    merged_gpkg: Path,
    raster_path: Path,
    out_dir: Path,
    tile_size_px: int,
    skip_empty: bool = True,
    layer: str | None = None,
) -> dict:
    """
    Split merged detections + raster into a grid of review tiles.

    Detections are assigned to tiles by centroid (no duplicates across boundaries).
    Returns a manifest dict describing all created tiles.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading merged detections: {merged_gpkg}")
    gdf = gpd.read_file(merged_gpkg, layer=layer)
    logger.info(f"  {len(gdf)} detections loaded")

    tiles_info = []

    with rasterio.open(raster_path) as src:
        W, H = src.width, src.height
        crs = src.crs
        transform = src.transform

        if gdf.crs and crs and gdf.crs != crs:
            gdf = gdf.to_crs(crs)

        # Centroid columns for tile assignment (avoids cross-boundary duplicates)
        gdf = gdf.copy()
        centroids = gdf.geometry.centroid
        gdf["_cx"] = centroids.x
        gdf["_cy"] = centroids.y

        n_cols = math.ceil(W / tile_size_px)
        n_rows = math.ceil(H / tile_size_px)
        logger.info(f"Review grid: {n_rows}×{n_cols} = {n_rows * n_cols} tiles at {tile_size_px}px")

        try:
            from tqdm import tqdm
            row_iter = tqdm(range(n_rows), desc="Splitting review tiles", unit="row")
        except ImportError:
            row_iter = range(n_rows)

        for r in row_iter:
            for c in range(n_cols):
                left = c * tile_size_px
                top = r * tile_size_px
                w = min(tile_size_px, W - left)
                h = min(tile_size_px, H - top)

                win = Window(left, top, w, h)
                win_transform = rasterio.windows.transform(win, transform)
                xmin, ymin, xmax, ymax = rasterio.windows.bounds(win, transform)

                # Filter by centroid strictly inside this tile
                mask = (
                    (gdf["_cx"] >= xmin) & (gdf["_cx"] < xmax) &
                    (gdf["_cy"] >= ymin) & (gdf["_cy"] < ymax)
                )
                tile_gdf = gdf[mask].drop(columns=["_cx", "_cy"])
                n_det = len(tile_gdf)

                if skip_empty and n_det == 0:
                    continue

                tile_id = f"tile_r{r:03d}_c{c:03d}"
                tile_dir = out_dir / tile_id
                tile_dir.mkdir(exist_ok=True)

                # Write raster chip (original data, preserves dtype)
                chip_path = tile_dir / "chip.tif"
                data = src.read(window=win)
                profile = src.profile.copy()
                profile.update({"height": h, "width": w, "transform": win_transform})
                with rasterio.open(chip_path, "w", **profile) as dst:
                    dst.write(data)

                # Write detections GPKG
                det_path = tile_dir / "detections.gpkg"
                if n_det > 0:
                    tile_gdf.to_file(det_path, driver="GPKG", layer="detections")
                else:
                    gpd.GeoDataFrame({"geometry": []}, crs=crs).to_file(
                        det_path, driver="GPKG", layer="detections"
                    )

                tiles_info.append({
                    "id": tile_id,
                    "row": r,
                    "col": c,
                    "bounds_map": [xmin, ymin, xmax, ymax],
                    "pixel_window": [left, top, w, h],
                    "n_detections": n_det,
                    "chip": str(chip_path),
                    "detections_gpkg": str(det_path),
                    "corrected_gpkg": str(tile_dir / "detections_corrected.gpkg"),
                    "status": "pending",
                })

    manifest = {
        "raster": str(raster_path),
        "merged_gpkg": str(merged_gpkg),
        "tile_size_px": tile_size_px,
        "total_tiles": len(tiles_info),
        "total_detections": int(len(gdf)),
        "tiles": tiles_info,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(f"Created {len(tiles_info)} review tiles → {out_dir}")
    logger.info(f"Manifest written: {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split merged detections into spatial review tiles. "
            "Each tile gets a raster chip + detections GeoPackage for QGIS review."
        )
    )
    parser.add_argument(
        "--merged_gpkg", required=True,
        help="Merged detections GeoPackage from 07_merge_nms.py",
    )
    parser.add_argument(
        "--raster", required=True,
        help="Source raster GeoTIFF",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory for review tiles",
    )
    parser.add_argument(
        "--tile_size_px", type=int, default=512,
        help="Review tile size in pixels, non-overlapping (default: 512)",
    )
    parser.add_argument(
        "--layer", default="",
        help="Layer name in the merged GeoPackage (default: first layer)",
    )
    parser.add_argument(
        "--include_empty", action="store_true",
        help="Also create tiles that have zero detections",
    )
    args = parser.parse_args()

    from utils import resolve_path
    from paths import PROJECT_ROOT

    root = PROJECT_ROOT
    merged_gpkg = resolve_path(root, args.merged_gpkg)
    raster_path = resolve_path(root, args.raster)
    out_dir = resolve_path(root, args.out_dir)

    if not merged_gpkg.exists():
        raise FileNotFoundError(f"Merged GPKG not found: {merged_gpkg}")
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    logger.info("=" * 80)
    logger.info("YOLO Circle Detection - Split for Review")
    logger.info("=" * 80)
    logger.info(f"Merged GPKG:  {merged_gpkg}")
    logger.info(f"Raster:       {raster_path}")
    logger.info(f"Output dir:   {out_dir}")
    logger.info(f"Tile size:    {args.tile_size_px}px")

    manifest = split_for_review(
        merged_gpkg=merged_gpkg,
        raster_path=raster_path,
        out_dir=out_dir,
        tile_size_px=args.tile_size_px,
        skip_empty=not args.include_empty,
        layer=args.layer or None,
    )

    n = manifest["total_tiles"]
    logger.info("=" * 80)
    logger.info(f"Done — {n} review tiles created")
    logger.info("Next steps:")
    logger.info("  1. Browse the tile folders and open chip.tif + detections.gpkg in QGIS")
    logger.info("  2. Edit detections, save as detections_corrected.gpkg in the same folder")
    logger.info(f"  3. Ingest all corrections at once:")
    logger.info(f"       python pipeline/11_ingest_corrections.py \\")
    logger.info(f"           --corrections_dir {out_dir} \\")
    logger.info(f"           --raster {raster_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
