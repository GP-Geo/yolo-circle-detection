"""Tests for 11_ingest_corrections.py — HITL correction ingestion."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Load module via importlib (numbered filename pattern, same as test_nms.py)
spec = importlib.util.spec_from_file_location(
    "ingest_corrections", SCRIPTS / "11_ingest_corrections.py"
)
ingest_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest_mod)

ingest_corrections = ingest_mod.ingest_corrections
yolo_line = ingest_mod.yolo_line
choose_split = ingest_mod.choose_split
to_uint8_per_band = ingest_mod.to_uint8_per_band
compute_tile_size_px = ingest_mod.compute_tile_size_px


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_raster(tmp_path: Path) -> Path:
    """Small synthetic 4-band GeoTIFF (256×256 px, 1-unit pixels, no CRS).

    Using no CRS avoids PROJ database lookups that can fail when the conda
    environment has an outdated proj.db (schema mismatch). The ingestion logic
    only calls gdf.to_crs() when raster CRS differs from the GeoPackage CRS;
    with both set to None the reprojection step is skipped.
    """
    raster_path = tmp_path / "test_raster.tif"
    width, height, bands = 256, 256, 4
    # Simple affine: 1 unit/pixel, origin at (0, 256) with y going south.
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 256.0)

    rng = np.random.default_rng(42)
    data = rng.integers(100, 3000, size=(bands, height, width), dtype=np.uint16)

    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint16",
        transform=transform,
        # crs intentionally omitted → src.crs will be None
    ) as dst:
        dst.write(data)

    return raster_path


@pytest.fixture
def corrections_gpkg(tmp_path: Path) -> Path:
    """GeoPackage with 3 bounding-box polygons in the same coordinate space as
    synthetic_raster (unit coordinates, no CRS).

    Raster covers x:[0,256], y:[0,256] (y=256 at top, y=0 at bottom in map
    coords because the transform has -1 y resolution).
    """
    gpkg_path = tmp_path / "corrections.gpkg"
    # Three small circles placed inside the raster extent
    geoms = [
        box(10, 210, 40, 240),   # circle 1 — upper-left region
        box(60, 160, 90, 190),   # circle 2 — mid region
        box(120, 100, 150, 130), # circle 3 — central region
    ]
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=None)
    gdf.to_file(gpkg_path, driver="GPKG", layer="detections")
    return gpkg_path


@pytest.fixture
def yolo_dataset_dir(tmp_path: Path) -> Path:
    """Minimal YOLO dataset directory with required subdirectories."""
    ds_dir = tmp_path / "yolo_dataset_test"
    for split in ("train", "val", "test"):
        (ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (ds_dir / "images_rgb" / split).mkdir(parents=True, exist_ok=True)
    return ds_dir


@pytest.fixture
def default_tiling_cfg() -> dict:
    return {"tile_size_px": 128, "stride_frac": 0.5, "use_ground_size": False}


@pytest.fixture
def default_bands_cfg() -> dict:
    return {"ms": [1, 2, 3, 4], "rgb": [4, 3, 2]}


@pytest.fixture
def default_labels_cfg() -> dict:
    return {"class_id": 0, "class_name": "circle", "min_area_fraction": 0.1, "min_bbox_px": 2}


@pytest.fixture
def default_split_cfg() -> dict:
    return {"ratios": {"train": 0.7, "val": 0.3, "test": 0.0}, "seed": 42, "block_tiles": 4}


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_yolo_line_format(self):
        """YOLO line should be 'class_id cx cy bw bh' normalized to [0,1]."""
        line = yolo_line(0, 0, 128, 128, 256, 256, 0)
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class_id
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        assert abs(cx - 0.25) < 1e-5
        assert abs(cy - 0.25) < 1e-5
        assert abs(bw - 0.5) < 1e-5
        assert abs(bh - 0.5) < 1e-5

    def test_yolo_line_normalized_values(self):
        """All values must be in [0, 1]."""
        line = yolo_line(10, 20, 100, 80, 256, 256, 0)
        parts = line.split()
        for val in parts[1:]:
            assert 0.0 <= float(val) <= 1.0

    def test_choose_split_deterministic(self):
        """Same inputs always produce the same split."""
        r1 = choose_split(3, 7, 4, 42, {"train": 0.7, "val": 0.3, "test": 0.0})
        r2 = choose_split(3, 7, 4, 42, {"train": 0.7, "val": 0.3, "test": 0.0})
        assert r1 == r2
        assert r1 in ("train", "val", "test")

    def test_to_uint8_per_band_range(self):
        """Output must be in [0, 255] uint8."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 5000, size=(4, 64, 64)).astype(np.float32)
        result = to_uint8_per_band(img)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_compute_tile_size_px_from_pixels(self):
        """Should return configured pixel size when use_ground_size is False."""
        cfg = {"tile_size_px": 256, "use_ground_size": False}
        size = compute_tile_size_px(cfg, pixel_size_m=10.0)
        assert size == 256

    def test_compute_tile_size_px_must_be_divisible_32(self):
        """Non-multiple-of-32 tile size must raise."""
        cfg = {"tile_size_px": 100, "use_ground_size": False}
        with pytest.raises(ValueError, match="divisible by 32"):
            compute_tile_size_px(cfg, pixel_size_m=10.0)


# ---------------------------------------------------------------------------
# Tests: ingest_corrections integration
# ---------------------------------------------------------------------------


class TestIngestCorrections:
    def test_produces_tiles_and_labels(
        self,
        synthetic_raster,
        corrections_gpkg,
        yolo_dataset_dir,
        default_tiling_cfg,
        default_bands_cfg,
        default_labels_cfg,
        default_split_cfg,
    ):
        """Ingestion should write at least one .tif and matching .txt label."""
        summary = ingest_corrections(
            corrections_gpkg=corrections_gpkg,
            raster_path=synthetic_raster,
            dataset_dir=yolo_dataset_dir,
            tiling_cfg=default_tiling_cfg,
            bands_cfg=default_bands_cfg,
            labels_cfg=default_labels_cfg,
            split_cfg=default_split_cfg,
            round_id="round1",
            input_id="testinput",
        )

        total_new = summary["tiles_new"]
        total_replaced = summary["tiles_replaced"]
        assert total_new + total_replaced > 0, "Expected at least one tile to be written"

        # Collect all written tiles across splits
        all_tifs = []
        all_txts = []
        for split in ("train", "val", "test"):
            all_tifs.extend((yolo_dataset_dir / "images" / split).glob("*.tif"))
            all_txts.extend((yolo_dataset_dir / "labels" / split).glob("*.txt"))

        assert len(all_tifs) > 0, "Expected at least one image tile"
        assert len(all_txts) > 0, "Expected at least one label file"
        assert len(all_tifs) == len(all_txts), "Each tile must have a matching label"

    def test_label_file_yolo_format(
        self,
        synthetic_raster,
        corrections_gpkg,
        yolo_dataset_dir,
        default_tiling_cfg,
        default_bands_cfg,
        default_labels_cfg,
        default_split_cfg,
    ):
        """Label files must contain valid YOLO lines (5 floats, values in [0,1])."""
        ingest_corrections(
            corrections_gpkg=corrections_gpkg,
            raster_path=synthetic_raster,
            dataset_dir=yolo_dataset_dir,
            tiling_cfg=default_tiling_cfg,
            bands_cfg=default_bands_cfg,
            labels_cfg=default_labels_cfg,
            split_cfg=default_split_cfg,
            round_id="round1",
            input_id="testinput",
        )

        for split in ("train", "val", "test"):
            for txt_path in (yolo_dataset_dir / "labels" / split).glob("*.txt"):
                for line in txt_path.read_text().strip().splitlines():
                    parts = line.split()
                    assert len(parts) == 5, f"Expected 5 values per label line, got {len(parts)}"
                    class_id = int(parts[0])
                    assert class_id == 0
                    cx, cy, bw, bh = (float(p) for p in parts[1:])
                    assert 0.0 <= cx <= 1.0
                    assert 0.0 <= cy <= 1.0
                    assert 0.0 < bw <= 1.0
                    assert 0.0 < bh <= 1.0

    def test_deduplication_replaces_existing_tile(
        self,
        synthetic_raster,
        corrections_gpkg,
        yolo_dataset_dir,
        default_tiling_cfg,
        default_bands_cfg,
        default_labels_cfg,
        default_split_cfg,
    ):
        """Running ingestion twice: second run should replace tiles, not add new ones."""
        summary1 = ingest_corrections(
            corrections_gpkg=corrections_gpkg,
            raster_path=synthetic_raster,
            dataset_dir=yolo_dataset_dir,
            tiling_cfg=default_tiling_cfg,
            bands_cfg=default_bands_cfg,
            labels_cfg=default_labels_cfg,
            split_cfg=default_split_cfg,
            round_id="round1",
            input_id="testinput",
        )
        new_first = summary1["tiles_new"]
        assert new_first > 0

        summary2 = ingest_corrections(
            corrections_gpkg=corrections_gpkg,
            raster_path=synthetic_raster,
            dataset_dir=yolo_dataset_dir,
            tiling_cfg=default_tiling_cfg,
            bands_cfg=default_bands_cfg,
            labels_cfg=default_labels_cfg,
            split_cfg=default_split_cfg,
            round_id="round2",
            input_id="testinput",  # same input_id → same filenames
        )
        # Second run: tiles that existed get replaced, not counted as new
        assert summary2["tiles_replaced"] == new_first
        assert summary2["tiles_new"] == 0

    def test_empty_gpkg_returns_zero_tiles(
        self,
        tmp_path,
        synthetic_raster,
        yolo_dataset_dir,
        default_tiling_cfg,
        default_bands_cfg,
        default_labels_cfg,
        default_split_cfg,
    ):
        """An empty GeoPackage should produce no tiles."""
        empty_gpkg = tmp_path / "empty.gpkg"
        gdf = gpd.GeoDataFrame({"geometry": []}, crs=None)
        gdf.to_file(empty_gpkg, driver="GPKG", layer="detections")

        summary = ingest_corrections(
            corrections_gpkg=empty_gpkg,
            raster_path=synthetic_raster,
            dataset_dir=yolo_dataset_dir,
            tiling_cfg=default_tiling_cfg,
            bands_cfg=default_bands_cfg,
            labels_cfg=default_labels_cfg,
            split_cfg=default_split_cfg,
            round_id="round1",
            input_id="empty",
        )
        assert summary["tiles_new"] == 0
        assert summary["tiles_replaced"] == 0

    def test_summary_keys_present(
        self,
        synthetic_raster,
        corrections_gpkg,
        yolo_dataset_dir,
        default_tiling_cfg,
        default_bands_cfg,
        default_labels_cfg,
        default_split_cfg,
    ):
        """Summary dict must contain the expected keys."""
        summary = ingest_corrections(
            corrections_gpkg=corrections_gpkg,
            raster_path=synthetic_raster,
            dataset_dir=yolo_dataset_dir,
            tiling_cfg=default_tiling_cfg,
            bands_cfg=default_bands_cfg,
            labels_cfg=default_labels_cfg,
            split_cfg=default_split_cfg,
            round_id="round1",
            input_id="testinput",
        )
        assert "tiles_new" in summary
        assert "tiles_replaced" in summary
        assert "tiles_skipped_no_labels" in summary
