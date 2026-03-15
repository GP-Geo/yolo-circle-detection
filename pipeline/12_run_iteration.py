# pipeline/12_run_iteration.py
"""Orchestrator for one HITL inference iteration cycle.

Chains scripts 05 → 06 → 07 → 09 to run inference on a raster and export
detections as a GeoPackage ready for QGIS review.

After this script completes, the user should:
  1. Open the output GeoPackage + raster in QGIS
  2. Review detections: delete false positives, add missing circles, adjust bboxes
  3. Save the corrected GeoPackage to data/corrections/
  4. Run 11_ingest_corrections.py to merge corrections into the training dataset
  5. Run 04_train.py to retrain, then run this script again

Usage:
    python pipeline/12_run_iteration.py \\
        --model  models/runs/training_runs/yolo11s_s2_v2/weights/best.pt \\
        --raster data/external/active/s2_large_tile.tif \\
        --config configs/pipeline.yaml \\
        --output-dir outputs/iterations/round1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import setup_logger

logger = setup_logger(__name__)

SCRIPTS_DIR = ROOT / "pipeline"


def _run(cmd: list[str], step: str) -> None:
    """Run a subprocess command, raising on non-zero exit."""
    logger.info(f"--- Step: {step} ---")
    logger.info("Command: " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{step}' failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(str(c) for c in cmd)}"
        )
    logger.info(f"Step '{step}' completed successfully.")


def run_iteration(
    model: Path,
    raster: Path,
    config: Path,
    output_dir: Path,
    input_id: str,
    conf: float = 0.25,
    iou: float = 0.6,
    nms_iou: float = 0.30,
    min_conf: float = 0.3,
    device: str = "auto",
    review_tile_size: int = 512,
) -> Path:
    """
    Run one full inference iteration cycle (05 → 06 → 07 → 13 → 09).

    Parameters
    ----------
    model:             Path to trained YOLO .pt weights.
    raster:            Path to the GeoTIFF raster to run inference on.
    config:            Path to pipeline.yaml.
    output_dir:        Root output directory for this iteration.
    input_id:          Input identifier (must match an entry in config inputs).
    conf:              YOLO confidence threshold.
    iou:               YOLO IoU threshold for predict().
    nms_iou:           Post-prediction NMS IoU threshold.
    min_conf:          Drop boxes below this confidence before NMS (reduces NMS input size).
    device:            Compute device: 'auto', 'mps', or 'cpu'.
    review_tile_size:  Size of review tiles in pixels (0 to skip step 13).

    Returns
    -------
    Path to the review tiles directory (or merged GPKG if review tiles skipped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles_dir = output_dir / "tiles"
    preds_dir = output_dir / "predictions"
    merged_dir = output_dir / "merged"
    gpkg_dir = output_dir / "gpkg"

    python = sys.executable

    # ------------------------------------------------------------------
    # 05 — Tile the raster
    # ------------------------------------------------------------------
    _run(
        [
            python,
            str(SCRIPTS_DIR / "05_make_inference_tiles.py"),
            "--config", str(config),
            "--input_id", input_id,
            "--out", str(tiles_dir),
        ],
        "05 make_inference_tiles",
    )

    tiles_meta = tiles_dir / "tiles_meta.json"

    # ------------------------------------------------------------------
    # 06 — Run YOLO inference
    # ------------------------------------------------------------------
    _run(
        [
            python,
            str(SCRIPTS_DIR / "06_infer.py"),
            "--model", str(model),
            "--source", str(tiles_dir / "images_ms"),
            "--meta", str(tiles_meta),
            "--out", str(preds_dir),
            "--conf", str(conf),
            "--iou", str(iou),
            "--device", device,
        ],
        "06 infer",
    )

    pred_csv = preds_dir / "predictions_tiles.csv"

    # ------------------------------------------------------------------
    # 07 — Merge predictions + NMS
    # ------------------------------------------------------------------
    # We need a dataset_meta.json that maps input_id → raster path.
    # Build a minimal one from tiles_meta if the full dataset_meta is absent.
    tiles_meta_data = json.loads(tiles_meta.read_text(encoding="utf-8"))
    # dataset_meta expected format: {"raster": "...", "tile_size": N, "stride": N}
    #   or {"inputs": [...], "tile_size": N, "stride": N}
    ds_meta = {
        "tile_size": tiles_meta_data["tile_size"],
        "stride": tiles_meta_data["stride"],
        "inputs": tiles_meta_data.get("inputs", []),
    }
    if tiles_meta_data.get("raster"):
        ds_meta["raster"] = tiles_meta_data["raster"]

    ds_meta_path = output_dir / "dataset_meta_iter.json"
    ds_meta_path.write_text(json.dumps(ds_meta, indent=2), encoding="utf-8")

    _run(
        [
            python,
            str(SCRIPTS_DIR / "07_merge_nms.py"),
            "--pred_csv", str(pred_csv),
            "--meta", str(ds_meta_path),
            "--out", str(merged_dir),
            "--nms_iou", str(nms_iou),
            "--min_conf", str(min_conf),
        ],
        "07 merge_nms",
    )

    # ------------------------------------------------------------------
    # 13 — Split merged detections into review tiles
    # ------------------------------------------------------------------
    merged_gpkg = merged_dir / "detections_merged.gpkg"
    review_dir = output_dir / "review_tiles"

    if review_tile_size > 0:
        _run(
            [
                python,
                str(SCRIPTS_DIR / "13_split_for_review.py"),
                "--merged_gpkg", str(merged_gpkg),
                "--raster", str(raster),
                "--out_dir", str(review_dir),
                "--tile_size_px", str(review_tile_size),
            ],
            "13 split_for_review",
        )

    # ------------------------------------------------------------------
    # 09 — Export full merged GeoPackage (geographic coords)
    # ------------------------------------------------------------------
    out_gpkg = gpkg_dir / f"{input_id}_detections.gpkg"
    _run(
        [
            python,
            str(SCRIPTS_DIR / "09_export_predictions_gpkg.py"),
            "--pred", str(pred_csv),
            "--tiles_meta", str(tiles_meta),
            "--out_gpkg", str(out_gpkg),
            "--layer", input_id,
        ],
        "09 export_predictions_gpkg",
    )

    return review_dir if review_tile_size > 0 else merged_gpkg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one HITL inference iteration: tile → infer → NMS → export GeoPackage. "
            "After completion, open the GeoPackage in QGIS to review and correct detections."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained YOLO .pt weights.",
    )
    parser.add_argument(
        "--raster",
        required=True,
        help="Path to the raster GeoTIFF to run inference on.",
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Pipeline config YAML (default: configs/pipeline.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="",
        help="Root output directory for this iteration. Defaults to outputs/iterations/YYYY-MM-DD.",
    )
    parser.add_argument(
        "--input-id",
        dest="input_id",
        default="large_tile",
        help="Input identifier used for tile naming (default: large_tile).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="YOLO IoU threshold (predict).")
    parser.add_argument("--nms_iou", type=float, default=0.30, help="Post-NMS IoU threshold.")
    parser.add_argument("--min_conf", type=float, default=0.3, help="Drop boxes below this confidence before NMS (default: 0.3).")
    parser.add_argument(
        "--device", default="auto", help="Compute device: auto | mps | cpu (default: auto)."
    )
    parser.add_argument(
        "--review-tile-size",
        dest="review_tile_size",
        type=int,
        default=512,
        help="Size of review tiles in pixels for step 13 (default: 512, 0 to skip).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.append(str(root))

    from utils import resolve_path

    root_path = root

    model = resolve_path(root_path, args.model)
    raster = resolve_path(root_path, args.raster)
    config = resolve_path(root_path, args.config)

    if args.output_dir:
        output_dir = resolve_path(root_path, args.output_dir)
    else:
        output_dir = root_path / "outputs" / "iterations" / str(date.today())

    if not model.exists():
        raise FileNotFoundError(f"Model weights not found: {model}")
    if not raster.exists():
        raise FileNotFoundError(f"Raster not found: {raster}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    logger.info("=" * 80)
    logger.info("YOLO Circle Detection - HITL Iteration Runner")
    logger.info("=" * 80)
    logger.info(f"Model:      {model}")
    logger.info(f"Raster:     {raster}")
    logger.info(f"Config:     {config}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Input ID:   {args.input_id}")
    logger.info(f"Confidence: {args.conf} | IoU: {args.iou} | NMS IoU: {args.nms_iou}")
    logger.info(f"Review tile size: {args.review_tile_size}px")

    output_path = run_iteration(
        model=model,
        raster=raster,
        config=config,
        output_dir=output_dir,
        input_id=args.input_id,
        conf=args.conf,
        iou=args.iou,
        nms_iou=args.nms_iou,
        min_conf=args.min_conf,
        device=args.device,
        review_tile_size=args.review_tile_size,
    )

    logger.info("=" * 80)
    logger.info("Iteration complete!")
    if args.review_tile_size > 0:
        logger.info(f"Review tiles: {output_path}")
    else:
        logger.info(f"Merged GPKG:  {output_path}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("NEXT STEPS — HITL Review:")
    if args.review_tile_size > 0:
        logger.info("  1. Browse the review tile folders:")
        logger.info(f"       {output_path}")
        logger.info("       Each folder has chip.tif + detections.gpkg")
    else:
        logger.info("  1. Open the merged GeoPackage and raster in QGIS:")
        logger.info(f"       {output_path}")
        logger.info(f"       {raster}")
    logger.info("  2. Review detections:")
    logger.info("       • Delete false positives (select feature → Delete key)")
    logger.info("       • Add missing circles (digitize new polygon bboxes)")
    logger.info("       • Adjust inaccurate bboxes")
    logger.info("  3. Save the corrected GeoPackage to data/corrections/")
    logger.info("  4. Ingest corrections into the training dataset:")
    logger.info("       python pipeline/11_ingest_corrections.py \\")
    logger.info("           --corrections_gpkg data/corrections/round1_corrected.gpkg \\")
    logger.info(f"          --raster {raster}")
    logger.info("  5. Retrain the model:")
    logger.info("       python pipeline/04_train.py")
    logger.info("  6. Run another iteration with the improved model:")
    logger.info("       python pipeline/12_run_iteration.py --model <new_best.pt> ...")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
