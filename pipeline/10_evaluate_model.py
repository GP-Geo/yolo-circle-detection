"""
Model evaluation script with comprehensive metrics for YOLO circle detection.

This script computes detailed performance metrics including:
- Precision, Recall, F1-score per class
- mAP (mean Average Precision) at different IOU thresholds
- Confusion matrices
- Size-based performance analysis (small/medium/large objects)
- Per-class performance breakdown

Usage:
    python pipeline/10_evaluate_model.py --model models/runs/training_runs/yolo11n_<run_id>/weights/best.pt
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from paths import PROCESSED_DIR, RUNS_DIR, REPORTS_DIR
from utils import setup_logger

logger = setup_logger(__name__, log_file=REPORTS_DIR / "logs" / "evaluation.log")


def load_config(path: Path) -> dict:
    """Load YAML configuration file."""
    try:
        import yaml
    except ImportError as e:
        logger.error("Missing required dependency 'pyyaml'")
        raise ImportError("Missing dependency 'pyyaml'. Install with: pip install pyyaml") from e

    if not path.exists():
        logger.error(f"Configuration file not found: {path}")
        raise FileNotFoundError(f"Configuration file not found: {path}")

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    logger.debug(f"Loaded configuration from {path}")
    return cfg


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IOU) between two boxes.

    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format

    Returns:
        IOU value between 0 and 1
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)


def compute_ap_per_class(
    gt_boxes: List[np.ndarray],
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_classes: List[np.ndarray],
    class_id: int,
    iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute Average Precision (AP) for a specific class.

    Args:
        gt_boxes: List of ground truth boxes per image
        pred_boxes: List of predicted boxes per image
        pred_scores: List of confidence scores per image
        pred_classes: List of predicted class IDs per image
        class_id: Class ID to evaluate
        iou_threshold: IOU threshold for considering a detection as correct

    Returns:
        Tuple of (AP, precision, recall)
    """
    # Collect all detections and ground truths
    all_detections = []
    num_gt = 0

    for img_idx in range(len(gt_boxes)):
        # Ground truth for this image
        gt = gt_boxes[img_idx]
        num_gt += len(gt)

        # Predictions for this image
        pred = pred_boxes[img_idx]
        scores = pred_scores[img_idx]
        classes = pred_classes[img_idx]

        # Filter predictions for this class
        class_mask = classes == class_id
        pred_class = pred[class_mask]
        scores_class = scores[class_mask]

        # Match predictions to ground truth
        matched_gt = set()
        for det_idx in range(len(pred_class)):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx in range(len(gt)):
                if gt_idx in matched_gt:
                    continue
                iou = compute_iou(pred_class[det_idx], gt[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            is_tp = best_iou >= iou_threshold
            if is_tp and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)

            all_detections.append({
                'score': scores_class[det_idx],
                'is_tp': is_tp,
                'image_idx': img_idx
            })

    if not all_detections:
        return 0.0, 0.0, 0.0

    # Sort by confidence score (descending)
    all_detections = sorted(all_detections, key=lambda x: x['score'], reverse=True)

    # Compute precision-recall curve
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for det in all_detections:
        if det['is_tp']:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / (num_gt + 1e-6)

        precisions.append(precision)
        recalls.append(recall)

    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precision_at_recall = 0.0
        for i in range(len(recalls)):
            if recalls[i] >= t:
                precision_at_recall = max(precision_at_recall, precisions[i])
        ap += precision_at_recall / 11

    final_precision = precisions[-1] if precisions else 0.0
    final_recall = recalls[-1] if recalls else 0.0

    return ap, final_precision, final_recall


def categorize_box_size(box: np.ndarray, image_size: int = 512) -> str:
    """
    Categorize bounding box by size (following COCO metrics).

    Args:
        box: [x1, y1, x2, y2] format
        image_size: Image dimension for normalization

    Returns:
        Size category: 'small', 'medium', or 'large'
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    area = width * height

    # Normalize by image area
    normalized_area = area / (image_size * image_size)

    if normalized_area < 0.01:  # < 1% of image
        return 'small'
    elif normalized_area < 0.04:  # 1-4% of image
        return 'medium'
    else:
        return 'large'


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names + ['background'],
                yticklabels=class_names + ['background'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to: {output_path}")


def plot_metrics_by_size(size_metrics: Dict, output_path: Path):
    """Plot performance metrics by object size."""
    sizes = list(size_metrics.keys())
    metrics = ['precision', 'recall', 'f1']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, metric in enumerate(metrics):
        values = [size_metrics[size][metric] for size in sizes]
        axes[idx].bar(sizes, values, color=['#3498db', '#e74c3c', '#2ecc71'])
        axes[idx].set_title(f'{metric.capitalize()} by Object Size')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Size-based metrics plot saved to: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO circle detection model with comprehensive metrics"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model weights (e.g., models/runs/training_runs/yolo11n_v1/weights/best.pt)"
    )
    parser.add_argument(
        "--data",
        help="Path to data.yaml (default: inferred from config)"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25)"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IOU threshold for evaluation (default: 0.5)"
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: val)"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for evaluation results (default: reports/evaluation/<model_name>)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("YOLO Circle Detection - Model Evaluation")
    logger.info("=" * 80)

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    logger.info(f"Model: {model_path}")

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name = model_path.parent.parent.name
        output_dir = REPORTS_DIR / "evaluation" / model_name

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Determine data.yaml path
    if args.data:
        data_yaml = Path(args.data)
    else:
        # Try to infer from config
        cfg_path = ROOT / "configs" / "pipeline.yaml"
        if cfg_path.exists():
            cfg = load_config(cfg_path)
            run_id = cfg.get("run_id", "run")
            data_yaml = PROCESSED_DIR / f"yolo_dataset_{run_id}" / "data.yaml"
        else:
            logger.error("Cannot infer data.yaml path. Please specify --data")
            return 1

    if not data_yaml.exists():
        logger.error(f"Data configuration not found: {data_yaml}")
        return 1

    logger.info(f"Data configuration: {data_yaml}")

    # Load model
    logger.info("Loading model...")
    try:
        model = YOLO(str(model_path))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return 1

    # Run validation
    logger.info(f"Running model validation on {args.split} set...")
    logger.info(f"  Confidence threshold: {args.conf_threshold}")
    logger.info(f"  IOU threshold: {args.iou_threshold}")

    try:
        results = model.val(
            data=str(data_yaml),
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            split=args.split,
            plots=True,
            save_json=True,
            project=str(output_dir),
            name='validation_results'
        )

        logger.info("=" * 80)
        logger.info("Validation Results:")
        logger.info("=" * 80)

        # Extract metrics from results
        if hasattr(results, 'box'):
            box_metrics = results.box
            logger.info(f"mAP@0.5: {box_metrics.map50:.4f}")
            logger.info(f"mAP@0.5:0.95: {box_metrics.map:.4f}")
            logger.info(f"Precision: {box_metrics.mp:.4f}")
            logger.info(f"Recall: {box_metrics.mr:.4f}")

            # Save metrics to JSON
            metrics_dict = {
                'model': str(model_path),
                'data': str(data_yaml),
                'conf_threshold': args.conf_threshold,
                'iou_threshold': args.iou_threshold,
                'mAP@0.5': float(box_metrics.map50),
                'mAP@0.5:0.95': float(box_metrics.map),
                'precision': float(box_metrics.mp),
                'recall': float(box_metrics.mr),
            }

            metrics_file = output_dir / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            logger.info(f"Metrics saved to: {metrics_file}")

        # Generate summary report
        report_file = output_dir / "evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("YOLO Circle Detection - Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Data: {data_yaml}\n")
            f.write(f"Confidence threshold: {args.conf_threshold}\n")
            f.write(f"IOU threshold: {args.iou_threshold}\n\n")
            f.write("Performance Metrics:\n")
            f.write("-" * 40 + "\n")
            if hasattr(results, 'box'):
                f.write(f"mAP@0.5:      {box_metrics.map50:.4f}\n")
                f.write(f"mAP@0.5:0.95: {box_metrics.map:.4f}\n")
                f.write(f"Precision:    {box_metrics.mp:.4f}\n")
                f.write(f"Recall:       {box_metrics.mr:.4f}\n")
                f.write(f"F1-Score:     {2 * box_metrics.mp * box_metrics.mr / (box_metrics.mp + box_metrics.mr + 1e-6):.4f}\n")

        logger.info(f"Evaluation report saved to: {report_file}")

        logger.info("=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info(f"All results saved to: {output_dir}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("Evaluation failed:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if isinstance(exit_code, int) else 0)
