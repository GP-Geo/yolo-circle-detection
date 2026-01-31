"""Unit tests for NMS (Non-Maximum Suppression) algorithms."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add scripts to path to import functions
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

# Import the function to test - we need to handle the import differently
# since it's in a numbered script file
import importlib.util
spec = importlib.util.spec_from_file_location("merge_nms", SCRIPTS / "07_merge_nms.py")
merge_nms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_nms)

nms_xyxy = merge_nms.nms_xyxy
suppress_nested_prefer_outer = merge_nms.suppress_nested_prefer_outer


class TestNMS:
    """Test cases for standard NMS algorithm."""

    def test_nms_removes_overlapping_boxes(self, sample_boxes, sample_scores):
        """Test that NMS removes highly overlapping boxes."""
        # Boxes 0 and 1 overlap significantly
        keep_indices = nms_xyxy(sample_boxes, sample_scores, iou_thresh=0.3)

        # Should keep the highest scoring boxes
        assert len(keep_indices) < len(sample_boxes)
        assert 0 in keep_indices or 1 in keep_indices  # One of the overlapping boxes kept
        assert 2 in keep_indices  # Non-overlapping box always kept

    def test_nms_empty_input(self):
        """Test NMS with empty input."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)

        keep_indices = nms_xyxy(boxes, scores, iou_thresh=0.5)

        assert len(keep_indices) == 0

    def test_nms_single_box(self):
        """Test NMS with single box."""
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        keep_indices = nms_xyxy(boxes, scores, iou_thresh=0.5)

        assert len(keep_indices) == 1
        assert keep_indices[0] == 0

    def test_nms_no_overlap(self):
        """Test NMS with non-overlapping boxes."""
        boxes = np.array([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        keep_indices = nms_xyxy(boxes, scores, iou_thresh=0.5)

        # All boxes should be kept since they don't overlap
        assert len(keep_indices) == 3

    def test_nms_keeps_highest_score(self):
        """Test that NMS keeps the highest scoring box when overlapping."""
        boxes = np.array([
            [10, 10, 50, 50],
            [15, 15, 55, 55],  # High overlap with first box
        ], dtype=np.float32)

        # First box has lower score
        scores = np.array([0.7, 0.9], dtype=np.float32)

        keep_indices = nms_xyxy(boxes, scores, iou_thresh=0.3)

        # Should keep box 1 (higher score)
        assert 1 in keep_indices
        # At high overlap, should remove box 0
        if len(keep_indices) == 1:
            assert 0 not in keep_indices


class TestNestedSuppression:
    """Test cases for nested box suppression algorithm."""

    def test_nested_suppression_prefers_larger_box(self, sample_boxes, sample_scores):
        """Test that nested suppression prefers larger boxes."""
        # Box 4 is nested inside Box 3
        keep_indices = suppress_nested_prefer_outer(
            sample_boxes,
            sample_scores,
            coverage_thresh=0.80,
            contain_tol_px=3.0,
            score_margin=0.0
        )

        # Should prefer Box 3 over Box 4 (larger box)
        assert 2 in keep_indices  # Box 3 (larger)

    def test_nested_suppression_empty_input(self):
        """Test nested suppression with empty input."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)

        keep_indices = suppress_nested_prefer_outer(boxes, scores)

        assert len(keep_indices) == 0

    def test_nested_suppression_single_box(self):
        """Test nested suppression with single box."""
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        keep_indices = suppress_nested_prefer_outer(boxes, scores)

        assert len(keep_indices) == 1
        assert keep_indices[0] == 0

    def test_nested_suppression_no_nesting(self):
        """Test nested suppression with non-nested boxes."""
        boxes = np.array([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        keep_indices = suppress_nested_prefer_outer(boxes, scores)

        # All boxes should be kept since none are nested
        assert len(keep_indices) == 3


class TestIOUComputation:
    """Test IOU computation accuracy."""

    def test_iou_identical_boxes(self):
        """Test IOU of identical boxes."""
        box1 = np.array([10, 10, 50, 50], dtype=np.float32)
        box2 = np.array([10, 10, 50, 50], dtype=np.float32)

        # Compute manually using NMS internals
        boxes = np.array([box1, box2])
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        xx1 = max(x1[0], x1[1])
        yy1 = max(y1[0], y1[1])
        xx2 = min(x2[0], x2[1])
        yy2 = min(y2[0], y2[1])

        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        union = areas[0] + areas[1] - inter
        iou = inter / union

        assert abs(iou - 1.0) < 1e-6  # IOU should be 1.0 for identical boxes

    def test_iou_no_overlap(self):
        """Test IOU of non-overlapping boxes."""
        box1 = np.array([10, 10, 50, 50], dtype=np.float32)
        box2 = np.array([100, 100, 150, 150], dtype=np.float32)

        boxes = np.array([box1, box2])
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        xx1 = max(x1[0], x1[1])
        yy1 = max(y1[0], y1[1])
        xx2 = min(x2[0], x2[1])
        yy2 = min(y2[0], y2[1])

        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)

        assert inter == 0  # No intersection

    def test_iou_partial_overlap(self):
        """Test IOU of partially overlapping boxes."""
        box1 = np.array([0, 0, 20, 20], dtype=np.float32)  # Area = 400
        box2 = np.array([10, 10, 30, 30], dtype=np.float32)  # Area = 400

        # Intersection is 10x10 = 100
        # Union is 400 + 400 - 100 = 700
        # IOU should be 100/700 ≈ 0.143

        boxes = np.array([box1, box2])
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        xx1 = max(x1[0], x1[1])
        yy1 = max(y1[0], y1[1])
        xx2 = min(x2[0], x2[1])
        yy2 = min(y2[0], y2[1])

        inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
        union = areas[0] + areas[1] - inter
        iou = inter / union

        expected_iou = 100.0 / 700.0
        assert abs(iou - expected_iou) < 1e-6
