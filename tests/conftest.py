"""Pytest configuration and fixtures for YOLO circle detection tests."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


@pytest.fixture
def sample_boxes():
    """Fixture providing sample bounding boxes for testing."""
    return np.array([
        [10, 10, 50, 50],   # Box 1
        [30, 30, 70, 70],   # Box 2 (overlaps with Box 1)
        [100, 100, 150, 150],  # Box 3 (no overlap)
        [105, 105, 145, 145],  # Box 4 (nested in Box 3)
    ], dtype=np.float32)


@pytest.fixture
def sample_scores():
    """Fixture providing confidence scores."""
    return np.array([0.9, 0.8, 0.95, 0.7], dtype=np.float32)


@pytest.fixture
def sample_image_size():
    """Fixture providing standard image size."""
    return 512
