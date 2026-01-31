"""Tests for configuration validation."""

import pytest
from pathlib import Path

try:
    from pydantic import ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="pydantic not installed")

from utils.config_schema import validate_config, PipelineConfig


class TestConfigValidation:
    """Test configuration validation with Pydantic."""

    @pytest.fixture
    def valid_config(self):
        """Return a valid configuration dictionary."""
        return {
            "run_id": "test_run",
            "inputs": [
                {
                    "id": "test_input",
                    "raster": "data/test.tif",
                    "masks_gpkg": "data/test.gpkg",
                    "masks_layer": "labels"
                }
            ],
            "bands": {
                "ms": [1, 2, 3],
                "rgb": [1, 2, 3]
            },
            "tiling": {
                "tile_size_px": 512,
                "overlap_fraction": 0.5
            },
            "labels": {
                "object_class": "circle",
                "min_width_m": 5.0,
                "min_height_m": 5.0
            },
            "split": {
                "train": 0.7,
                "val": 0.2,
                "test": 0.1
            }
        }

    def test_valid_config_passes(self, valid_config):
        """Test that a valid config passes validation."""
        config = validate_config(valid_config)
        assert isinstance(config, PipelineConfig)
        assert config.run_id == "test_run"
        assert len(config.inputs) == 1

    def test_invalid_tile_size_not_divisible_by_32(self, valid_config):
        """Test that tile size not divisible by 32 fails."""
        valid_config["tiling"]["tile_size_px"] = 500
        with pytest.raises(ValidationError) as exc_info:
            validate_config(valid_config)
        assert "divisible by 32" in str(exc_info.value)

    def test_invalid_overlap_fraction_too_high(self, valid_config):
        """Test that overlap fraction > 1.0 fails."""
        valid_config["tiling"]["overlap_fraction"] = 1.5
        with pytest.raises(ValidationError):
            validate_config(valid_config)

    def test_invalid_overlap_fraction_negative(self, valid_config):
        """Test that negative overlap fraction fails."""
        valid_config["tiling"]["overlap_fraction"] = -0.1
        with pytest.raises(ValidationError):
            validate_config(valid_config)

    def test_invalid_split_ratios_dont_sum_to_one(self, valid_config):
        """Test that split ratios not summing to 1.0 fail."""
        valid_config["split"]["train"] = 0.5
        valid_config["split"]["val"] = 0.3
        valid_config["split"]["test"] = 0.3  # Sum = 1.1
        with pytest.raises(ValidationError) as exc_info:
            validate_config(valid_config)
        assert "sum to 1.0" in str(exc_info.value)

    def test_missing_required_field_run_id(self, valid_config):
        """Test that missing run_id fails."""
        del valid_config["run_id"]
        with pytest.raises(ValidationError):
            validate_config(valid_config)

    def test_empty_inputs_list(self, valid_config):
        """Test that empty inputs list fails."""
        valid_config["inputs"] = []
        with pytest.raises(ValidationError):
            validate_config(valid_config)

    def test_default_values_applied(self):
        """Test that default values are applied for optional fields."""
        minimal_config = {
            "run_id": "minimal",
            "inputs": [
                {
                    "id": "input1",
                    "raster": "test.tif",
                    "masks_gpkg": "test.gpkg"
                }
            ]
        }
        config = validate_config(minimal_config)
        assert config.tiling.tile_size_px == 512  # default
        assert config.split.train == 0.7  # default
        assert config.split.val == 0.2  # default
        assert config.split.test == 0.1  # default
