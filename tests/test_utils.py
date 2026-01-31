"""Tests for utility functions."""

import pytest
from pathlib import Path
import tempfile
import yaml

from utils import load_config, resolve_path


class TestLoadConfig:
    """Test load_config function."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML file."""
        config_data = {
            "run_id": "test",
            "inputs": [{"id": "test_input", "raster": "data.tif", "masks_gpkg": "labels.gpkg"}]
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = load_config(config_file)
        assert result["run_id"] == "test"
        assert len(result["inputs"]) == 1

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises ValueError."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError) as exc_info:
            load_config(config_file)
        assert "Failed to parse YAML" in str(exc_info.value)


class TestResolvePath:
    """Test resolve_path function."""

    def test_absolute_path_unchanged(self):
        """Test that absolute paths are returned as-is."""
        root = Path("/home/user/project")
        abs_path = "/absolute/path/to/file.txt"
        result = resolve_path(root, abs_path)
        assert result == Path(abs_path)
        assert result.is_absolute()

    def test_relative_path_resolved(self):
        """Test that relative paths are resolved relative to root."""
        root = Path("/home/user/project")
        rel_path = "data/file.txt"
        result = resolve_path(root, rel_path)
        # Check that the path ends with the expected relative part
        assert str(result).endswith("project/data/file.txt")
        assert result.is_absolute()

    def test_path_object_input(self):
        """Test that Path objects are handled correctly."""
        root = Path("/home/user/project")
        rel_path = Path("data/file.txt")
        result = resolve_path(root, rel_path)
        # Check that the path ends with the expected relative part
        assert str(result).endswith("project/data/file.txt")

    def test_current_directory_reference(self):
        """Test handling of './' in relative paths."""
        root = Path("/home/user/project")
        rel_path = "./data/file.txt"
        result = resolve_path(root, rel_path)
        assert result.is_absolute()
        assert "project" in str(result)
