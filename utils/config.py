"""Configuration utilities for loading and validating pipeline configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

try:
    from .config_schema import PipelineConfig, validate_config as _validate_config
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    PipelineConfig = None
    _validate_config = None


def load_config(path: Path, validate: bool = False) -> Union[dict[str, Any], 'PipelineConfig']:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML config file
        validate: If True, validate config against Pydantic schema

    Returns:
        Dictionary containing configuration, or PipelineConfig if validate=True

    Raises:
        ImportError: If pyyaml is not installed
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If validate=True and config is invalid
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Missing dependency 'pyyaml'. Install with: pip install pyyaml"
        ) from e

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config at {path}: {e}") from e

    if validate:
        if not PYDANTIC_AVAILABLE:
            raise ImportError(
                "Pydantic is required for config validation. "
                "Install with: pip install pydantic>=2.0.0"
            )
        return _validate_config(cfg)

    return cfg


def validate_config(cfg: dict[str, Any]) -> 'PipelineConfig':
    """
    Validate configuration dictionary against schema.

    Args:
        cfg: Configuration dictionary

    Returns:
        Validated PipelineConfig object

    Raises:
        ImportError: If pydantic is not installed
        ValidationError: If configuration is invalid
    """
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "Pydantic is required for config validation. "
            "Install with: pip install pydantic>=2.0.0"
        )
    return _validate_config(cfg)
