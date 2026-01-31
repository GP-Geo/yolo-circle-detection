"""Pydantic schema for pipeline configuration validation."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class InputConfig(BaseModel):
    """Configuration for a single input raster and labels."""
    id: str = Field(..., description="Unique identifier for this input")
    raster: str = Field(..., description="Path to raster file")
    masks_gpkg: str = Field(..., description="Path to GeoPackage with labels")
    masks_layer: str = Field(default="", description="Layer name in GeoPackage")


class BandsConfig(BaseModel):
    """Configuration for band selection."""
    ms: list[int] = Field(default_factory=list, description="Multispectral band indices")
    rgb: list[int] = Field(default_factory=list, description="RGB band indices")


class TilingConfig(BaseModel):
    """Configuration for tiling strategy."""
    use_ground_size: bool = Field(default=False, description="Use ground size in meters")
    tile_size_m: float = Field(default=0, description="Tile size in meters")
    tile_size_px: int = Field(default=512, description="Tile size in pixels")
    overlap_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Overlap fraction")

    @field_validator('tile_size_px')
    @classmethod
    def validate_divisible_by_32(cls, v: int) -> int:
        """Validate tile size is divisible by 32 (YOLO requirement)."""
        if v <= 0:
            raise ValueError("tile_size_px must be positive")
        if v % 32 != 0:
            raise ValueError(f"tile_size_px must be divisible by 32 (got {v})")
        return v


class LabelsConfig(BaseModel):
    """Configuration for label filtering."""
    object_class: str = Field(default="circle", description="Object class name")
    min_width_m: float = Field(default=0, ge=0, description="Minimum width in meters")
    min_height_m: float = Field(default=0, ge=0, description="Minimum height in meters")
    buffer_m: float = Field(default=0, ge=0, description="Buffer around labels in meters")


class SplitConfig(BaseModel):
    """Configuration for train/val/test split."""
    train: float = Field(default=0.7, gt=0, lt=1, description="Training set ratio")
    val: float = Field(default=0.2, gt=0, lt=1, description="Validation set ratio")
    test: float = Field(default=0.1, gt=0, lt=1, description="Test set ratio")
    strategy: str = Field(default="block", description="Split strategy")
    block_tiles: int = Field(default=4, gt=0, description="Tiles per block")
    seed: int = Field(default=42, description="Random seed")

    @model_validator(mode='after')
    def validate_ratios_sum(self) -> 'SplitConfig':
        """Validate that train + val + test = 1.0."""
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total:.3f})")
        return self


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    run_id: str = Field(..., description="Unique run identifier")
    inputs: list[InputConfig] = Field(..., min_length=1, description="Input rasters and labels")
    bands: BandsConfig = Field(default_factory=BandsConfig, description="Band selection")
    tiling: TilingConfig = Field(default_factory=TilingConfig, description="Tiling configuration")
    labels: LabelsConfig = Field(default_factory=LabelsConfig, description="Label filtering")
    split: SplitConfig = Field(default_factory=SplitConfig, description="Dataset split")


def validate_config(cfg: dict[str, Any]) -> PipelineConfig:
    """
    Validate pipeline configuration against schema.

    Args:
        cfg: Raw configuration dictionary

    Returns:
        Validated PipelineConfig object

    Raises:
        ValidationError: If configuration is invalid
    """
    return PipelineConfig(**cfg)
