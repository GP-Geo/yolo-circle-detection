"""Path utilities for resolving relative and absolute paths."""

from __future__ import annotations

from pathlib import Path


def resolve_path(root: Path, p: str | Path) -> Path:
    """
    Resolve a path relative to a root directory.

    If the path is absolute, it is returned as-is.
    If the path is relative, it is resolved relative to the root directory.

    Args:
        root: Root directory for relative path resolution
        p: Path to resolve (can be string or Path object)

    Returns:
        Resolved absolute path

    Examples:
        >>> resolve_path(Path("/home/user"), "data/file.txt")
        Path('/home/user/data/file.txt')
        >>> resolve_path(Path("/home/user"), "/absolute/path.txt")
        Path('/absolute/path.txt')
    """
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()
