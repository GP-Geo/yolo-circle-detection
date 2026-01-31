"""
Inspect GeoPackage files and list their layers, geometry types, and basic statistics.

Usage:
    # Inspect specific GeoPackage files
    python scripts/00_inspect_gpkg.py data/external/active/s2_image1.gpkg data/external/active/s2_image2.gpkg

    # Inspect all GPKGs in active directory (default)
    python scripts/00_inspect_gpkg.py

    # Inspect all GPKGs in a specific directory
    python scripts/00_inspect_gpkg.py data/raw/vectors/
"""

from pathlib import Path
import sys
import argparse
import geopandas as gpd
import fiona

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from paths import EXTERNAL_DIR


def inspect_gpkg(gpkg_path: Path, verbose: bool = True) -> dict:
    """
    Inspect a GeoPackage file and return statistics.

    Args:
        gpkg_path: Path to the GeoPackage file
        verbose: If True, print detailed information

    Returns:
        Dictionary with layer statistics
    """
    if not gpkg_path.exists():
        print(f"❌ File not found: {gpkg_path}")
        return {}

    if verbose:
        print("=" * 80)
        print(f"📦 GeoPackage: {gpkg_path}")
        print("=" * 80)

    try:
        layers = fiona.listlayers(str(gpkg_path))
        if verbose:
            print(f"Layers found: {len(layers)}")
            print()

        stats = {}
        for layer in layers:
            if verbose:
                print(f"Layer: {layer}")
                print("-" * 40)

            gdf = gpd.read_file(gpkg_path, layer=layer)

            layer_stats = {
                'rows': len(gdf),
                'crs': str(gdf.crs) if gdf.crs else 'None',
                'geom_types': gdf.geom_type.value_counts().to_dict(),
                'invalid_geoms': (~gdf.is_valid).sum(),
                'empty_geoms': gdf.geometry.is_empty.sum(),
                'columns': list(gdf.columns),
            }

            if verbose:
                print(f"  Rows: {layer_stats['rows']}")
                print(f"  CRS: {layer_stats['crs']}")
                print(f"  Geometry types:")
                for geom_type, count in layer_stats['geom_types'].items():
                    print(f"    - {geom_type}: {count}")
                print(f"  Invalid geometries: {layer_stats['invalid_geoms']}")
                print(f"  Empty geometries: {layer_stats['empty_geoms']}")
                print(f"  Columns: {', '.join(layer_stats['columns'])}")

                # Show bounds
                if len(gdf) > 0:
                    bounds = gdf.total_bounds
                    print(f"  Bounds (minx, miny, maxx, maxy): {bounds}")

                print()

            stats[layer] = layer_stats

        if verbose:
            print()

        return stats

    except Exception as e:
        print(f"❌ Error inspecting {gpkg_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Inspect GeoPackage files and list layers, geometry types, and statistics."
    )
    parser.add_argument(
        'paths',
        nargs='*',
        type=str,
        help='Paths to GeoPackage files or directories. If not provided, searches data/external/active/'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Determine which files to inspect
    gpkg_files = []

    if not args.paths:
        # Default: look in active directory
        active_dir = EXTERNAL_DIR / "active"
        if active_dir.exists():
            gpkg_files = list(active_dir.glob("*.gpkg"))
        if not gpkg_files:
            print(f"No GeoPackage files found in {active_dir}")
            print("Usage: python scripts/00_inspect_gpkg.py <path_to_gpkg> [<path_to_gpkg2> ...]")
            return 1
    else:
        # User provided paths
        for path_str in args.paths:
            path = Path(path_str)
            if path.is_dir():
                # If directory, find all .gpkg files
                gpkg_files.extend(path.glob("*.gpkg"))
            elif path.suffix.lower() == '.gpkg':
                # If file, add it
                gpkg_files.append(path)
            else:
                print(f"⚠️  Skipping non-GPKG file: {path}")

    if not gpkg_files:
        print("No GeoPackage files found.")
        return 1

    # Inspect each file
    print(f"Found {len(gpkg_files)} GeoPackage file(s) to inspect\n")

    for gpkg_path in gpkg_files:
        inspect_gpkg(gpkg_path, verbose=not args.quiet)

    print("✅ Inspection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
