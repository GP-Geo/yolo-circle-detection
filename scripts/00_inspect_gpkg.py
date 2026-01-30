from pathlib import Path
import sys
import geopandas as gpd
import fiona

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from paths import RAW_DIR

gpkg = RAW_DIR / "vectors" / "labeling_WV3.gpkg"

print("GPKG:", gpkg)
print("Exists:", gpkg.exists())

print("Layers:", fiona.listlayers(gpkg))

layer = fiona.listlayers(gpkg)[0]
gdf = gpd.read_file(gpkg, layer=layer)

print("Rows:", len(gdf))
print("CRS:", gdf.crs)
print("Geom types:", gdf.geom_type.value_counts())
print("Invalid geoms:", (~gdf.is_valid).sum())
print("Empty geoms:", gdf.geometry.is_empty.sum())
