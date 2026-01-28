from pathlib import Path
import geopandas as gpd
import fiona

ROOT = Path(__file__).resolve().parents[1]  # .../YOLO_detections
gpkg = ROOT / "00_raw" / "vectors" / "labeling_WV3.gpkg"

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

