import subprocess
from pathlib import Path

minZoom = 0
maxZoom = 9

# Paths
shapefile = Path("./data/osm_land_4326_unsplit/land_polygons.shp")
geojsonseq = Path("./data/osm_land_4326_unsplit/land_polygons.geojsonseq")
pbftiles = Path(f"../osm-coastlines-pbftiles")

if not shapefile.exists():
    print(f"❌ Shapefile {shapefile} does not exist.")
    exit(1)

# Step 1: Convert shapefile → GeoJSONSeq
if not geojsonseq.exists():
    print(f"Converting {shapefile} to {geojsonseq}...")
    try:
        subprocess.run([
            "ogr2ogr",
            "-f", "GeoJSONSeq",
            "-wrapdateline",
            str(geojsonseq),
            str(shapefile)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Failed to convert shapefile:", e)
        exit(1)

# Step 2: Convert GeoJSONSeq → pbftiles
print(f"Generating {pbftiles} with tippecanoe...")
try:
    subprocess.run([
        "tippecanoe",
        "-l", "coastlines",
        "-Z", f"{minZoom}",
        "-z", f"{maxZoom}",
        "--output-to-directory", str(pbftiles),
        "--drop-densest-as-needed",
        "--coalesce-densest-as-needed",
        "--extend-zooms-if-still-dropping",
        "--simplify-only-low-zooms",
        "--projection=EPSG:4326",
        "--force",
        str(geojsonseq)
    ], check=True)
except subprocess.CalledProcessError as e:
    print("❌ Failed to generate pbftiles:", e)
    exit(1)

print("✅ Done.")
