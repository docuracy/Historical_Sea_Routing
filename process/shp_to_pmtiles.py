import subprocess
from pathlib import Path

'''

Unlike .mbtiles, .pmtiles can be used directly in a web server without needing to run a separate tileserver.

'''

minZoom = 9
maxZoom = 9

# Paths
shapefile = Path("./data/osm_land_4326_unsplit/land_polygons.shp")
geojsonseq = Path("./data/osm_land_4326_unsplit/land_polygons.geojsonseq")
pmtiles = Path(f"../app/public/data/osm-coastlines-z{maxZoom}.pmtiles")

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

# Step 2: Convert GeoJSONSeq → pmtiles
print(f"Generating {pmtiles} with tippecanoe...")
try:
    subprocess.run([
        "tippecanoe",
        "-o", str(pmtiles),
        "-l", "coastlines",
        "-Z", f"{minZoom}",
        "-z", f"{maxZoom}",
        "--drop-densest-as-needed",
        "--coalesce-densest-as-needed",
        "--extend-zooms-if-still-dropping",
        "--simplify-only-low-zooms",
        "--projection=EPSG:4326",
        "--force",
        str(geojsonseq)
    ], check=True)
except subprocess.CalledProcessError as e:
    print("❌ Failed to generate pmtiles:", e)
    exit(1)

print("✅ Done.")
