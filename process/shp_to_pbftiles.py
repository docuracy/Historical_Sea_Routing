import gzip
import subprocess
from pathlib import Path

minZoom = 0
maxZoom = 7

# Paths
shapefile = Path("./data/osm_land_4326_unsplit/land_polygons.shp")
geojsonseq = Path("./data/osm_land_4326_unsplit/land_polygons.geojsonseq")
pbftiles = Path(f"../osm-countries-tiles")

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

print("Renaming .pbf files to .mvt and decompressing if gzipped...")

def decompress_if_gzipped(file_path: Path):
    """Check if file is gzipped and decompress in-place if yes."""
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(2)
        if magic == b'\x1f\x8b':  # gzip magic bytes
            print(f"Decompressing gzipped file: {file_path}")
            # Read gzip content and overwrite file with decompressed data
            with gzip.open(file_path, 'rb') as gz:
                decompressed_data = gz.read()
            with open(file_path, 'wb') as f:
                f.write(decompressed_data)
        else:
            print(f"File not gzipped: {file_path}")
    except Exception as e:
        print(f"Error decompressing {file_path}: {e}")

# Rename .pbf → .mvt and decompress
for pbf_file in pbftiles.rglob("*.pbf"):
    mvt_file = pbf_file.with_suffix(".mvt")
    try:
        pbf_file.rename(mvt_file)
        decompress_if_gzipped(mvt_file)
    except Exception as e:
        print(f"Failed to rename or decompress {pbf_file}: {e}")

print("✅ Done.")
