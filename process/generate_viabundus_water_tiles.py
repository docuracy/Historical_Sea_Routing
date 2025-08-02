import concurrent.futures
import gzip
import subprocess
from pathlib import Path

from gitdb.util import mkdir
from tqdm import tqdm

from process.config import AOIS, head_directory

minZoom = 0
maxZoom = 10

AOI = AOIS[0]

data_dir = head_directory / "app" / "public" / "data"

# Paths
geojson = data_dir / "Viabundus-2-water-1500.geojson"
pbftiles = Path(f"../viabundus-water-tiles")
pbftiles.mkdir(exist_ok=True)

print(f"Generating {pbftiles} with tippecanoe...")
try:
    subprocess.run([
        "tippecanoe",
        "-l", "viabundus_water",
        "-Z", f"{minZoom}",
        "-z", f"{maxZoom}",
        "--output-to-directory", str(pbftiles),
        "--drop-densest-as-needed",
        "--coalesce-densest-as-needed",
        "--extend-zooms-if-still-dropping",
        "--simplify-only-low-zooms",
        "--projection=EPSG:4326",
        "--force",
        str(geojson)
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
            with gzip.open(file_path, 'rb') as gz:
                decompressed_data = gz.read()
            with open(file_path, 'wb') as f:
                f.write(decompressed_data)
        else:
            print(f"File not gzipped: {file_path}")
    except Exception as e:
        print(f"Error decompressing {file_path}: {e}")


# Rename .pbf → .mvt in zoom range
for z in range(minZoom, maxZoom + 1):
    zoom_dir = pbftiles / str(z)
    if not zoom_dir.exists():
        continue
    for pbf_file in zoom_dir.rglob("*.pbf"):
        mvt_file = pbf_file.with_suffix(".mvt")
        try:
            pbf_file.rename(mvt_file)
        except Exception as e:
            print(f"❌ Failed to rename {pbf_file}: {e}")

# Collect .mvt files in range
mvt_files = []
for z in range(minZoom, maxZoom + 1):
    zoom_dir = pbftiles / str(z)
    if zoom_dir.exists():
        mvt_files.extend(zoom_dir.rglob("*.mvt"))

print(f"Decompressing {len(mvt_files)} .mvt files in parallel...")

with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(decompress_if_gzipped, mvt_files), total=len(mvt_files)))

print("✅ Done decompressing.")