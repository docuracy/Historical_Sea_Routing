import concurrent.futures
import gzip
import json
import subprocess
from pathlib import Path

import fiona
from shapely.geometry.geo import mapping

from tqdm import tqdm

from process.config import AOIS, head_directory

minZoom = 0
maxZoom = 9

AOI = AOIS[0]

data_dir = head_directory / "app" / "public" / "data" / AOI["name"]

graph_gpkg = data_dir / "graph.gpkg"
edges_gpkg = data_dir / "edges.gpkg"
output_dir = data_dir / "graph_tiles"

output_dir.mkdir(exist_ok=True)


# Step 1: Extract layers from graph.gpkg and edges.gpkg and combine into one GeoJSONSeq

def extract_layers_to_geojsonseq(gpkg_path: Path, layer_prefix: str, tmpfile_path: Path):
    """
    Extract all layers starting with layer_prefix from the gpkg file,
    write features sequentially to tmpfile_path in GeoJSONSeq format.
    """
    with fiona.Env():
        with open(tmpfile_path, "w", encoding="utf-8") as outfile:
            with fiona.open(gpkg_path) as src:
                layers = fiona.listlayers(str(gpkg_path))
            for layer_name in layers:
                if not layer_name.startswith(layer_prefix):
                    continue
                with fiona.open(gpkg_path, layer=layer_name) as layer:
                    for feature in layer:
                        geojson = mapping(feature["geometry"])
                        # Convert the fiona.collection.Properties object to a standard dictionary
                        geojson["properties"] = dict(feature["properties"])
                        outfile.write(json.dumps(geojson) + "\n")

# Extract hex polygons
hexes_seq = output_dir / "hexes.geojsonseq"
print("Extracting hex polygon layers...")
extract_layers_to_geojsonseq(graph_gpkg, "hexes_r", hexes_seq)

# Extract edges linestrings
edges_seq = output_dir / "edges.geojsonseq"
print("Extracting edge line layers...")
extract_layers_to_geojsonseq(edges_gpkg, "edges_r", edges_seq)

# Step 2: Convert GeoJSONSeq → output_dir
print(f"Generating {output_dir} with tippecanoe...")
try:
    subprocess.run([
        "tippecanoe",
        "-Z", f"{minZoom}",
        "-z", f"{maxZoom}",
        "--output-to-directory", str(output_dir),
        "--drop-densest-as-needed",
        "--coalesce-densest-as-needed",
        "--extend-zooms-if-still-dropping",
        "--simplify-only-low-zooms",
        "--projection=EPSG:4326",
        "--force",
        # Input files with layer names assigned:
        "-l", "hexes",
        str(hexes_seq),
        "-l", "edges",
        str(edges_seq),
    ], check=True)
except subprocess.CalledProcessError as e:
    print("❌ Failed to generate pbftiles:", e)
    exit(1)

# Clean up temporary files
hexes_seq.unlink(missing_ok=True)
edges_seq.unlink(missing_ok=True)

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
    zoom_dir = output_dir / str(z)
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
    zoom_dir = output_dir / str(z)
    if zoom_dir.exists():
        mvt_files.extend(zoom_dir.rglob("*.mvt"))

print(f"Decompressing {len(mvt_files)} .mvt files in parallel...")

with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(decompress_if_gzipped, mvt_files), total=len(mvt_files)))

print("✅ Done decompressing.")
