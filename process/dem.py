import io
import math
import time

import h3
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.transform import from_bounds
from shapely.geometry.geo import shape

from process.config import copernicus_data_directory

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

CACHE_DIR = copernicus_data_directory / "terrarium_cache"


def retry(max_retries=3, backoff_factor=1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, requests.HTTPError) as e:
                    retries += 1
                    if retries > max_retries:
                        raise
                    wait = backoff_factor * (2 ** (retries - 1))
                    print(f"Retry {retries}/{max_retries} after error: {e}. Waiting {wait:.1f}s...")
                    time.sleep(wait)

        return wrapper

    return decorator


def ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def tile_cache_path(zoom, x, y):
    return CACHE_DIR / str(zoom) / str(x) / f"{y}.tif"


def save_tile_to_cache(dataset, zoom, x, y):
    tile_path = tile_cache_path(zoom, x, y)
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tile_path, 'w', **dataset.meta) as dst:
        dst.write(dataset.read(1), 1)


@retry(max_retries=3, backoff_factor=1)
def fetch_terrarium_tile(zoom, x, y):
    url = TERRARIUM_URL.format(z=zoom, x=x, y=y)
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content))
    arr = np.array(img)
    elevation = (arr[:, :, 0].astype('int32') * 256 +
                 arr[:, :, 1].astype('int32') +
                 arr[:, :, 2].astype('int32') / 256) - 32768
    return elevation


def build_dataset_from_elevation(elevation, zoom, x, y):
    n = 2 ** zoom
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))

    transform = from_bounds(lon_left, lat_bottom, lon_right, lat_top, elevation.shape[1], elevation.shape[0])
    memfile = rasterio.io.MemoryFile()
    dataset = memfile.open(
        driver='GTiff',
        height=elevation.shape[0],
        width=elevation.shape[1],
        count=1,
        dtype=elevation.dtype,
        transform=transform,
        crs=CRS.from_epsg(4326),
    )
    dataset.write(elevation, 1)
    return dataset


def fetch_and_cache_tile(zoom, x, y):
    tile_path = tile_cache_path(zoom, x, y)
    if tile_path.exists():
        return

    elevation = fetch_terrarium_tile(zoom, x, y)
    dataset = build_dataset_from_elevation(elevation, zoom, x, y)
    save_tile_to_cache(dataset, zoom, x, y)
    dataset.close()


def deg2num(lat_deg, lon_deg, zoom):
    try:
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return x, y
    except Exception as e:
        print(f"deg2num error for lat={lat_deg}, lon={lon_deg}, zoom={zoom}: {e}")
        return None


def h3_max_elevation(h3_id, zoom=12, hex_shape=None):
    """
    Compute the maximum elevation within an H3 cell.

    Args:
        h3_id (str): The H3 cell ID.
        zoom (int): Zoom level for fetching Terrarium tiles.
        hex_shape (shapely.geometry.Polygon, optional): Precomputed cell shape. If not provided, it will be computed.

    Returns:
        float or None: Maximum elevation in metres, or None if no valid data found.
    """
    # 1. Get H3 polygon & bounds in lat/lon

    if hex_shape is None:
        hex_shape = shape(h3.cells_to_h3shape([h3_id]).__geo_interface__)

    lon_min, lat_min, lon_max, lat_max = hex_shape.bounds

    # 2. Determine intersecting tiles
    x_min, y_max = deg2num(lat_min, lon_min, zoom)
    x_max, y_min = deg2num(lat_max, lon_max, zoom)

    max_elevation = None

    # 3. For each tile, load cached or fetch, then clip
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            # Load from cache or fetch
            tile_path = tile_cache_path(zoom, x, y)
            if not tile_path.exists():
                fetch_and_cache_tile(zoom, x, y)

            with rasterio.open(tile_path) as src:
                try:
                    # 4. Clip tile by H3 polygon
                    clipped, _ = mask(src, [hex_shape], crop=True, nodata=None)
                    # clipped is a 3D array: (bands, rows, cols)
                    data = clipped[0]
                    # Filter out no-data or invalid values (like zero or negative if applicable)
                    valid_data = data[data != src.nodata]
                    if valid_data.size == 0:
                        continue
                    tile_max = valid_data.max()

                    # 5. Track max elevation over all tiles
                    if max_elevation is None or tile_max > max_elevation:
                        max_elevation = tile_max
                except Exception:
                    # If masking fails or tile does not overlap, skip
                    continue

    return max_elevation


if __name__ == "__main__":
    ensure_cache_dir()

    # Example usage of h3_max_elevation
    h3_id = "8928308280fffff"  # Example H3 cell ID
    print(f"Max elevation for H3 cell {h3_id}: {h3_max_elevation(h3_id)} metres")

    # Check for coordinates of a known mountain
    lat, lon = 27.9881, 86.9250  # Coordinates of Everest
    h3_id = h3.latlng_to_cell(lat, lon, 5)
    print(f"Max elevation for H3 cell {h3_id}: {h3_max_elevation(h3_id)} metres")
