import gzip
import json
import logging
import math
import os
import shutil
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fiona
import h3
import msgpack
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from tqdm import tqdm

from process.config import AOIS, head_directory, copernicus_data_directory, COASTAL_SEA_RESOLUTION, datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ZARR_DIR = copernicus_data_directory / "zarr"
BACKUP_DIR = copernicus_data_directory / "zarr_backups"

import warnings

warnings.filterwarnings("ignore",
                        message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*")


class DatasetCache:
    def __init__(self, wind, current, weather, bathymetry, daylight, visibility_distance):
        self.wind = wind
        self.current = current
        self.weather = weather
        self.bathymetry = bathymetry
        self.daylight = daylight
        self.visibility_distance = visibility_distance
        self.spatial_indexes = {}

        self._build_spatial_indexes()

    def _build_spatial_indexes(self):
        datasets = {
            "wind": self.wind,
            "current": self.current,
            "weather": self.weather,
            "bathymetry": self.bathymetry,
            # "daylight": << DO NOT INDEX
            # "visibility_distance": << DO NOT INDEX
        }

        for name, ds in datasets.items():
            if ds:
                self.spatial_indexes[name] = self._build_tree(ds)

    def _build_tree(self, ds):
        lat = ds.latitude.values
        lng = ds.longitude.values
        lat_grid, lon_grid = np.meshgrid(lat, lng, indexing='ij')
        coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        tree = cKDTree(coords)
        return {
            "tree": tree,
            "grid_shape": lat_grid.shape,
            "lat_grid": lat_grid,
            "lon_grid": lon_grid,
        }

    def query_index(self, name, lat, lng):
        tree_data = self.spatial_indexes[name]
        tree = tree_data["tree"]
        grid_shape = tree_data["grid_shape"]
        _, idx = tree.query(np.array([[lat, lng]]))
        return np.unravel_index(idx, grid_shape)

    def batch_query_indices(self, name, lat_array, lon_array):
        tree_data = self.spatial_indexes[name]
        tree = tree_data["tree"]
        lat_shape = tree_data["lat_shape"]
        query_points = np.column_stack([lat_array, lon_array])
        _, idx = tree.query(query_points)
        return np.unravel_index(idx, lat_shape)


# --- Global worker cache ---
_worker_dataset_cache = None


def worker_init(cache_paths):
    global _worker_dataset_cache
    wind = None
    current = None
    weather = None
    bathymetry = None
    daylight = None
    visibility_distance = None

    if "wind" in cache_paths:
        wind = xr.open_zarr(cache_paths["wind"], consolidated=True)
    if "current" in cache_paths:
        current = xr.open_zarr(cache_paths["current"], consolidated=True)
    if "weather" in cache_paths:
        weather = xr.open_zarr(cache_paths["weather"], consolidated=True)
    if "bathymetry" in cache_paths:
        bathymetry = xr.open_zarr(cache_paths["bathymetry"], consolidated=True,
                                  chunks={"latitude": 100, "longitude": 100})
    if "daylight" in cache_paths:
        daylight = xr.open_zarr(cache_paths["daylight"], consolidated=True, chunks=None)
    if "visibility_distance" in cache_paths:
        visibility_distance = xr.open_zarr(cache_paths["visibility_distance"], consolidated=True)

    _worker_dataset_cache = DatasetCache(wind, current, weather, bathymetry, daylight, visibility_distance)


def estimate_visibility(temp_depression: xr.DataArray,
                        precip_rate: xr.DataArray,
                        low_cloud_cover: xr.DataArray,
                        max_visibility_m: float = 50000.0,  # Max visibility 50 km
                        min_visibility_m: float = 10.0,  # Min visibility 10 m, avoid 0
                        td_sensitivity_factor: float = 10000.0,  # Metres * degrees C for temp depression
                        td_epsilon: float = 0.5,  # Small constant to prevent div by zero for temp depression
                        precip_sensitivity_factor: float = 5000.0,  # Metres * (kg/m^2) for precipitation
                        precip_epsilon: float = 0.01,  # Small constant to prevent div by zero for precipitation
                        lcc_sensitivity_factor: float = 5000.0,
                        lcc_epsilon: float = 0.01
                        ) -> xr.DataArray:
    """
    Estimates atmospheric visibility in metres based on temperature depression,
    precipitation rate, and low cloud cover.
    """

    # Ensure all inputs are positive for calculations where appropriate
    temp_depression = temp_depression.clip(min=0)
    precip_rate = precip_rate.clip(min=0)

    # 1. Visibility from Humidity (Temperature Depression)
    # Lower temp_depression (higher humidity/fog) -> lower visibility
    # Add td_epsilon to prevent division by zero or extremely large values when Td is near zero.
    vis_from_humidity = td_sensitivity_factor / (temp_depression + td_epsilon)

    # 2. Visibility from Precipitation
    # Higher precip_rate -> lower visibility
    # Add precip_epsilon to prevent division by zero or extremely large values when precip is near zero.
    vis_from_precip = precip_sensitivity_factor / (precip_rate + precip_epsilon)

    # 3. Visibility from Low Cloud Cover (LCC)
    # Higher LCC (closer to 1) -> lower visibility
    lcc_impact_inverse = 1 / (low_cloud_cover + lcc_epsilon)
    vis_from_lcc = lcc_impact_inverse * lcc_sensitivity_factor

    # 4. Combine all factors by taking the minimum
    combined_visibility = xr.concat(
        [vis_from_humidity, vis_from_precip, vis_from_lcc],
        dim="visibility_component"
    ).min(dim="visibility_component")

    # 5. Clip the final result to the defined min and max visibility
    final_visibility = combined_visibility.clip(min=min_visibility_m, max=max_visibility_m)

    return final_visibility.rename("visibility_m")


def get_weather(lat, lng):
    global _worker_dataset_cache

    try:
        i, j = _worker_dataset_cache.query_index("weather", lat, lng)
        ds = _worker_dataset_cache.weather[["tp", "t2m", "d2m", "lcc"]].isel(latitude=i, longitude=j)

        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        # Precompute derived variables
        t2m_c = ds["t2m"] - 273.15
        d2m_c = ds["d2m"] - 273.15
        temp_depression = (t2m_c - d2m_c).clip(min=0).rename("temp_depression")
        precip_rate = ds["tp"].clip(min=0).rename("precip_rate")
        low_cloud_cover = ds["lcc"].rename("low_cloud_cover")

        # Estimate visibility once for all timestamps
        visibility = estimate_visibility(
            temp_depression=temp_depression,
            precip_rate=precip_rate,
            low_cloud_cover=low_cloud_cover,
        )

        # Group by month and compute mean
        monthly_mean = visibility.groupby("time.month").mean(dim="time").compute()

        # Fill in missing months with max visibility
        visibility_m = [int(monthly_mean.sel(month=m).item()) if m in monthly_mean.month else 50000
                        for m in range(1, 13)]

        return visibility_m

    except Exception as e:
        logger.warning(f"Visibility lookup failed for edge: {e}")
        return np.zeros(12, dtype=int)


def get_daylight_ratios(latlon: tuple):
    global _worker_dataset_cache
    lat, lon = latlon
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of range")

    ds = _worker_dataset_cache.daylight

    ratios = []
    for month in range(1, 13):
        try:
            interpolated = ds.daylight_ratio.interp(latitude=lat, month=month)
            ratios.append(int(interpolated.compute().item() * 100))  # Convert to percentage
        except Exception as e:
            logger.warning(f"Daylight lookup failed at {latlon} for month {month}: {e}")
            ratios.append(50)

    return ratios


def weighted_circular_mean_deg(angles_deg, weights, return_int=True):
    if len(angles_deg) == 0 or len(weights) == 0 or np.sum(weights) == 0:
        return 0.0
    angles_rad = np.radians(angles_deg)
    sum_sin = np.sum(weights * np.sin(angles_rad))
    sum_cos = np.sum(weights * np.cos(angles_rad))
    mean_angle = np.degrees(np.arctan2(sum_sin, sum_cos)) % 360
    return int(mean_angle) if return_int else mean_angle


def get_flow(lat, lng, angle, dataset, u_var, v_var):
    global _worker_dataset_cache
    global _bin_edges

    try:
        i, j = _worker_dataset_cache.query_index(dataset, lat, lng)
        ds = getattr(_worker_dataset_cache, dataset).isel(latitude=i, longitude=j)

        # Extract u and v time series
        u = ds[u_var].values
        v = ds[v_var].values

        # Compute speed and direction
        speed = np.sqrt(u ** 2 + v ** 2)
        direction = (np.degrees(np.arctan2(u, v)) + 360) % 360  # 0–360°

        # Edge vector (unit) based on given angle
        edge_rad = np.radians(angle)
        edge_dx = np.sin(edge_rad)
        edge_dy = np.cos(edge_rad)

        # Dot product to classify forward/reverse
        dot = u * edge_dx + v * edge_dy
        is_forward = dot >= 0

        # Get months per timepoint
        months = ds["time"].dt.month.values

        # Initialise outputs
        forward_angle = np.zeros(12, dtype=int)
        forward_mag = np.zeros(12, dtype=float)
        reverse_angle = np.zeros(12, dtype=int)
        reverse_mag = np.zeros(12, dtype=float)

        for month in range(1, 13):
            idx = np.where(months == month)[0]
            if len(idx) == 0:
                continue

            month_dir = direction[idx]
            month_mag = speed[idx]
            month_is_forward = is_forward[idx]

            # Forward
            f_dir = month_dir[month_is_forward]
            f_mag = month_mag[month_is_forward]
            if len(f_mag) > 0 and np.sum(f_mag) > 0:
                forward_angle[month - 1] = weighted_circular_mean_deg(f_dir, f_mag)
                forward_mag[month - 1] = float(np.mean(f_mag))

            # Reverse
            r_dir = month_dir[~month_is_forward]
            r_mag = month_mag[~month_is_forward]
            if len(r_mag) > 0 and np.sum(r_mag) > 0:
                reverse_angle[month - 1] = weighted_circular_mean_deg(r_dir, r_mag)
                reverse_mag[month - 1] = float(np.mean(r_mag))

        return (
            forward_angle.tolist(),
            forward_mag.tolist(),
            reverse_angle.tolist(),
            reverse_mag.tolist(),
        )

    except Exception as e:
        logger.warning(f"Wind lookup failed for edge: {e}")
        return [0] * 12, [0.0] * 12, [0] * 12, [0.0] * 12


def process_node(item):
    """
    Returns:
      - A single row for the `nodes` table
    """
    global _worker_dataset_cache
    h3id, lat, lng, dist_m = item

    # Get nearest index into bathymetry grid
    try:
        i, j = _worker_dataset_cache.query_index("bathymetry", lat, lng)
        bathy_val = _worker_dataset_cache.bathymetry["deptho"].values[i, j]
        bathymetry = None if np.isnan(bathy_val) else int(bathy_val.item())
    except Exception as e:
        logger.warning(f"Bathymetry lookup failed for node {h3id}: {e}")
        bathymetry = None

    try:
        clear_land = _worker_dataset_cache.visibility_distance["distance_to_visible_land_m"].sel(
            hex_id=str(h3id)).compute().item()
        if np.isnan(clear_land):
            clear_land = None
        else:
            clear_land = int(clear_land)
    except Exception as e:
        clear_land = None

    node_update_row = (bathymetry, clear_land, h3id)

    return (node_update_row)


def process_edge(item):
    """
    Returns:
      - A list of 12 rows for the `edge_monthly` table (one per month)
    """
    global _worker_dataset_cache
    source_h3id, target_h3id, midpoint_lat, midpoint_lng, length_m, dx, dy, angle = item

    visibility_m = get_weather(midpoint_lat, midpoint_lng)
    daylight_ratios = get_daylight_ratios((midpoint_lat, midpoint_lng))
    forward_wind_angle, forward_wind_mag, reverse_wind_angle, reverse_wind_mag = get_flow(
        midpoint_lat, midpoint_lng, angle, dataset="wind", u_var="eastward_wind", v_var="northward_wind")

    forward_current_angle, forward_current_mag, reverse_current_angle, reverse_current_mag = get_flow(
        midpoint_lat, midpoint_lng, angle, dataset="current", u_var="utotal", v_var="vtotal")

    data = zip(
        visibility_m, daylight_ratios,
        forward_wind_angle, forward_wind_mag,
        forward_current_angle, forward_current_mag,
        reverse_wind_angle, reverse_wind_mag,
        reverse_current_angle, reverse_current_mag
    )

    return ([
        (source_h3id, target_h3id, month, *values)
        for month, values in enumerate(data, start=1)
    ])


def compute_direction_vector_and_angle(latlon1, latlon2):
    """
    Compute the azimuth angle and corresponding unit vector pointing from latlon1 to latlon2.

    Returns:
        (dx, dy): Unit vector components.
        angle_deg: Forward angle in degrees (0 = North, increasing clockwise).
    """
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2

    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δλ = np.radians(lon2 - lon1)

    x = np.sin(Δλ) * np.cos(φ2)
    y = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(Δλ)

    θ_rad = (np.arctan2(x, y) + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2π)
    θ_deg = np.degrees(θ_rad)

    dx = np.sin(θ_rad)
    dy = np.cos(θ_rad)

    return (dx, dy), θ_deg


def geodetic_midpoint(source_latlng, target_latlng):
    lat1, lon1 = source_latlng
    lat2, lon2 = target_latlng

    # Convert degrees to radians
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])

    # Convert to Cartesian coordinates
    x1 = np.cos(lat1_rad) * np.cos(lon1_rad)
    y1 = np.cos(lat1_rad) * np.sin(lon1_rad)
    z1 = np.sin(lat1_rad)

    x2 = np.cos(lat2_rad) * np.cos(lon2_rad)
    y2 = np.cos(lat2_rad) * np.sin(lon2_rad)
    z2 = np.sin(lat2_rad)

    # Compute average vector
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    z = (z1 + z2) / 2

    # Convert back to latitude/longitude
    lon_mid = np.arctan2(y, x)
    hyp = np.sqrt(x ** 2 + y ** 2)
    lat_mid = np.arctan2(z, hyp)

    # Convert radians back to degrees
    return np.degrees(lat_mid), np.degrees(lon_mid)


def init_db(db_path, node_gpkg, edge_gpkg, sqlite_batch_size=10_000, testmode=False) -> tuple:
    if testmode:
        logger.info("Running in test mode, using smaller batch size for SQLite operations.")
        sqlite_batch_size = 100

    skip_creation = False
    if db_path.exists():
        skip_creation = True

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if skip_creation:
        logger.info(f"Using existing database at {db_path}.")
    else:
        # Create nodes table if not exists
        c.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            h3id TEXT PRIMARY KEY,
            lat REAL,
            lng REAL,
            bathymetry INTEGER,
            clear_land INTEGER,
            dist_m INTEGER
        )
        ''')

        # Create edges table if not exists
        c.execute('''
        CREATE TABLE IF NOT EXISTS edges (
            source_h3id TEXT,
            target_h3id TEXT,
            midpoint_lat REAL,
            midpoint_lng REAL,
            length_m INTEGER,
            dx REAL,
            dy REAL,
            angle INTEGER,
            PRIMARY KEY (source_h3id, target_h3id)
            FOREIGN KEY (source_h3id) REFERENCES nodes(h3id),
            FOREIGN KEY (target_h3id) REFERENCES nodes(h3id)
        )
        ''')

        # Create edge_monthly table if not exists
        c.execute('''
        CREATE TABLE IF NOT EXISTS edge_monthly (
            source_h3id TEXT,
            target_h3id TEXT,
            month INTEGER,
            visibility_m INTEGER,
            daylight_ratio INTEGER,
            forward_wind_angle INTEGER,
            forward_wind_mag REAL,
            forward_current_angle INTEGER,
            forward_current_mag REAL,
            reverse_wind_angle INTEGER,
            reverse_wind_mag REAL,
            reverse_current_angle INTEGER,
            reverse_current_mag REAL,
            PRIMARY KEY (source_h3id, target_h3id, month)
            FOREIGN KEY (source_h3id) REFERENCES nodes(h3id),
            FOREIGN KEY (target_h3id) REFERENCES nodes(h3id)
        )
        ''')

        # Create indexes for faster lookups
        c.execute('CREATE INDEX idx_edge_monthly_month ON edge_monthly(month)')
        c.execute('CREATE INDEX idx_edges_source ON edges(source_h3id)')
        c.execute('CREATE INDEX idx_edges_target ON edges(target_h3id)')

        conn.commit()

        # Now read the node and edge GeoPackage files into the respective tables
        node_layers = fiona.listlayers(node_gpkg)
        hex_layers = sorted(
            [layer for layer in node_layers if layer.startswith("hexes_r")],
            key=lambda x: int(x.split('_r')[1])
        )

        def insert_nodes(nodes, layer_name):
            try:
                c.executemany('''
                INSERT OR IGNORE INTO nodes (h3id, lat, lng, dist_m)
                VALUES (?, ?, ?, ?)
                ''', nodes)
                conn.commit()
            except Exception as e:
                logger.error(f"Error inserting nodes for layer {layer_name}: {e}")
                conn.rollback()

        for layer_name in hex_layers:
            logger.info(f"Processing node layer: {layer_name}")
            nodes_to_insert = []

            with fiona.open(node_gpkg, layer=layer_name) as src:
                with tqdm(total=len(src), desc=f"Inserting nodes from {layer_name}") as pbar:
                    for feature in src:
                        h3id = feature["properties"]["h3_id"]
                        lat, lng = h3.cell_to_latlng(h3id)
                        dist_m = int(feature["properties"]['dist_m'])
                        nodes_to_insert.append((h3id, lat, lng, dist_m))
                        pbar.update(1)

                        # Check if the batch size is reached
                        if len(nodes_to_insert) >= sqlite_batch_size:
                            insert_nodes(nodes_to_insert, layer_name)
                            nodes_to_insert.clear()
                            if testmode:
                                break

            # Insert any remaining items in the last, partial batch
            if nodes_to_insert:
                insert_nodes(nodes_to_insert, layer_name)
                nodes_to_insert.clear()

        edge_layers = fiona.listlayers(edge_gpkg)
        edge_layers = sorted(
            [layer for layer in edge_layers if layer.startswith("edges_r")],
            key=lambda x: int(x.split('_r')[1])
        )

        def insert_edges(edges, layer_name):
            try:
                c.executemany('''
                INSERT OR IGNORE INTO edges (source_h3id, target_h3id, midpoint_lat, midpoint_lng, length_m, dx, dy, angle)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', edges)
                conn.commit()
            except Exception as e:
                logger.error(f"Error inserting edges for layer {layer_name}: {e}")
                conn.rollback()

        for layer_name in edge_layers:
            logger.info(f"Processing edge layer: {layer_name}")
            edges_to_insert = []

            with fiona.open(edge_gpkg, layer=layer_name) as src:
                with tqdm(total=len(src), desc=f"Inserting edges from {layer_name}") as pbar:
                    for feature in src:
                        source = feature["properties"]["source"]
                        target = feature["properties"]["target"]
                        if source == target:
                            pbar.update(1)
                            continue
                        if source > target:
                            # Ensure edges are in canonical order (source < target)
                            source, target = target, source
                        source_latlng = h3.cell_to_latlng(source)
                        target_latlng = h3.cell_to_latlng(target)
                        midpoint_lat, midpoint_lng = geodetic_midpoint(source_latlng, target_latlng)
                        length_m = int(h3.great_circle_distance(source_latlng, target_latlng, unit='m'))
                        (dx, dy), angle = compute_direction_vector_and_angle(source_latlng, target_latlng)
                        edges_to_insert.append(
                            (source, target, midpoint_lat, midpoint_lng, length_m, dx, dy, int(angle)))
                        pbar.update(1)

                        # Check if the batch size is reached
                        if len(edges_to_insert) >= sqlite_batch_size:
                            insert_edges(edges_to_insert, layer_name)
                            edges_to_insert.clear()
                            if testmode:
                                break

            # Insert any remaining items in the last, partial batch
            if edges_to_insert:
                insert_edges(edges_to_insert, layer_name)
                edges_to_insert.clear()

    # Log sample data from the nodes and edges tables
    c.execute("SELECT * FROM nodes ORDER BY RANDOM() LIMIT 5")
    sample_nodes = c.fetchall()
    logger.info("Sample nodes data:")
    for row in sample_nodes:
        logger.info(dict(row))
    c.execute("SELECT * FROM edges ORDER BY RANDOM() LIMIT 5")
    sample_edges = c.fetchall()
    logger.info("Sample edges data:")
    for row in sample_edges:
        logger.info(dict(row))
    c.execute("SELECT * FROM edge_monthly ORDER BY RANDOM() LIMIT 15")
    sample_monthly = c.fetchall()
    logger.info("Sample edge_monthly data:")
    for row in sample_monthly:
        logger.info(dict(row))

    # Count and log total inserted nodes and edges
    c.execute("SELECT COUNT(*) FROM nodes")
    total_nodes = c.fetchone()[0]
    logger.info(f"Total nodes: {total_nodes}")

    c.execute("SELECT COUNT(*) FROM edges")
    total_edges = c.fetchone()[0]
    logger.info(f"Total edges: {total_edges}")

    return conn, c, total_nodes, total_edges


def get_batches(cursor, batch_size, type="node", samplelimit=None):
    if type == "node":
        fields = "h3id, lat, lng, dist_m"
    elif type == "edge":
        fields = "source_h3id, target_h3id, midpoint_lat, midpoint_lng, length_m, dx, dy, angle"
    if samplelimit is not None:
        # Randomly sample a fixed number of rows
        cursor.execute(f"SELECT {fields} FROM {type}s ORDER BY RANDOM() LIMIT {samplelimit}")
        rows = cursor.fetchall()
        for i in range(0, len(rows), batch_size):
            yield rows[i:i + batch_size]
    else:
        # Sequential read in batches
        cursor.execute(f"SELECT {fields} FROM {type}s")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield rows


def keyset_paged_query(cursor, table, key_columns, batch_size=1000, columns='*', where_clause=None, parameters=()):
    """
    Generator yielding rows from a table using keyset pagination.

    :param cursor: SQLite cursor
    :param table: Table name
    :param key_columns: Single column name or tuple/list of key column names
    :param batch_size: Number of rows per batch
    :param columns: Columns to select (default: '*')
    :param where_clause: Optional SQL WHERE clause (without 'WHERE')
    :param parameters: Parameters for the WHERE clause
    :yield: list of rows as sqlite.Row objects
    """
    if isinstance(key_columns, str):
        key_columns = [key_columns]

    last_seen = None
    base_query = f"SELECT {columns} FROM {table}"
    order_by = "ORDER BY " + ", ".join(key_columns)

    while True:
        conds = []
        params = list(parameters)

        if where_clause:
            conds.append(where_clause)

        if last_seen is not None:
            # Build tuple-wise comparison: (col1, col2) > (?, ?)
            cols_sql = f"({', '.join(key_columns)}) > ({', '.join(['?'] * len(key_columns))})"
            conds.append(cols_sql)
            params.extend(last_seen)

        where_sql = f"WHERE {' AND '.join(conds)}" if conds else ""
        sql = f"{base_query} {where_sql} {order_by} LIMIT {batch_size}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        if not rows:
            break

        yield rows
        last_seen = [rows[-1][col] for col in key_columns]


def float_to_sci_str(x, digits=6):
    # Used to reduce graphology file size by converting floats to scientific notation strings
    if x == 0 or x is None:
        return "0"
    if isinstance(x, int) and abs(x) < 10**digits:
        return str(x)

    exp = int(math.floor(math.log10(abs(x))))
    mantissa = round(x / 10**exp * 10**(digits - 1))
    return f"{mantissa}e{exp - (digits - 1)}"


def export_graphology_format(msgpack_gzip_path, cursor, batch_size=1000):
    """
    Export graph data in an ultra-lean positional array format to reduce file size by
    eliminating object keys. This format uses nested lists with fixed field order
    instead of dictionaries.

    Structure of the exported `graph` object (a list of two lists: nodes and edges):

    graph = [
        nodes,  # list of nodes
        edges   # list of edges
    ]

    Nodes:
        Each node is a list:
            [
                key (str),           # Node identifier (h3id)
                [
                    lat (str),       # Latitude in scientific notation string
                    lng (str),       # Longitude in scientific notation string
                    bathymetry (str),
                    clear_land (str),
                    dist_m (str)
                ]
            ]

    Edges:
        Each edge is a list:
            [
                key (str),          # Combined source_target string
                source (str),       # Source node id
                target (str),       # Target node id
                [
                    length_m (float),
                    dx (str),       # Scientific notation string
                    dy (str),       # Scientific notation string
                    angle (float)
                ],
                [
                    visibility_m (list[str]),   # 12 monthly values as sci strings
                    daylight_ratio (list[int]), # 12 monthly values as integers
                    forward (list[list]),       # 12 monthly lists of 4 values each:
                                                # [wind_angle (float), wind_mag (str),
                                                #  current_angle (float), current_mag (str)]
                    reverse (list[list])        # Same structure as forward
                ]
            ]

    Args:
        msgpack_gzip_path (str): Output path for the compressed msgpack file.
        cursor (sqlite3.Cursor): Database cursor for querying graph data.
        batch_size (int, optional): Number of records to fetch per batch.

    """
    logger.info("Assembling and exporting graph...")
    graph = [
        [],  # nodes list
        []   # edges list
    ]

    # Export nodes as [key, [lat, lng, bathymetry, clear_land, dist_m]]
    cursor.execute("SELECT COUNT(*) FROM nodes")
    total_nodes = cursor.fetchone()[0]

    with tqdm(total=total_nodes, desc="Nodes") as pbar:
        for batch in keyset_paged_query(cursor, table="nodes", key_columns="h3id", batch_size=batch_size):
            for node in batch:
                node_entry = [
                    node["h3id"],
                    [
                        float_to_sci_str(node["bathymetry"]),
                        float_to_sci_str(node["clear_land"]),
                        float_to_sci_str(node["dist_m"])
                    ]
                ]
                graph[0].append(node_entry)
            pbar.update(len(batch))

    # Export edges as [key, source, target, [length_m, dx, dy, angle], [visibility_m[], daylight_ratio[], forward[], reverse[]]]
    cursor.execute("SELECT COUNT(*) FROM edges")
    total_edges = cursor.fetchone()[0]

    with tqdm(total=total_edges, desc="Edges") as pbar:
        for batch in keyset_paged_query(cursor, table="edges", key_columns=("source_h3id", "target_h3id"),
                                        batch_size=batch_size):
            for edge in batch:
                source = edge["source_h3id"]
                target = edge["target_h3id"]
                cursor.execute(
                    f"SELECT * FROM edge_monthly WHERE source_h3id = '{source}' AND target_h3id = '{target}' ORDER BY month"
                )
                months = cursor.fetchall()
                month_data = [dict(row) for row in months]
                month_index = {m["month"]: m for m in month_data}
                default = {
                    "visibility_m": 0,
                    "daylight_ratio": 50,
                    "forward_wind_angle": 0,
                    "forward_wind_mag": 0,
                    "forward_current_angle": 0,
                    "forward_current_mag": 0,
                    "reverse_wind_angle": 0,
                    "reverse_wind_mag": 0,
                    "reverse_current_angle": 0,
                    "reverse_current_mag": 0,
                }

                complete_months = [
                    {**default, **month_index.get(m, {}), "month": m}
                    for m in range(1, 13)
                ]

                visibility_m = [float_to_sci_str(m["visibility_m"]) for m in complete_months]
                daylight_ratio = [m["daylight_ratio"] for m in complete_months]

                def reversed_angle(angle):
                    return (angle + 180) % 360

                # forward and reverse as list of arrays: [wind_angle, wind_mag, current_angle, current_mag]
                forward = [
                    [
                        reversed_angle(m["reverse_wind_angle"]) if m["forward_wind_mag"] == 0 else m["forward_wind_angle"],
                        float_to_sci_str(m["reverse_wind_mag"]) if m["forward_wind_mag"] == 0 else float_to_sci_str(m["forward_wind_mag"]),
                        reversed_angle(m["reverse_current_angle"]) if m["forward_current_mag"] == 0 else m["forward_current_angle"],
                        float_to_sci_str(m["reverse_current_mag"]) if m["forward_current_mag"] == 0 else float_to_sci_str(m["forward_current_mag"]),
                    ]
                    for m in complete_months
                ]
                reverse = [
                    [
                        m["reverse_wind_angle"],
                        float_to_sci_str(m["reverse_wind_mag"]),
                        m["reverse_current_angle"],
                        float_to_sci_str(m["reverse_current_mag"]),
                    ]
                    for m in complete_months
                ]

                edge_entry = [
                    f"{source}_{target}",
                    source,
                    target,
                    [
                        edge["length_m"],
                        edge["angle"]
                    ],
                    [
                        visibility_m,
                        daylight_ratio,
                        forward,
                        reverse
                    ]
                ]
                graph[1].append(edge_entry)
            pbar.update(len(batch))

    # === Write compressed MessagePack ===
    logger.info("Writing to %s...", msgpack_gzip_path)
    packed = msgpack.packb(graph, use_bin_type=True)
    with gzip.open(msgpack_gzip_path, "wb") as f:
        f.write(packed)

    # Copy to the docs directory
    docs_path = Path(str(msgpack_gzip_path).replace("/docs/", "/app/public/"))
    shutil.copy2(msgpack_gzip_path, docs_path)

    logger.info("✔ Graph export complete.")
    return len(graph[0]), len(graph[1])


def save_metadata(AOI, geo_output_directory, node_count, edge_count):
    lon_min, lat_min, lon_max, lat_max = AOI["bounds"]
    metadata_dict = {
        "name": AOI["name"],
        "bounds": {
            "west": lon_min,
            "south": lat_min,
            "east": lon_max,
            "north": lat_max
        },
        "h3_resolution": COASTAL_SEA_RESOLUTION,
        "node_count": node_count,
        "edge_count": edge_count,
        "sources": datasets
    }
    metadata_file = geo_output_directory / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata_dict, f, indent=4)

    docs_path = Path(str(metadata_file).replace("/docs/", "/app/public/"))
    shutil.copy2(metadata_file, docs_path)

    logger.info(f"✅ Saved metadata to {metadata_file}")


def main(batch_size=5000):
    max_workers = os.cpu_count() - 1
    logger.info(f"Using {max_workers} worker processes.")

    AOI = AOIS[0]  # Use Europe bounding box
    bbox = list(AOI["bounds"])
    logger.info(f"Using AOI: {AOI['name']} with bounds {bbox}")

    docs_directory = head_directory / "docs"
    geo_output_directory = docs_directory / "data" / AOI["name"]
    geo_output_directory.mkdir(parents=True, exist_ok=True)

    output_path = geo_output_directory / "routing_graph.msgpack.gz"
    if output_path.exists():
        logger.warning(f"✔ Output file {output_path} already exists. Cannot continue.")
        return

    node_gpkg = geo_output_directory / "graph.gpkg"
    edge_gpkg = geo_output_directory / "edges.gpkg"
    if not node_gpkg.exists() or not edge_gpkg.exists():
        logger.error(f"GeoPackage files {node_gpkg} or {edge_gpkg} do not exist. Please run the graph creation first.")
        return

    db_path = geo_output_directory / "graph.sqlite3"

    conn, c, total_nodes, total_edges = init_db(db_path, node_gpkg, edge_gpkg)

    logger.info("Database initialised successfully.")

    # cache_paths = {
    #     "bathymetry": str(copernicus_data_directory / "zarr" / "bathymetry.zarr"),
    #     "visibility_distance": str(copernicus_data_directory / "zarr" / "visibility_distance.zarr"),
    # }
    #
    # with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init, initargs=(cache_paths,)) as executor:
    #
    #     buffer = []
    #
    #     def update_from_buffer(buffer):
    #         node_update_data = []
    #
    #         for update_row in buffer:
    #             node_update_data.append(update_row)
    #
    #         conn.executemany(
    #             """
    #             UPDATE nodes
    #             SET bathymetry = ?, clear_land = ?
    #             WHERE h3id = ?
    #             """,
    #             node_update_data
    #         )
    #
    #         conn.commit()
    #
    #     with tqdm(total=total_nodes, desc="Processing nodes") as pbar:
    #         for node_batch in get_batches(c, batch_size, samplelimit=None):
    #             futures = {
    #                 executor.submit(process_node, (row["h3id"], row["lat"], row["lng"], row["dist_m"])): row["h3id"]
    #                 for row in node_batch
    #             }
    #
    #             for future in as_completed(futures):
    #                 result = future.result()
    #                 if result is not None:
    #                     buffer.append(result)
    #
    #                 pbar.update(1)
    #
    #                 if len(buffer) >= batch_size:
    #                     update_from_buffer(buffer)
    #                     buffer.clear()
    #
    #         if buffer:
    #             update_from_buffer(buffer)
    #             buffer.clear()
    #
    # cache_paths = {
    #     "wind": str(copernicus_data_directory / "wind_hourly_subset.zarr"),
    #     "current": str(copernicus_data_directory / "current_hourly_subset.zarr"),
    #     "weather": str(copernicus_data_directory / "zarr" / "weather.zarr"),
    #     "daylight": str(copernicus_data_directory / "zarr" / "daylight_ratios.zarr"),
    # }
    #
    # with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init, initargs=(cache_paths,)) as executor:
    #
    #     buffer = []
    #
    #     def update_from_buffer(buffer):
    #         edge_monthly_data = []
    #
    #         for monthly_rows in buffer:
    #             edge_monthly_data.extend(monthly_rows)
    #
    #         conn.executemany(
    #             """
    #             INSERT OR REPLACE INTO edge_monthly (
    #                 source_h3id, target_h3id, month, visibility_m, daylight_ratio,
    #                 forward_wind_angle, forward_wind_mag,
    #                 forward_current_angle, forward_current_mag,
    #                 reverse_wind_angle, reverse_wind_mag,
    #                 reverse_current_angle, reverse_current_mag
    #             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    #             """,
    #             edge_monthly_data
    #         )
    #
    #         conn.commit()
    #
    #     with tqdm(total=total_edges, desc="Processing edges") as pbar:
    #         for edge_batch in get_batches(c, batch_size, type="edge", samplelimit=None):
    #             futures = {
    #                 executor.submit(process_edge, (
    #                     row["source_h3id"], row["target_h3id"], row["midpoint_lat"], row["midpoint_lng"],
    #                     row["length_m"],
    #                     row["dx"], row["dy"], row["angle"])): f"{row["source_h3id"]}-{row["target_h3id"]}"
    #                 for row in edge_batch
    #             }
    #
    #             for future in as_completed(futures):
    #                 result = future.result()
    #                 if result is not None:
    #                     buffer.append(result)
    #
    #                 pbar.update(1)
    #
    #                 if len(buffer) >= batch_size:
    #                     update_from_buffer(buffer)
    #                     buffer.clear()
    #
    #         if buffer:
    #             update_from_buffer(buffer)
    #             buffer.clear()

    node_count, edge_count = export_graphology_format(output_path, c)

    save_metadata(AOI, geo_output_directory, node_count, edge_count)


if __name__ == "__main__":
    main()
