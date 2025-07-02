import gzip
import json
import logging
import math
import pprint
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from fiona import listlayers
from tqdm import tqdm

from process.config import AOIS, datasets
from process.sea_graph_v3 import COASTAL_SEA_RESOLUTION

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AOI = AOIS[1]
DRAUGHT_THRESHOLD = 4.0  # Vessel draught in metres
DARKNESS_PENALTY = 3.0  # Penalty multiplier for darkness
LAND_INVISIBILITY_MULTIPLIER = 3.0  # Penalty multiplier for land visibility
CURRENT_PENALTY = 2.0  # Penalty multiplier for current direction
WAVE_PENALTY = 1.2  # Penalty multiplier for wave direction
WIND_PENALTY = 1.5  # Penalty multiplier for wind direction

docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / AOI["name"]
geo_output_directory.mkdir(parents=True, exist_ok=True)

node_gpkg = geo_output_directory / "graph.gpkg"
edge_gpkg = geo_output_directory / "edges.gpkg"
visibility_parquet = geo_output_directory / "visibility_distance.parquet"
copernicus_parquet = geo_output_directory / "copernicus.parquet"
node_env_db = geo_output_directory / "node_env.sqlite3"


# def visibility_penalty(h2: dict, e2: dict) -> float:
#     h2_daylight_ratio = float(h2["daylight_ratio"] or 0.0)
#     darkness_factor = DARKNESS_PENALTY * (1.0 - h2_daylight_ratio)
#
#     h2_clear_land = float(h2["clear_land"] or 0.0)
#     e2_visibility_m = float(e2["visibility_m"] or 0.0)
#
#     if h2_clear_land == 0.0 or h2_clear_land > e2_visibility_m:
#         return darkness_factor * LAND_INVISIBILITY_MULTIPLIER
#
#     return darkness_factor


# def directional_penalty(dx: float, dy: float, vector: Tuple[float, float]) -> float:
#     """Vector alignment penalty; lower is better."""
#     vx, vy = vector
#
#     if vx is None or vy is None:
#         # If missing vector components, no penalty (neutral)
#         return 1.0
#
#     dot = dx * vx + dy * vy
#     norm_vec = np.hypot(vx, vy)
#     norm_dir = np.hypot(dx, dy)
#     if norm_vec == 0 or norm_dir == 0:
#         return 1.0
#     angle = np.arccos(np.clip(dot / (norm_vec * norm_dir), -1.0, 1.0))
#     return 1.0 + (angle / np.pi)  # 1 (aligned) to 2 (opposite)


def load_all_edge_layers() -> gpd.GeoDataFrame:
    layers = listlayers(str(edge_gpkg))
    dfs = []
    for layer in layers:
        gdf = gpd.read_file(edge_gpkg, layer=layer)
        dfs.append(gdf)
    all_edges = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs=dfs[0].crs)
    all_edges = all_edges[all_edges["source"] != all_edges["target"]].copy()  # Remove any self-loops
    return all_edges


def load_edges() -> pd.DataFrame:
    gdf = load_all_edge_layers()

    # Check for invalid or missing geometries before coordinate extraction
    invalid_geoms = gdf[gdf.geometry.is_empty | gdf.geometry.isna()]
    if not invalid_geoms.empty:
        logger.warning(f"{len(invalid_geoms)} edges have empty or missing geometries.")
        logger.debug(f"Sample invalid geometries:\n{invalid_geoms.head()}")

    def extract_latlng(geom, point_index: int):
        try:
            coord = geom.coords[point_index]
            return (coord[1], coord[0])  # (lat, lng)
        except Exception as e:
            logger.warning(f"Could not extract lat/lng from geometry {geom}: {e}")
            return (None, None)

    gdf["h1_latlng"] = gdf.geometry.apply(lambda geom: extract_latlng(geom, 0))
    gdf["h2_latlng"] = gdf.geometry.apply(lambda geom: extract_latlng(geom, -1))

    # Log any rows where lat/lng could not be extracted
    missing_coords = gdf[
        gdf["h1_latlng"].isna() | gdf["h2_latlng"].isna() |
        gdf["h1_latlng"].apply(lambda x: None in x) |
        gdf["h2_latlng"].apply(lambda x: None in x)
        ]
    if not missing_coords.empty:
        logger.warning(f"{len(missing_coords)} edges have missing h1/h2 latlng coordinates.")
        logger.debug(f"Sample bad rows:\n{missing_coords[['source', 'target', 'geometry']].head()}")
    else:
        logger.info("All edges have valid h1/h2 latlng coordinates.")

    # Compute great-circle distances where lat/lng are valid
    valid_rows = gdf[
        gdf["h1_latlng"].apply(lambda x: None not in x) &
        gdf["h2_latlng"].apply(lambda x: None not in x)
        ].copy()

    valid_rows["length_m"] = valid_rows.apply(
        lambda row: h3.great_circle_distance(row["h1_latlng"], row["h2_latlng"], unit='m'), axis=1
    )

    return pd.DataFrame(valid_rows[["source", "target", "h1_latlng", "h2_latlng", "length_m"]])


def safe_value(val):
    return None if pd.isna(val) else val


def direction_to_vector(deg):
    """Convert direction in degrees to a unit vector pointing in the 'to' direction."""
    if pd.isna(deg):
        return (None, None)
    rad = np.radians((deg + 180) % 360)
    return (np.sin(rad), np.cos(rad))


def compute_unit_direction(latlon1, latlon2):
    """Compute a unit vector pointing from (lat1, lon1) to (lat2, lon2)."""
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2

    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    Δλ = np.radians(lon2 - lon1)

    x = np.sin(Δλ) * np.cos(φ2)
    y = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(Δλ)

    θ_rad = np.arctan2(x, y)
    θ_deg = (np.degrees(θ_rad) + 360) % 360  # Normalize to [0, 360)

    return direction_to_vector(θ_deg)


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(node_env_db)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Create nodes table if not exists
    c.execute('''
    CREATE TABLE IF NOT EXISTS nodes (
        h3_id TEXT PRIMARY KEY,
        lat REAL,
        lng REAL,
        bathymetry REAL,
        daylight_ratio REAL,
        clear_land REAL
    )
    ''')

    # Create node_env table if not exists
    c.execute('''
    CREATE TABLE IF NOT EXISTS node_env (
        h3_id TEXT,
        month INTEGER,
        current_x REAL,
        current_y REAL,
        swell_x REAL,
        swell_y REAL,
        swell_height REAL,
        swell_period REAL,
        wave_x REAL,
        wave_y REAL,
        wave_height REAL,
        wave_period REAL,
        wind_x REAL,
        wind_y REAL,
        visibility_m REAL,
        PRIMARY KEY (h3_id, month)
    )
    ''')

    # Create edges table if not exists
    c.execute('''
    CREATE TABLE IF NOT EXISTS edges (
        source TEXT,
        target TEXT,
        length_m REAL,
        dx REAL,
        dy REAL,
        PRIMARY KEY (source, target)
    )
    ''')

    # Create edge costs table if not exists
    # c.execute('''
    #     CREATE TABLE IF NOT EXISTS edge_costs (
    #     source TEXT NOT NULL,
    #     target TEXT NOT NULL,
    #     month INTEGER NOT NULL,
    #     weight REAL NOT NULL,
    #     PRIMARY KEY (source, target, month)
    # )
    # ''')

    conn.commit()
    return conn, c


def log_random_edge_sample(conn: sqlite3.Connection, cursor: sqlite3.Cursor):
    cursor.execute("SELECT COUNT(*) FROM nodes")
    total_nodes = cursor.fetchone()[0]
    logger.info(f"Total nodes: {total_nodes}")

    cursor.execute("SELECT COUNT(*) FROM nodes WHERE lat IS NULL")
    missing_lat_count = cursor.fetchone()[0]
    logger.info(f"{missing_lat_count} nodes have no latitude.")

    # Select a random edge
    cursor.execute("SELECT source, target, length_m, dx, dy FROM edges ORDER BY RANDOM() LIMIT 1")
    row = cursor.fetchone()
    if not row:
        logger.warning("No edges found in the database.")
        conn.close()
        return

    source, target, length_m, dx, dy = row
    logger.info(f"Selected random edge: {source} → {target}: {length_m}m, dx={dx}, dy={dy}")

    # Generate a random month (1-12)
    random_month = random.randint(1, 12)

    for h3_id in (source, target):
        # Static node data
        cursor.execute("SELECT * FROM nodes WHERE h3_id = ?", (h3_id,))
        static_row = cursor.fetchone()
        if static_row:
            colnames = [desc[0] for desc in cursor.description]
            static_dict = dict(zip(colnames, static_row))
            logger.info(f"Static data for {h3_id}:\n{pprint.pformat(static_dict)}")
        else:
            logger.warning(f"No static data found for {h3_id}")
            continue

        cursor.execute("""
            SELECT * FROM node_env WHERE h3_id = ? AND month = ?
        """, (h3_id, random_month))
        env_row = cursor.fetchone()
        if env_row:
            colnames = [desc[0] for desc in cursor.description]
            env_dict = dict(zip(colnames, env_row))
            logger.info(f"Monthly data for {h3_id} (month {random_month}):\n{pprint.pformat(env_dict)}")
        else:
            logger.warning(f"No monthly data for {h3_id} in month {random_month}")

    # Log the computed edge cost for this edge and month if it exists
    # cursor.execute("""
    #     SELECT * FROM edge_costs WHERE source = ? AND target = ? AND month = ?
    # """, (source, target, random_month))
    # edge_cost_row = cursor.fetchone()
    # if edge_cost_row:
    #     colnames = [desc[0] for desc in cursor.description]
    #     edge_cost_dict = dict(zip(colnames, edge_cost_row))
    #     logger.info(
    #         f"Edge cost data for {source} → {target} in month {random_month}:\n{pprint.pformat(edge_cost_dict)}")
    # else:
    #     logger.warning(f"No edge cost found for {source} → {target} in month {random_month}")

    # Log the computed edge cost for this edge REVERSED and month if it exists
    # cursor.execute("""
    #     SELECT * FROM edge_costs WHERE source = ? AND target = ? AND month = ?
    # """, (target, source, random_month))
    # edge_cost_row = cursor.fetchone()
    # if edge_cost_row:
    #     colnames = [desc[0] for desc in cursor.description]
    #     edge_cost_dict = dict(zip(colnames, edge_cost_row))
    #     logger.info(
    #         f"Edge cost data for {source} → {target} in month {random_month}:\n{pprint.pformat(edge_cost_dict)}")
    # else:
    #     logger.warning(f"No edge cost found for {source} → {target} in month {random_month}")


def save_edges_to_db(conn: sqlite3.Connection, cursor: sqlite3.Cursor, edges_df: pd.DataFrame):
    """
    Save edges from DataFrame to SQLite edges table.
    """
    # Prepare insert statement
    sql = '''
    INSERT OR REPLACE INTO edges (source, target, length_m, dx, dy)
    VALUES (?, ?, ?, ?, ?)
    '''

    # Prepare list of tuples for batch insertion
    data_to_insert = [
        (row.source, row.target, row.length_m, row.dx, row.dy)
        for row in edges_df.itertuples()
    ]

    cursor.executemany(sql, data_to_insert)
    conn.commit()


def load_node_env(conn: sqlite3.Connection, cursor: sqlite3.Cursor, batch_size: int = 1000) -> None:
    logger.info(f"Node environment DB does not exist, creating new: {node_env_db}")

    logger.info("Loading edge data...")
    edges_df = load_edges()
    h3_ids = set(edges_df["source"]) | set(edges_df["target"])

    edges_df["dx"], edges_df["dy"] = zip(*edges_df.apply(
        lambda row: compute_unit_direction(row["h1_latlng"], row["h2_latlng"]),
        axis=1
    ))

    logger.info("Saving edges to SQLite DB...")
    save_edges_to_db(conn, cursor, edges_df)

    logger.info("Loading node data and merging with Copernicus + visibility...")

    # Load environmental data
    cop_df = pd.read_parquet(copernicus_parquet)
    vis_df = pd.read_parquet(visibility_parquet).rename(columns={"hex_id": "h3_id"})

    # Filter
    cop_df = cop_df[cop_df["h3_id"].isin(h3_ids)]
    vis_df = vis_df[vis_df["h3_id"].isin(h3_ids)]
    env_df = cop_df.merge(vis_df, on="h3_id", how="left")

    # Prepare lookup for lat/lng and distance
    h3_static_map = (
        pd.concat([
            edges_df[["source", "h1_latlng"]].rename(columns={"source": "h3_id", "h1_latlng": "latlng"}),
            edges_df[["target", "h2_latlng"]].rename(columns={"target": "h3_id", "h2_latlng": "latlng"})
        ])
        .drop_duplicates("h3_id")
        .set_index("h3_id")[["latlng"]]
        .to_dict("index")
    )
    logger.info(f"h3_static_map has entries for {len(h3_static_map)} / {len(h3_ids)} nodes")

    static_rows = []
    monthly_rows = []

    for idx, h3_id in enumerate(tqdm(sorted(h3_ids), desc="Building node_env")):
        monthly = env_df[env_df["h3_id"] == h3_id]
        if monthly.empty:
            continue

        static_data = {
            "h3_id": h3_id,
            "lat": None,
            "lng": None,
            "bathymetry": None,
            "daylight_ratio": None,
            "clear_land": None
        }

        h3_meta = h3_static_map.get(h3_id)
        if h3_meta:
            static_data["lat"] = h3_meta["latlng"][0]
            static_data["lng"] = h3_meta["latlng"][1]

        # Collect static bathymetry and daylight_ratio from first available monthly record
        for row in monthly.itertuples():
            if pd.isna(row.month):
                continue
            if static_data["bathymetry"] is None:
                static_data["bathymetry"] = safe_value(row.deptho)
            if static_data["daylight_ratio"] is None:
                static_data["daylight_ratio"] = safe_value(row.daylight_ratio)
            if static_data["clear_land"] is None:
                static_data["clear_land"] = safe_value(row.distance_to_visible_land_m)
            break  # Only need first valid record

        # Append static node data row
        static_rows.append(tuple(static_data.values()))

        # Append monthly environmental data rows (one per month)
        for row in monthly.itertuples():
            if pd.isna(row.month):
                continue

            current_x, current_y = safe_value(row.VSDX), safe_value(row.VSDY)
            swell_x, swell_y = direction_to_vector(row.VMDR_SW1)
            swell_period = safe_value(row.VTM01_SW1)
            swell_height = safe_value(row.VHM0_SW1)
            wave_x, wave_y = direction_to_vector(row.VMDR_WW)
            wave_period = safe_value(row.VTM01_WW)
            wave_height = safe_value(row.VHM0_WW)
            wind_x, wind_y = safe_value(row.eastward_wind), safe_value(row.northward_wind)
            visibility_m = safe_value(row.visibility_m)

            monthly_rows.append((
                h3_id,
                row.month,
                current_x,
                current_y,
                swell_x,
                swell_y,
                swell_height,
                swell_period,
                wave_x,
                wave_y,
                wave_height,
                wave_period,
                wind_x,
                wind_y,
                visibility_m
            ))

        if (idx + 1) % batch_size == 0 or (idx + 1) == len(h3_ids):
            cursor.executemany("""
                INSERT OR REPLACE INTO nodes (
                    h3_id, lat, lng, bathymetry, daylight_ratio, clear_land
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, static_rows)

            cursor.executemany("""
                INSERT OR REPLACE INTO node_env (
                    h3_id, month,
                    current_x, current_y,
                    swell_x, swell_y, swell_height, swell_period,
                    wave_x, wave_y, wave_height, wave_period,
                    wind_x, wind_y,
                    visibility_m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, monthly_rows)

            conn.commit()
            static_rows = []
            monthly_rows = []

    logger.info(f"Saved node_env to SQLite DB: {node_env_db}")

    log_random_edge_sample(conn, cursor)


# def compute_and_store_edge_costs(conn: sqlite3.Connection, cursor: sqlite3.Cursor, batch_size: int = 1000):
#     logger.info("Clearing existing edge costs...")
#     cursor.execute("DELETE FROM edge_costs")
#     conn.commit()
#
#     logger.info("Computing edge costs...")
#     cursor.execute("SELECT source, target, length_m, dx, dy FROM edges")
#     rows = cursor.fetchall()
#
#     batch = []
#     for i, (source, target, length_m, dx, dy) in enumerate(tqdm(rows, desc="Computing edge costs")):
#
#         # Load static node data for source and target
#         cursor.execute("SELECT * FROM nodes WHERE h3_id = ?", (source,))
#         static_source = cursor.fetchone()
#         cursor.execute("SELECT * FROM nodes WHERE h3_id = ?", (target,))
#         static_target = cursor.fetchone()
#
#         if not static_source or not static_target:
#             logger.warning(f"Missing static data for edge {source} → {target}, skipping.")
#             continue
#
#         # Load env data for source and target
#         cursor.execute("SELECT * FROM node_env WHERE h3_id = ?", (source,))
#         env_source = {row["month"]: row for row in cursor.fetchall()}
#         cursor.execute("SELECT * FROM node_env WHERE h3_id = ?", (target,))
#         env_target = {row["month"]: row for row in cursor.fetchall()}
#
#         common_months = set(env_source) & set(env_target)
#         if not common_months:
#             continue
#
#         for (h1, h2, env2, d_sign) in [
#             (static_source, static_target, env_target, 1),
#             (static_target, static_source, env_source, -1),
#         ]:
#             dx_signed, dy_signed = dx * d_sign, dy * d_sign
#
#             for month in common_months:
#                 e2 = env2[month]
#
#                 cost = length_m \
#                        * directional_penalty(dx_signed, dy_signed,
#                                              (e2["current_x"], e2["current_y"])) ** CURRENT_PENALTY \
#                        * directional_penalty(dx_signed, dy_signed, (e2["wind_x"], e2["wind_y"])) ** WIND_PENALTY \
#                        * directional_penalty(dx_signed, dy_signed, (e2["wave_x"], e2["wave_y"])) ** WAVE_PENALTY \
#                        * visibility_penalty(h2, e2)
#
#                 batch.append((h1["h3_id"], h2["h3_id"], month, cost))
#
#         if len(batch) >= batch_size:
#             cursor.executemany("""
#                 INSERT OR REPLACE INTO edge_costs (source, target, month, weight)
#                 VALUES (?, ?, ?, ?)
#             """, batch)
#             conn.commit()
#             batch.clear()
#
#     if batch:
#         cursor.executemany("""
#             INSERT OR REPLACE INTO edge_costs (source, target, month, weight)
#             VALUES (?, ?, ?, ?)
#         """, batch)
#         conn.commit()
#
#     logger.info("✔ Completed storing edge costs.")


def clean_number(x):
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
    if isinstance(x, (int, float)):
        return x
    return None


def iter_nodes_keyset(cursor, batch_size=1000):
    env_columns = [
        "current_x", "current_y", "swell_x", "swell_y",
        "swell_height", "swell_period", "wave_x", "wave_y",
        "wave_height", "wave_period", "wind_x", "wind_y", "visibility_m"
    ]
    node_columns = ["lat", "lng", "bathymetry", "daylight_ratio", "clear_land"]

    last_h3_id = None

    while True:
        if last_h3_id is None:
            cursor.execute(f"""
                SELECT h3_id, {', '.join(node_columns)}
                FROM nodes
                ORDER BY h3_id
                LIMIT {batch_size}
            """)
        else:
            cursor.execute(f"""
                SELECT h3_id, {', '.join(node_columns)}
                FROM nodes
                WHERE h3_id > ?
                ORDER BY h3_id
                LIMIT {batch_size}
            """, (last_h3_id,))

        node_rows = cursor.fetchall()
        if not node_rows:
            break

        node_ids = [row[0] for row in node_rows]

        # Fetch environment data for these nodes
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            cursor.execute(f"""
                SELECT h3_id, month, {', '.join(env_columns)}
                FROM node_env
                WHERE h3_id IN ({placeholders})
            """, node_ids)
            env_rows = cursor.fetchall()
        else:
            env_rows = []

        # Organize env data by h3_id and month
        env_map = defaultdict(dict)
        for h3_id, month, *env_vals in env_rows:
            env_map[h3_id][month] = dict(zip(env_columns, env_vals))

        for row in node_rows:
            h3_id = row[0]
            attr_vals = row[1:]
            attrs = dict(zip(node_columns, attr_vals))
            attrs = {k: clean_number(v) for k, v in attrs.items()}
            env_data = env_map.get(h3_id, {})
            attrs["env"] = {str(month): vals for month, vals in env_data.items()}

            yield {"key": h3_id, "attributes": attrs}
            last_h3_id = h3_id


def iter_edges_keyset(cursor, batch_size=1000):
    last_source = None
    last_target = None

    while True:
        if last_source is None:
            cursor.execute(f"""
                SELECT source, target, length_m, dx, dy
                FROM edges
                ORDER BY source, target
                LIMIT {batch_size}
            """)
        else:
            cursor.execute(f"""
                SELECT source, target, length_m, dx, dy
                FROM edges
                WHERE (source > ?)
                   OR (source = ? AND target > ?)
                ORDER BY source, target
                LIMIT {batch_size}
            """, (last_source, last_source, last_target))

        batch = cursor.fetchall()
        if not batch:
            break

        for source, target, length_m, dx, dy in batch:
            yield {
                "source": source,
                "target": target,
                "attributes": {
                    "length_m": clean_number(length_m),
                    "dx": clean_number(dx),
                    "dy": clean_number(dy),
                }
            }
            last_source, last_target = source, target


# def iter_edges_keyset(cursor, batch_size=1000):
#     last_source = None
#     last_target = None
#
#     while True:
#         if last_source is None:
#             cursor.execute(f"""
#                 SELECT ec.source, ec.target, ec.month, ec.weight,
#                        e.length_m, e.dx, e.dy
#                 FROM edge_costs ec
#                 JOIN edges e ON (e.source = ec.source AND e.target = ec.target) OR (e.source = ec.target AND e.target = ec.source)
#                 ORDER BY ec.source, ec.target
#                 LIMIT {batch_size}
#             """)
#         else:
#             cursor.execute(f"""
#                 SELECT ec.source, ec.target, ec.month, ec.weight,
#                        e.length_m, e.dx, e.dy
#                 FROM edge_costs ec
#                 JOIN edges e ON (e.source = ec.source AND e.target = ec.target) OR (e.source = ec.target AND e.target = ec.source)
#                 WHERE (ec.source > ?)
#                    OR (ec.source = ? AND ec.target > ?)
#                 ORDER BY ec.source, ec.target
#                 LIMIT {batch_size}
#             """, (last_source, last_source, last_target))
#
#         batch = cursor.fetchall()
#         if not batch:
#             break
#
#         # Group by (source, target)
#         grouped = defaultdict(lambda: {"w": [None] * 12})
#         for source, target, month, weight, length_m, dx, dy in batch:
#             key = (source, target)
#             g = grouped[key]
#             g["length_m"] = clean_number(length_m)
#             g["dx"] = clean_number(dx)
#             g["dy"] = clean_number(dy)
#             g["w"][month - 1] = clean_number(weight)
#
#         # Yield edges grouped by (source,target)
#         for (source, target), attrs in grouped.items():
#             yield {"source": source, "target": target, "attributes": attrs}
#             last_source, last_target = source, target


def build_and_export_graphs_from_db(cursor, batch_size=1000):
    logger.info("Streaming graph export...")
    gzip_path = geo_output_directory / "routing_graph.json.gz"

    with gzip.open(gzip_path, "wt", encoding="utf-8") as f:
        f.write("{\n")
        f.write('"attributes": {},\n')

        # Get total nodes count
        cursor.execute("SELECT COUNT(*) FROM nodes")
        total_nodes = cursor.fetchone()[0]

        # Write nodes with progress bar and total
        f.write('"nodes": [\n')
        first = True
        for node in tqdm(iter_nodes_keyset(cursor, batch_size), desc="Exporting nodes", unit="nodes",
                         total=total_nodes):
            if not first:
                f.write(",\n")
            else:
                first = False
            json.dump(node, f, separators=(",", ":"))
        f.write("\n],\n")
        print("Node example:", json.dumps(node, indent=2), flush=True)

        # Get total edges count
        cursor.execute("SELECT COUNT(*) FROM edges")
        total_edges = cursor.fetchone()[0] * 2  # Each edge is used twice (source→target and target→source)

        # Write edges with progress bar and total
        f.write('"edges": [\n')
        first = True
        for edge in tqdm(iter_edges_keyset(cursor, batch_size), desc="Exporting edges", unit="edges",
                         total=total_edges):
            if not first:
                f.write(",\n")
            else:
                first = False
            json.dump(edge, f, separators=(",", ":"))
        f.write("\n]\n")
        print("Edge example:", json.dumps(edge, indent=2), flush=True)

        f.write("}\n")

    logger.info(f"✔ Exported graph to {gzip_path}")


# def build_and_export_graphs_from_db(cursor: sqlite3.Cursor):
#     logger.info("Building unified graph for all months...")
#
#     # Query all monthly edge costs at once
#     cursor.execute("""
#         SELECT ec.source, ec.target, ec.month, ec.weight,
#                e.length_m,
#                n.bathymetry
#         FROM edge_costs ec
#         JOIN edges e ON
#                 (e.source = ec.source AND e.target = ec.target)
#                 OR
#                 (e.source = ec.target AND e.target = ec.source)
#         JOIN nodes n ON n.h3_id = ec.target
#     """)
#     rows = cursor.fetchall()
#
#     # Build graph dictionary structure
#     js_graph = defaultdict(dict)
#
#     for source, target, month, weight, length_m, bathymetry in rows:
#         if bathymetry is None:
#             bathymetry = 0
#
#         edge = js_graph[source].setdefault(target, {
#             "w": [None] * 12,
#             "l": int(length_m),
#             "b": int(bathymetry),
#         })
#
#         edge["w"][month - 1] = int(weight)
#
#     preview = dict(random.sample(sorted(js_graph.items()), k=min(2, len(js_graph))))
#     print("Graph preview:", json.dumps(preview, indent=2))
#
#     json_data = json.dumps(js_graph, separators=(",", ":"))  # Compact form (no indent)
#
#     gzip_path = geo_output_directory / "routing_graph.json.gz"
#     with gzip.open(gzip_path, "wt", encoding="utf-8") as f:
#         f.write(json_data)
#
#     logger.info(f"✔ Exported {gzip_path} ({len(js_graph)} nodes)")


def save_metadata():
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
        "sources": datasets
    }
    metadata_file = geo_output_directory / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata_dict, f, indent=4)
    logger.info(f"✅ Saved metadata to {metadata_file}")


def main():
    if node_env_db.exists():
        conn, cursor = init_db()
        log_random_edge_sample(conn, cursor)
    else:
        conn, cursor = init_db()
        load_node_env(conn, cursor)

    # compute_and_store_edge_costs(conn, cursor)
    build_and_export_graphs_from_db(cursor)

    conn.close()

    logger.info("✅ All graphs built and exported.")

    save_metadata()


if __name__ == "__main__":
    main()
