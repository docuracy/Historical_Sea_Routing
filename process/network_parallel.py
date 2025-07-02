import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import folium
import geopandas as gpd
import h3
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points, unary_union
from tqdm import tqdm

from process.utils import polygons_from_gdf, fill_holes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / "geo"

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
COASTAL_ZONE = 5000  # 5 km buffer from land (in metres)

logger.info("Starting H3 graph generation...")

land_gdf = gpd.read_file(geo_output_directory / "countries.gpkg")
open_sea_gdf = gpd.read_file(geo_output_directory / "open_sea.gpkg")
coastal_zone_gdf = gpd.read_file(geo_output_directory / "coastal_zone.gpkg")

land_geom = unary_union(land_gdf.geometry)
sea_geoms = polygons_from_gdf(open_sea_gdf)

# H3 resolutions
coastal_res = 8
sea_res = 6


def parse_cells(cells, graph, apply_weight=False, island=None,
                current_geom_idx=None, total_geoms=None):  # Added arguments
    weights = {}
    edge_cells_collector = []

    latlng_lookup = ({c: h3.cell_to_latlng(c) for c in cells})

    geom_progress_str = ""
    if current_geom_idx is not None and total_geoms is not None:
        geom_progress_str = f" (Geom {current_geom_idx}/{total_geoms})"

    if apply_weight:
        def calculate_cell_weight(cell_data):
            cell, (lat, lon) = cell_data
            point = Point(lon, lat)
            nearest = nearest_points(point, island)[1]
            dist_m = point.distance(nearest) * 111_000
            cell_weight = 1 / (1 + (1000 / (dist_m + 10)))
            return cell, cell_weight

        with ThreadPoolExecutor(max_workers=8) as executor:
            cell_data_for_processing = list(latlng_lookup.items())
            futures = [executor.submit(calculate_cell_weight, item) for item in cell_data_for_processing]

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"  Calculating cell weights{geom_progress_str}",
                               leave=False, mininterval=0.1):
                cell, cell_weight = future.result()
                weights[cell] = cell_weight

    def process_single_cell(cell):
        current_latlng = latlng_lookup[cell]
        neighbours = [n for n in h3.grid_disk(cell, 1) if n in cells]

        cell_graph_entries = {}
        is_edge_cell = False

        for n in neighbours:
            neighbour_latlng = latlng_lookup[n]
            base_dist = h3.great_circle_distance(current_latlng, neighbour_latlng) * 1000
            if apply_weight:
                w = (weights[cell] + weights[n]) / 2
                cell_graph_entries[n] = int(base_dist * w)
            else:
                cell_graph_entries[n] = int(base_dist)

        if len(neighbours) < 7:
            if island is None:
                is_edge_cell = True
            else:
                if apply_weight and weights[cell] > 0.8:
                    is_edge_cell = True

        return cell, cell_graph_entries, is_edge_cell

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_cell, cell) for cell in cells]

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"  Building H3 graph{geom_progress_str}",
                           leave=False, mininterval=0.1):
            cell, cell_graph_entries, is_edge_cell = future.result()
            graph[cell] = cell_graph_entries
            if is_edge_cell:
                edge_cells_collector.append(cell)

    edge_coords = [(lng, lat) for lat, lng in (latlng_lookup[c] for c in edge_cells_collector)]
    return edge_cells_collector, edge_coords


cache_dir = geo_output_directory / "cache"
cache_dir.mkdir(exist_ok=True)

graph_path = cache_dir / "graph.pkl"
sea_cells_path = cache_dir / "sea_cells.pkl"
sea_edge_cells_path = cache_dir / "sea_edge_cells.pkl"
sea_edge_coords_path = cache_dir / "sea_edge_coords.pkl"
sea_edge_xy_path = cache_dir / "sea_edge_xy.pkl"


def save_cache():
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    with open(sea_cells_path, "wb") as f:
        pickle.dump(sea_cells, f)
    with open(sea_edge_cells_path, "wb") as f:
        pickle.dump(sea_edge_cells, f)
    with open(sea_edge_coords_path, "wb") as f:
        pickle.dump(sea_edge_coords, f)
    with open(sea_edge_xy_path, "wb") as f:
        pickle.dump(sea_edge_xy, f)


def load_cache():
    global graph, sea_edge_cells, sea_edge_coords, sea_edge_xy, sea_tree
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    with open(sea_cells_path, "rb") as f:
        sea_cells = pickle.load(f)
    with open(sea_edge_cells_path, "rb") as f:
        sea_edge_cells = pickle.load(f)
    with open(sea_edge_coords_path, "rb") as f:
        sea_edge_coords = pickle.load(f)
    with open(sea_edge_xy_path, "rb") as f:
        sea_edge_xy = pickle.load(f)

    sea_tree = cKDTree(np.array(sea_edge_xy))
    logger.info("Loaded cached graph and sea edge data")


if all(path.exists() for path in
       [graph_path, sea_cells_path, sea_edge_cells_path, sea_edge_coords_path, sea_edge_xy_path]):
    load_cache()
else:
    sea_cells = set()
    for sea_geom in sea_geoms:
        if sea_geom.is_empty:
            logger.warning("Skipping empty geometry in open sea area.")
            continue
        sea_h3_shape = h3.geo_to_h3shape(sea_geom)
        sea_cells.update(h3.h3shape_to_cells(sea_h3_shape, sea_res))
    logger.info(f"Found {len(sea_cells)} cells in open sea area.")

    graph = {}
    sea_edge_cells, sea_edge_coords = parse_cells(sea_cells, graph)
    sea_edge_xy = [transformer.transform(*coord) for coord in sea_edge_coords]
    sea_tree = cKDTree(np.array(sea_edge_xy))
    logger.info("Processed sea edge cells and created spatial index.")
    save_cache()

# Process Islands one by one (iterate coastal_zone_gdf)
coastal_geoms = polygons_from_gdf(coastal_zone_gdf)
logger.info(f"Found {len(coastal_geoms)} coastal zone geometries.")

MAX_DISTANCE = 12000

logger.info("Starting sequential processing of coastal geoms...")


def process_coastal_geom(coastal_geom, current_geom_idx, total_geoms): # Added arguments
    local_graph = {}
    local_edge_cells = set()

    try:
        filled_coastal_geom = fill_holes(coastal_geom)
        island_geom = land_geom.intersection(filled_coastal_geom)

        logger.info(f"Processing coastal geometry {current_geom_idx}/{total_geoms}...")
        coastal_h3_shape = h3.geo_to_h3shape(coastal_geom)
        coastal_cells = h3.h3shape_to_cells(coastal_h3_shape, coastal_res)
        # Pass geom_idx and total_geoms to parse_cells
        coastal_edge_cells, coastal_edge_coords = parse_cells(
            coastal_cells, local_graph, apply_weight=True, island=island_geom,
            current_geom_idx=current_geom_idx, total_geoms=total_geoms # Pass it here
        )
        local_edge_cells.update(coastal_edge_cells)

        if not coastal_edge_cells:
            return local_edge_cells, local_graph

        for coastal_edge_cell, coastal_edge_coord in zip(coastal_edge_cells, coastal_edge_coords):
            coastal_xy = transformer.transform(*coastal_edge_coord)
            distances, indices = sea_tree.query(coastal_xy, k=2)
            for dist_m, idx in zip(distances, indices):
                if dist_m <= MAX_DISTANCE:
                    sea_edge_cell = sea_edge_cells[idx]
                    local_graph.setdefault(coastal_edge_cell, {})[sea_edge_cell] = int(dist_m)
                    local_graph.setdefault(sea_edge_cell, {})[coastal_edge_cell] = int(dist_m)

        return local_edge_cells, local_graph

    except Exception as e:
        logger.error(f"Error processing geometry {current_geom_idx}: {e}")
        return set(), {}


all_coastal_edge_cells = set()
total_coastal_geoms = len(coastal_geoms) # Get total count once

for i, coastal_geom in enumerate(coastal_geoms, start=1):
    edge_cells, local_graph = process_coastal_geom(coastal_geom, i, total_coastal_geoms) # Pass count
    all_coastal_edge_cells.update(edge_cells)
    for k, v in local_graph.items():
        graph.setdefault(k, {}).update(v)

# Save graph
with open(f"{geo_output_directory}/weighted_h3_graph.json", "w") as f:
    json.dump(graph, f)


def h3_cell_to_polygon(cell):
    boundary = h3.cell_to_boundary(cell)
    return Polygon([(lng, lat) for lat, lng in boundary])  # ensure (x, y) order


# Build GeoDataFrames
node_records = []
for cell in all_coastal_edge_cells:  # sea_edge_cells: # graph.keys():
    node_records.append({
        "h3_id": cell,
        "geometry": h3_cell_to_polygon(cell)
    })
nodes_gdf = gpd.GeoDataFrame(node_records, crs="EPSG:4326")

edge_records = []
for src, targets in graph.items():
    src_point = Point(h3.cell_to_latlng(src)[::-1])
    for tgt, weight in targets.items():
        tgt_point = Point(h3.cell_to_latlng(tgt)[::-1])
        edge_records.append({
            "source": src,
            "target": tgt,
            "weight": weight,
            "geometry": LineString([src_point, tgt_point])
        })
edges_gdf = gpd.GeoDataFrame(edge_records, crs="EPSG:4326")

# Write to GPKG
gpkg_path = geo_output_directory / "h3_graph.gpkg"
nodes_gdf.to_file(gpkg_path, layer="h3_nodes", driver="GPKG")
edges_gdf.to_file(gpkg_path, layer="h3_edges", driver="GPKG")

logger.info(f"Saved H3 graph to {gpkg_path}")
