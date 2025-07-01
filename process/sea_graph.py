# sea_routes.py
import gzip
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import fiona
import geopandas as gpd
import h3
import shapely
from cltk.utils import file_exists
from pyproj import Geod
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.geo import shape, mapping
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from shapely.ops import unary_union, nearest_points
from shapely.strtree import STRtree
from tqdm import tqdm

from process.utils import buffer_geodesic, save_geopackage, filter_polygons

"""
Optimisation Strategy:

1. Acquire more detailed country geometries.
2. Use parallel processing to perform geodesic buffering of individual geometries rather than whole countries.
3. Buffer all geometries in advance and store in geopackage.
4. Use parallel processing to fetch all hexes in advance for each resolution step and store in files.
5. Use parallel processing to handle edge + length generation.
6. Once edges are generated, use parallel processing again to compute weights for default coastal zone. Consider whether differential bidirectional edges are needed.
7. Store both length and default weights in the graph.
8. In the browser, unpack into IndexedDB with default weights. Weight can later be adjusted based on user preferences.

"""

# === Configuration for Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / "geo"
output_gpkg = geo_output_directory / "sea_zones.gpkg"
graph_output_path = geo_output_directory / "sea_graph.json.gz"

# === PARAMETERS ===
# Area of Interest in WGS84 (EPSG:4326 - lon, lat)
AOI_BOUNDS_4326 = (-45.00, 25.00, 37.00,
                   72.00)  # lon_min (east coast of Greenland), lat_min (Canarias), lon_max (eastern Mediterranean), lat_max (north coast of Norway)

# UK Only for Testing
# AOI_BOUNDS_4326 = (-11.0, 49.5, 2.0, 61.0)

MINIMUM_BUFFER = 1000  # Minimum buffer from land (in metres)
BUFFER_STEP_FACTOR = 2  # Factor to increase buffer size for each resolution step
MAXIMUM_RESOLUTION = 9  # H3 resolution for land-adjacent zones

COASTAL_SAILING_DISTANCE = 5000  # Coastal sailing distance in metres
COASTAL_SAILING_DISTANCE_BUFFER = 1500  # Tolerance buffer for coastal sailing distance in metres
COASTAL_EDGE_WEIGHT_FACTOR = 0.5  # Weight for edges in coastal zones

geod = Geod(ellps="WGS84")  # Geodesic calculations using WGS84 ellipsoid


def get_countries():
    logger.info("Loading and processing country geometries...")

    # Check if AOI and countries files already exist
    if file_exists(geo_output_directory / "aoi.gpkg") and file_exists(geo_output_directory / "countries.gpkg"):
        AOI_POLYGON = gpd.read_file(geo_output_directory / "aoi.gpkg").geometry.iloc[0]
        countries_gdf = gpd.read_file(geo_output_directory / "countries.gpkg")
        logger.info("Loaded existing AOI and countries geometries.")
        return countries_gdf, AOI_POLYGON
    else:
        AOI_POLYGON = Polygon.from_bounds(*AOI_BOUNDS_4326)
        save_geopackage(AOI_POLYGON, "aoi")

        try:
            viabundus_water_gdf = gpd.read_file(geo_output_directory / "viabundus-2-water-1500-reduced.geojson")
            viabundus_water_gdf['geometry'] = viabundus_water_gdf.geometry.buffer(0)

            countries_gdf = gpd.read_file("./data/countries.geojson")

            # Filter intersecting geometries
            countries_gdf = countries_gdf[countries_gdf.geometry.intersects(AOI_POLYGON)]

            if countries_gdf.empty:
                logger.error("No country geometries intersect the AOI.")
                exit()

            # Clip to AOI
            countries_gdf["geometry"] = countries_gdf.intersection(AOI_POLYGON)

            # Drop invalid or empty geometries
            countries_gdf = countries_gdf[countries_gdf.is_valid & ~countries_gdf.is_empty]

        except Exception as e:
            logger.error(f"Error loading or processing countries.geojson: {e}")
            exit()

        if countries_gdf.empty:
            logger.error("No valid geometries remain after clipping.")
            exit()

        logger.info(f"Loaded {len(countries_gdf)} country geometries intersecting AOI.")

        water_union = viabundus_water_gdf.union_all()

        # Difference operation to subtract water from countries
        countries_gdf['geometry'] = countries_gdf.geometry.apply(lambda geom: geom.difference(water_union))

        # Drop empty or invalid after difference
        countries_gdf = countries_gdf[countries_gdf.is_valid & ~countries_gdf.is_empty]

        save_geopackage(countries_gdf, "countries")

        return countries_gdf, AOI_POLYGON


def generate_sea_zones(start_resolution=MAXIMUM_RESOLUTION, initial_buffer=MINIMUM_BUFFER):
    sea_zones = []
    resolution = start_resolution
    buffer = 0

    while resolution > 0:
        if resolution == start_resolution - 1:
            # For the first zone, use the initial buffer
            buffer = initial_buffer
        elif not resolution == start_resolution:
            # Double the average edge length of current resolution (i.e. previous zone's cells)
            edge_length = h3.average_hexagon_edge_length(resolution, unit='m')
            buffer += BUFFER_STEP_FACTOR * edge_length
            buffer = int(math.ceil(buffer / 1000.0)) * 1000

        sea_zones.append({"buffer": buffer, "resolution": resolution})
        resolution -= 1

    for zone in sea_zones:
        logger.info(f"Generated sea zone: Buffer={zone['buffer']}m, Resolution={zone['resolution']}")

    return list(reversed(sea_zones))


def cover_buffer_zone(zone_geom, resolution):
    """Cover a geometry with H3 cells at given resolution, filtering by centroid."""

    hex_cells = []
    hexes = []

    # If MultiPolygon or GeometryCollection, iterate over parts
    if zone_geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
        geoms = [geom for geom in zone_geom.geoms if geom.is_valid and not geom.is_empty]
    else:
        geoms = [zone_geom]

    for geom in geoms:
        if not geom.is_valid or geom.is_empty:
            continue
        try:
            h3_shape = h3.geo_to_h3shape(mapping(geom))
            cells = h3.h3shape_to_cells(h3_shape, resolution)
        except Exception as e:
            logger.warning(f"Failed to generate h3shape for a geometry: {e}")
            continue

        for cell in cells:
            boundary = shape(h3.cells_to_h3shape([cell]).__geo_interface__)
            if not boundary.is_valid or boundary.is_empty:
                continue
            if resolution == MAXIMUM_RESOLUTION or boundary.within(geom) or True:
                hex_cells.append(cell)
                hexes.append({
                    "h3_id": cell,
                    "geometry": boundary
                })

    return hex_cells, hexes


def geodesic_distance(p: Point, g: BaseGeometry) -> float:
    nearest, _ = nearest_points(g, p)
    _, _, dist = geod.inv(p.x, p.y, nearest.x, nearest.y)
    return dist


def get_edge(u, v, coastal_sailing_zone, land_spatial_index, country_geoms):
    try:
        u_centre_lat, u_centre_lon = h3.cell_to_latlng(u)
        v_centre_lat, v_centre_lon = h3.cell_to_latlng(v)

        length = h3.great_circle_distance((u_centre_lat, u_centre_lon), (v_centre_lat, v_centre_lon), unit='m')

        # Shapely LineString expects (lon, lat) order
        line = LineString([(u_centre_lon, u_centre_lat), (v_centre_lon, v_centre_lat)])

        # Create Shapely Point objects for the cell centers
        u_point = Point(u_centre_lon, u_centre_lat)
        v_point = Point(v_centre_lon, v_centre_lat)

        u_intersects = u_point.intersects(coastal_sailing_zone)
        v_intersects = v_point.intersects(coastal_sailing_zone)

        if land_spatial_index:
            edge_centroid = line.centroid
            nearby_land_indices = land_spatial_index.query(edge_centroid)
            nearby_land_geoms = [country_geoms.iloc[idx] for idx in nearby_land_indices]
            min_dist = min((geodesic_distance(edge_centroid, g) for g in nearby_land_geoms), default=MINIMUM_BUFFER)
            min_dist = min(min_dist, MINIMUM_BUFFER)
            # Apply a quadratic factor to the length based on the distance to land
            length *= (1 + ((MINIMUM_BUFFER - min_dist) / MINIMUM_BUFFER) ** 2)

        distance = int(length)
        if u_intersects or v_intersects:
            # If the edge intersects the coastal sailing zone, return a directionally-weighted distance
            weighted_distance = int(length * COASTAL_EDGE_WEIGHT_FACTOR)
            return (
                weighted_distance if u_intersects else distance,
                weighted_distance if v_intersects else distance,
                {"geometry": line},
                True
            )
        else:
            return (distance, distance, {"geometry": line}, False)

    except Exception as e:
        import traceback
        print(f"Exception in get_edge({u}, {v}): {e}")
        print(traceback.format_exc())
        exit(1)


def get_connected_components(graph):
    visited = set()
    components = []

    for node in graph:
        if node not in visited:
            stack = [node]
            component = set()
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                stack.extend(neighbour for neighbour, _ in graph[curr] if neighbour not in visited)
            components.append(component)
    return components


def remove_isolated_nodes(graph):
    logger.info("Identifying isolated nodes and their edges...")
    components = get_connected_components(graph)
    main_component = max(components, key=len)
    logger.info(f"Removing {len(graph) - len(main_component)} isolated nodes and their edges from the graph.")
    return {node: edges for node, edges in graph.items() if node in main_component}


def get_coastal_sailing_zone(land):
    logger.info("Computing coastal sailing zone...")
    outer = buffer_geodesic(land, COASTAL_SAILING_DISTANCE + COASTAL_SAILING_DISTANCE_BUFFER)
    inner = buffer_geodesic(land, COASTAL_SAILING_DISTANCE - COASTAL_SAILING_DISTANCE_BUFFER)
    return filter_polygons(outer.difference(inner))


def build_valid_spatial_index(gdf):
    valid_geoms = gdf.geometry[gdf.geometry.apply(lambda g: isinstance(g, BaseGeometry))].dropna()
    return STRtree(valid_geoms.tolist()), valid_geoms


def write_gpkg_layer(geometry_list, layer_name, feature_type_desc, crs="EPSG:4326", output_path=output_gpkg):
    if not geometry_list:
        return
    gdf = gpd.GeoDataFrame(geometry_list, geometry="geometry", crs=crs)
    gdf.to_file(output_path, driver="GPKG", layer=layer_name)
    logger.info(f"Wrote {len(gdf)} {feature_type_desc} to layer '{layer_name}'")


def save_graph(graph):
    # Save the graph to a JSON file
    with gzip.open(graph_output_path, "wt", encoding="utf-8") as f:
        json.dump(graph, f, separators=(",", ":"))  # compact JSON

    logger.info(f"Saved H3 cell graph to {graph_output_path}")


def get_zoned_hex_graph():
    """
    Create hexagonal zones based on the provided countries GeoDataFrame and AOI polygon.
    Returns a GeoDataFrame of hexagonal zones and a NetworkX graph of the sea zones.
    """

    countries_gdf, aoi_polygon = get_countries()

    sea_zones = generate_sea_zones()
    graph = defaultdict(list)
    land = unary_union(countries_gdf.geometry)

    coastal_sailing_zone = get_coastal_sailing_zone(land)

    land_buffered = []
    inverse_land = []
    zone_shape = []
    zone_hex_shape = []

    prev_periphery = set()

    existing_layers = fiona.listlayers(str(output_gpkg)) if output_gpkg.exists() else []
    if existing_layers:
        # Remove any which do not start with 'zone_shape_r'
        for layer in existing_layers:
            if not layer.startswith("zone_shape_r"):
                fiona.remove(str(output_gpkg), layer=layer, driver="GPKG")
                logger.info(f"Removed layer '{layer}' from {output_gpkg.name}")

    for i, zone in enumerate(sea_zones):

        logger.info(
            f"Processing sea zone {i + 1}/{len(sea_zones)}: Buffer={zone['buffer']}m, Resolution={zone['resolution']}")

        shape_layer_name = f"zone_shape_r{zone['resolution']}_b{zone['buffer'] // 1000}km"

        # Try to load pre-existing layer if available
        if shape_layer_name in existing_layers:
            shape_gdf = gpd.read_file(output_gpkg, layer=shape_layer_name)
            filtered_request_area = shape_gdf.geometry.union_all()
            logger.info(f"Loaded existing zone_shape from layer '{shape_layer_name}'")
        else:
            # buffer land
            land_buffered.append(buffer_geodesic(land, zone["buffer"]))
            logger.info(f"Buffered land by {zone['buffer']}m")

            # start from full AOI minus buffered land
            inverse_land.append(aoi_polygon.difference(land_buffered[i]))

            if i >= 2:
                earlier_hex_shapes = [zhs for zhs in zone_hex_shape[:i] if not zhs.is_empty]
                if not zone_hex_shape[i - 2].is_empty:
                    earlier_hex_shapes.append(buffer_geodesic(zone_hex_shape[i - 2],
                                                              h3.average_hexagon_edge_length(
                                                                  sea_zones[i - 2]["resolution"],
                                                                  unit='m') // BUFFER_STEP_FACTOR))
                if not inverse_land[i - 2].is_empty:
                    earlier_hex_shapes.append(inverse_land[i - 2])
                earlier_hex_union = unary_union(
                    earlier_hex_shapes) if earlier_hex_shapes else shapely.geometry.GeometryCollection()
                request_area = inverse_land[i].difference(inverse_land[i - 2]).difference(earlier_hex_union)
            elif i >= 1:
                request_area = inverse_land[i].difference(inverse_land[i - 1])
            else:
                request_area = inverse_land[i]

            # filter + save request zone
            filtered_request_area = filter_polygons(request_area)
            zone_shape.append(filtered_request_area)

            shape_gdf = gpd.GeoDataFrame(geometry=[filtered_request_area], crs=countries_gdf.crs)
            shape_gdf.to_file(output_gpkg, driver="GPKG", layer=shape_layer_name)
            logger.info(f"Wrote zone_shape to layer '{shape_layer_name}'")

        # Discourage routes in zone closest to land by weighting edges based on distance
        land_spatial_index, valid_geoms = build_valid_spatial_index(countries_gdf) if i == len(sea_zones) - 1 else (
            None, [])

        # compute hex coverage
        hex_cells, hexes = cover_buffer_zone(filtered_request_area, zone["resolution"])
        if hex_cells:
            h3_shape = shape(h3.cells_to_h3shape(hex_cells).__geo_interface__)
        else:
            h3_shape = shapely.geometry.GeometryCollection()

        if i < len(sea_zones) - 1:  # avoid last zone
            zone_hex_shape.append(h3_shape)

        current_periphery = set()
        edge_geometries = []
        coastal_edge_geometries = []
        for hex_cell in tqdm(hex_cells, desc="Processing hex cells"):
            neighbours = h3.grid_ring(hex_cell, 1)
            for neighbour in neighbours:
                if neighbour in hex_cells:
                    if hex_cell < neighbour:  # prevent duplicate processing
                        hex_cell_weight, neighbour_weight, edge_geometry, coastal = get_edge(hex_cell, neighbour,
                                                                                             coastal_sailing_zone,
                                                                                             land_spatial_index,
                                                                                             valid_geoms)
                        if hex_cell_weight == 0:
                            continue
                        if zone["resolution"] == MAXIMUM_RESOLUTION and edge_geometry["geometry"].intersects(land):
                            continue
                        graph[hex_cell].append([neighbour, neighbour_weight])
                        graph[neighbour].append([hex_cell, hex_cell_weight])
                        if coastal:
                            coastal_edge_geometries.append(edge_geometry)
                        else:
                            edge_geometries.append(edge_geometry)
                else:
                    current_periphery.add(hex_cell)
                    # Add edges between current periphery and previous periphery
                    if prev_periphery:
                        parent = h3.cell_to_parent(neighbour, zone["resolution"] - 1)
                        if parent in prev_periphery:
                            hex_cell_weight, parent_weight, edge_geometry, coastal = get_edge(hex_cell, parent,
                                                                                              coastal_sailing_zone,
                                                                                              land_spatial_index,
                                                                                              valid_geoms)
                            if hex_cell_weight == 0:
                                continue
                            graph[parent].append([hex_cell, hex_cell_weight])
                            graph[hex_cell].append([parent, parent_weight])
                            if coastal:
                                coastal_edge_geometries.append(edge_geometry)
                            else:
                                edge_geometries.append(edge_geometry)

        prev_periphery = current_periphery

        write_gpkg_layer(
            geometry_list=hexes,
            layer_name=f"sea_zone_r{zone['resolution']}_b{zone['buffer'] // 1000}km",
            feature_type_desc="features"
        )

        write_gpkg_layer(
            geometry_list=edge_geometries,
            layer_name=f"sea_zone_edges_r{zone['resolution']}_b{zone['buffer'] // 1000}km",
            feature_type_desc="edges"
        )

        write_gpkg_layer(
            geometry_list=coastal_edge_geometries,
            layer_name=(
                f"coastal_sailing_edges_r{zone['resolution']}_"
                f"d{COASTAL_SAILING_DISTANCE // 1000}km_"
                f"b{COASTAL_SAILING_DISTANCE_BUFFER // 1000}km"
            ),
            feature_type_desc="coastal sailing edges"
        )

        if i == len(sea_zones) - 1:  # Last zone
            graph = remove_isolated_nodes(graph)

        save_graph(graph)


def main():
    """Main function to run the sea zone and graph generation process."""
    get_zoned_hex_graph()


if __name__ == "__main__":
    main()
