# sea_routes.py
import gzip
import json
import logging
import math
from pathlib import Path

import fiona
import geopandas as gpd
import h3
import shapely
from cltk.utils import file_exists
from pyproj import Geod
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.geo import shape, mapping, box
from shapely.geometry.linestring import LineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from shapely.validation import make_valid, explain_validity
from tqdm import tqdm

from process.config import AOIS
from process.utils import buffer_geodesic, filter_polygons, geometries_to_MultiPolygon, chunked_unary_union, \
    safe_difference

"""
Weighting Strategy:

- Separate module for applying weights to vanilla edges.
- Separate module for enriching hexes with penalty attributes. These should be computed based on extraction from a raster for each hex centre.
- Run solely with approximate coastal sight weighting initially - this should eventually be just one of multiple hex penalty attributes.
- Add global monthly average data from Copernicus Marine Service:
  https://data.marine.copernicus.eu/product/WIND_GLO_PHY_CLIMATE_L4_MY_012_003/description
    - Eastward wind + Northward wind (WIND)
  https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_WAV_001_027/description:
    - Sea floor depth (for masking non-navigable H3 cells in shallow water).
    - Sea surface wave stokes drift x/y velocity (VSDXY).
    - Sea surface wind wave mean period + Sea surface wind wave from direction + Sea surface wind wave significant height (WW)
    - Sea surface primary swell wave from direction (SW1)
    
- Use Parquet during data processing (buffer zones, SST lookup, H3 coverage). Good for:
    - Web-scale querying, analytics, ML preprocessing.
    - Parallel processing pipelines (e.g. H3 + SST).
    - Efficient storage of large datasets.
    - Partitionable: data can be split into multiple files/folders by resolution, buffer, etc.

"""

# === Configuration for Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
AOI = AOIS[1]

docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / AOI["name"]
geo_output_directory.mkdir(parents=True, exist_ok=True)

output_gpkg = geo_output_directory / "graph.gpkg"
buffer_gpkg = geo_output_directory / "land_buffered.gpkg"
graph_output_path = geo_output_directory / "sea_graph.json.gz"

viabundus_geojson = docs_directory / "data" / "Viabundus-2-water-1500.geojson"

# === PARAMETERS ===
AOI_BOUNDS_4326 = AOI["bounds"]

MINIMUM_BUFFER = 1000  # Minimum buffer from land (in metres)
BUFFER_STEP_FACTOR = 2  # Factor to increase buffer size for each resolution step
MAXIMUM_RESOLUTION = 9  # H3 resolution for land-adjacent zones

COASTAL_SAILING_DISTANCE = 30000  # Coastal sailing distance in metres
COASTAL_SAILING_DISTANCE_BUFFER = 5000  # Tolerance buffer for coastal sailing distance in metres
COASTAL_EDGE_WEIGHT_FACTOR = 0.5  # Weight for edges in coastal zones

COASTAL_LAND_DEM_BUFFER = 50000  # Buffer for coastal land DEM in metres
COASTAL_LAND_DEM_RESOLUTION = 5  # H3 resolution for coastal land DEM

geod = Geod(ellps="WGS84")  # Geodesic calculations using WGS84 ellipsoid


def get_land():
    logger.info("Loading and processing country geometries...")

    # Check if AOI and land files already exist
    if file_exists(str(buffer_gpkg)):
        AOI_POLYGON = gpd.read_file(buffer_gpkg, layer="aoi").geometry.iloc[0]
        land_gdf = gpd.read_file(buffer_gpkg, layer='land_buffered_0km')
        logger.info("Loaded existing AOI and land geometries.")
        return land_gdf, AOI_POLYGON
    else:
        AOI_POLYGON = Polygon.from_bounds(*AOI_BOUNDS_4326)
        gdf = gpd.GeoDataFrame(geometry=[AOI_POLYGON], crs="EPSG:4326")
        gdf.to_file(buffer_gpkg, layer='aoi', driver="GPKG")
        logger.info(f"Saved AOI polygon to {buffer_gpkg}")

        try:
            viabundus_water_gdf = gpd.read_file(viabundus_geojson)

            # Load the OSM land polygons shapefile
            land_gdf = gpd.read_file("./data/osm_land_4326/land_polygons.shp")

            # Ensure CRS is EPSG:4326
            if land_gdf.crs != "EPSG:4326":
                land_gdf = land_gdf.to_crs("EPSG:4326")

            # Filter geometries that intersect the AOI
            land_gdf = land_gdf[land_gdf.geometry.intersects(AOI_POLYGON)]

            if land_gdf.empty:
                logger.error("No OSM land geometries intersect the AOI.")
                exit()

            # Clip to AOI
            land_gdf["geometry"] = land_gdf.intersection(AOI_POLYGON)

            # Drop invalid or empty geometries
            land_gdf = land_gdf[land_gdf.is_valid & ~land_gdf.is_empty]

        except Exception as e:
            logger.error(f"Error loading or processing OSM land shapefile: {e}")
            exit()

        if land_gdf.empty:
            logger.error("No valid geometries remain after clipping.")
            exit()

        logger.info(f"Loaded {len(land_gdf)} land geometries intersecting AOI.")

        viabundus_water_gdf['geometry'] = viabundus_water_gdf.geometry.apply(make_valid)
        water_union = filter_polygons(viabundus_water_gdf.union_all())
        logger.info(f"Constructed water union geometry.")

        water_bounds = water_union.bounds

        def subtract_if_bbox_intersects(geom):
            if not isinstance(geom, (Polygon, MultiPolygon)) or geom.is_empty:
                return geom
            # Check if geometry's bounds intersect water_union's bounds
            gminx, gminy, gmaxx, gmaxy = geom.bounds
            wminx, wminy, wmaxx, wmaxy = water_bounds
            if gmaxx < wminx or gminx > wmaxx or gmaxy < wminy or gminy > wmaxy:
                return geom  # No bbox intersection, skip
            if geom.intersects(water_union):
                return geom.difference(water_union)
            return geom

        tqdm.pandas()
        land_gdf['geometry'] = land_gdf.geometry.progress_apply(subtract_if_bbox_intersects)

        logger.info("Masked land geometries by water bodies from Viabundus GeoJSON.")

        # Drop empty or invalid after difference
        land_gdf = land_gdf[land_gdf.is_valid & ~land_gdf.is_empty]

        if land_gdf.empty:
            logger.error("No valid land geometries remain after masking with water bodies.")
            exit()

        # Flatten land_gdf.geometry and save to GeoDataFrame
        flattened_land = chunked_unary_union(land_gdf.geometry)
        flattened_land_gdf = gpd.GeoDataFrame(geometry=[flattened_land], crs="EPSG:4326")
        flattened_land_gdf.to_file(buffer_gpkg, layer='land_flattened', driver='GPKG')

        unified_land_geometry = geometries_to_MultiPolygon(list(land_gdf.geometry))

        # Skip saving if still empty
        if unified_land_geometry.is_empty:
            logger.warning("Unified land geometry is empty. Not saving 'unified_land_gdf'.")
            exit()

        # Final empty check after validation
        if unified_land_geometry.is_empty:
            logger.warning("Unified land geometry is empty after make_valid. Not saving.")
            exit()

        if not unified_land_geometry.is_valid:
            logger.warning(
                f"Unified land geometry is invalid (overlapping polygons are expected and acceptable): {explain_validity(unified_land_geometry)}")

        # Save to file
        unified_land_gdf = gpd.GeoDataFrame(geometry=[unified_land_geometry], crs="EPSG:4326")
        unified_land_gdf.to_file(buffer_gpkg, layer='land_buffered_0km', driver='GPKG')
        logger.info(f"Saved unified land geometry to 'land_buffered_0km' with type: {unified_land_geometry.geom_type}")

        return unified_land_gdf, AOI_POLYGON


def generate_sea_zones(start_resolution=MAXIMUM_RESOLUTION, initial_buffer=MINIMUM_BUFFER):
    sea_zones = [
        {"buffer": 0, "resolution": start_resolution},
        {"buffer": initial_buffer, "resolution": start_resolution - 1},
    ]
    resolution = start_resolution - 2
    buffer = initial_buffer

    while resolution > -1:
        edge_length = h3.average_hexagon_edge_length(resolution, unit='m')
        buffer += BUFFER_STEP_FACTOR * edge_length
        buffer = int(math.ceil(buffer / 1000.0)) * 1000
        sea_zones.append({"buffer": buffer, "resolution": resolution})
        resolution -= 1

    for zone in sea_zones:
        logger.info(f"Generated sea zone: Buffer={zone['buffer']}m, Resolution={zone['resolution']}")

    return list(reversed(sea_zones))


def get_missing_buffers(aoi_polygon, sea_zones):
    all_buffer_values = [zone["buffer"] for zone in sea_zones] + \
                        [-COASTAL_LAND_DEM_BUFFER]
    unique_buffers = sorted(list(set(all_buffer_values)))  # Remove any duplicates and sort in ascending order
    logger.info(f"Unique buffer values: {unique_buffers}")

    existing_layers = fiona.listlayers(str(buffer_gpkg)) if buffer_gpkg.exists() else []

    for index, buffer_amount_m in enumerate(unique_buffers):
        layer_name = f"land_buffered_{buffer_amount_m // 1000}km"
        logger.info(f"Processing {layer_name}")

        if buffer_amount_m < 0:
            if f"land_buffered_{-buffer_amount_m // 1000}km_inverse" in existing_layers:
                logger.info(f"Removing inverse layer 'land_buffered_{-buffer_amount_m // 1000}km_inverse' from {buffer_gpkg}.")
                fiona.remove(str(buffer_gpkg), layer=f"land_buffered_{-buffer_amount_m // 1000}km_inverse", driver="GPKG")
        elif layer_name in existing_layers:
            logger.info(f"Layer '{layer_name}' already exists in {buffer_gpkg}. Skipping.")
            continue

        previous_buffer_amount_m = unique_buffers[index - 1] if index > 0 else 0
        if previous_buffer_amount_m == 0:
            previous_buffered_land = gpd.read_file(buffer_gpkg, layer="land_flattened").geometry.iloc[0]
            logger.info("Using original land geometry for 0m buffer.")
        else:
            previous_buffered_land = \
                gpd.read_file(buffer_gpkg, layer=f"land_buffered_{previous_buffer_amount_m // 1000}km").geometry.iloc[0]
            logger.info(f"Found previous buffered land for {previous_buffer_amount_m // 1000}km")

        simplify = 10 * math.pi * MINIMUM_BUFFER ** 2 if index == 0 else False
        logger.info(f"Simplification set to {simplify} for buffer {buffer_amount_m}m")

        buffered_land = buffer_geodesic(previous_buffered_land, buffer_amount_m - previous_buffer_amount_m, aoi_polygon,
                                        simplify)

        if buffered_land.is_empty:
            logger.warning(
                f"Buffered land is empty for buffer {buffer_amount_m}m (after cropping). Skipping layer '{layer_name}'.")
            continue  # Skip to next buffer if empty

        gdf = gpd.GeoDataFrame(geometry=[buffered_land], crs="EPSG:4326")
        try:
            gdf.to_file(buffer_gpkg, layer=layer_name, driver="GPKG")
            logger.info(f"'{layer_name}' exported to {buffer_gpkg}.")
        except Exception as e:
            logger.error(f"Error exporting '{layer_name}' to GeoPackage: {e}")


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


def get_coastal_sailing_zone():
    # Check if the coastal sailing zone already exists
    if file_exists(
            str(buffer_gpkg)) and f"coastal_sailing_zone_{COASTAL_SAILING_DISTANCE // 1000}km" in fiona.listlayers(
        str(buffer_gpkg)):
        logger.info(f"Coastal sailing zone already exists in {buffer_gpkg}. Skipping computation.")
        return \
            gpd.read_file(buffer_gpkg,
                          layer=f"coastal_sailing_zone_{COASTAL_SAILING_DISTANCE // 1000}km").geometry.iloc[0]
    logger.info("Computing coastal sailing zone...")
    outer_layer_name = f"land_buffered_{(COASTAL_SAILING_DISTANCE + COASTAL_SAILING_DISTANCE_BUFFER) // 1000}km"
    inner_layer_name = f"land_buffered_{(COASTAL_SAILING_DISTANCE - COASTAL_SAILING_DISTANCE_BUFFER) // 1000}km"
    outer = gpd.read_file(buffer_gpkg, layer=outer_layer_name).geometry.iloc[0]
    inner = gpd.read_file(buffer_gpkg, layer=inner_layer_name).geometry.iloc[0]
    coastal = filter_polygons(outer.difference(inner))
    # Save the coastal sailing zone to a layer in the GeoPackage
    coastal_gdf = gpd.GeoDataFrame(geometry=[coastal], crs="EPSG:4326")
    coastal_gdf.to_file(buffer_gpkg, driver="GPKG", layer=f"coastal_sailing_zone_{COASTAL_SAILING_DISTANCE // 1000}km")
    return coastal


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
    Create hexagonal zones based on the provided land GeoDataFrame and AOI polygon.
    Returns a GeoDataFrame of hexagonal zones and a NetworkX graph of the sea zones.
    """

    _, aoi_polygon = get_land()

    # repair_geometries_in_layer(buffer_gpkg, layer_name="land_buffered_0km")

    sea_zones = generate_sea_zones()

    def get_zone_by_index(index, inverse=False):
        buffer_value = sea_zones[index]["buffer"]
        layer_name = f"land_buffered_{buffer_value // 1000}km"
        if inverse:
            layer_name = f"{layer_name}_inverse"
        geometry = gpd.read_file(buffer_gpkg, layer=layer_name).geometry.iloc[0]
        return geometry

    get_missing_buffers(aoi_polygon, sea_zones)

    covered_area = MultiPolygon()

    existing_layers = fiona.listlayers(str(buffer_gpkg)) if buffer_gpkg.exists() else []
    existing_output = fiona.listlayers(str(output_gpkg)) if output_gpkg.exists() else []

    # if existing_layers: # Enable this block to remove old layers
    #     # Remove any which start with 'zone_shape_r'
    #     for layer in existing_layers:
    #         if layer.startswith("zone_shape_r"):
    #             fiona.remove(str(buffer_gpkg), layer=layer, driver="GPKG")
    #             logger.info(f"Removed layer '{layer}' from {buffer_gpkg.name}")
    # existing_layers = fiona.listlayers(str(buffer_gpkg)) if buffer_gpkg.exists() else []

    for i, zone in enumerate(sea_zones):

        logger.info(
            f"Processing sea zone {i + 1}/{len(sea_zones)}: Buffer={zone['buffer']}m, Resolution={zone['resolution']}")

        hex_layer_name = f"hexes_r{zone['resolution']}_b{zone['buffer'] // 1000}km"
        if hex_layer_name in existing_output:
            logger.info(f"Layer '{hex_layer_name}' already exists in {output_gpkg}. Skipping zone {i + 1}.")
            continue

        shape_layer_name = f"zone_shape_r{zone['resolution']}_b{zone['buffer'] // 1000}km"

        # Try to load pre-existing zone_shape layer if available
        if shape_layer_name in existing_layers:
            shape_gdf = gpd.read_file(buffer_gpkg, layer=shape_layer_name)
            request_area = shape_gdf.geometry.union_all()
            del shape_gdf
            logger.info(f"Loaded existing zone_shape from layer '{shape_layer_name}'")
        else:
            # 1. Get previous covered area
            if i > 0:
                previous_layer_name = f"covered_area_r{sea_zones[i - 1]['resolution']}_b{sea_zones[i - 1]['buffer'] // 1000}km"
                if previous_layer_name in existing_layers:
                    covered_area = gpd.read_file(buffer_gpkg, layer=previous_layer_name).geometry.iloc[0]
                    logger.info(f"Loaded previous covered area from layer '{previous_layer_name}'")
                else:
                    logger.warning(
                        f"Previous layer '{previous_layer_name}' not found. Starting with empty covered_area.")
                    covered_area = MultiPolygon()
            else:
                # For the first zone, start with an empty covered_area
                covered_area = MultiPolygon()

            # 2. Subtract the covered area from the AOI polygon to get uncovered_area, and save it (or load if it exists)
            uncovered_layer_name = f"uncovered_area_r{zone['resolution']}_b{zone['buffer'] // 1000}km"
            if uncovered_layer_name in existing_layers:
                uncovered_gdf = gpd.read_file(buffer_gpkg, layer=uncovered_layer_name)
                uncovered_area = uncovered_gdf.geometry.iloc[0]
                del uncovered_gdf
                logger.info(f"Loaded existing uncovered area from layer '{uncovered_layer_name}'")
            else:
                logger.info(f"Calculating uncovered area for zone {i + 1}...")
                # The following safe_difference may exhaust memory with use_parallel=True
                uncovered_area_raw = safe_difference(aoi_polygon, covered_area, attempt_fix=False, use_parallel=False)
                uncovered_area = filter_polygons(uncovered_area_raw)
                del uncovered_area_raw
                uncovered_gdf = gpd.GeoDataFrame(geometry=[uncovered_area], crs="EPSG:4326")
                uncovered_gdf.to_file(buffer_gpkg, driver="GPKG", layer=uncovered_layer_name)
                del uncovered_gdf
                logger.info(f"Wrote uncovered area for zone {i + 1} to layer {uncovered_layer_name}")

            # 3. Subtract the buffered land from the uncovered_area to get the request_area, and save it
            logger.info(f"Calculating zone_shape for zone {i + 1}...")
            if i < len(sea_zones) - 1:
                buffered_land = get_zone_by_index(i)
            else:
                buffered_land = gpd.read_file(buffer_gpkg, layer='land_flattened').geometry.iloc[0]
                logger.info(f"Using full flattened land for the last zone {i + 1}")
            # The following safe_difference may exhaust memory with use_parallel=True
            request_area_raw = safe_difference(uncovered_area, buffered_land, attempt_fix=False, use_parallel=False)
            request_area = filter_polygons(request_area_raw)
            del request_area_raw
            del buffered_land
            request_area_gdf = gpd.GeoDataFrame(geometry=[request_area], crs="EPSG:4326")
            request_area_gdf.to_file(buffer_gpkg, driver="GPKG", layer=shape_layer_name)
            del request_area_gdf
            logger.info(f"Wrote zone_shape for zone {i + 1} to layer '{shape_layer_name}'")

            # # Try to load pre-existing INVERSE layer if available
            # inverse_layer_name = f"land_buffered_{zone['buffer'] // 1000}km_inverse"
            # if inverse_layer_name in existing_layers:
            #     inverse_gdf = gpd.read_file(buffer_gpkg, layer=inverse_layer_name)
            #     inverse_land = inverse_gdf.geometry.iloc[0]
            #     logger.info(f"Loaded existing inverse land from layer '{inverse_layer_name}'")
            # else:
            #     # start from full AOI minus buffered land
            #     exit(f"Inverse layer '{inverse_layer_name}' not found. Please comment out this exit statement to ensure the inverse layer is created before running this script.")
            #     positive_land = get_zone_by_index(i) if i < len(sea_zones) - 1 else gpd.read_file(buffer_gpkg, layer='land_flattened').geometry.iloc[0]
            #     inverse_land = safe_difference(aoi_polygon, positive_land)
            #     inverse_gdf = gpd.GeoDataFrame(geometry=[inverse_land], crs="EPSG:4326")
            #     inverse_gdf.to_file(buffer_gpkg, driver="GPKG", layer=inverse_layer_name)
            #
            # request_area = safe_difference(inverse_land, covered_area, attempt_fix=False)
            # filtered_request_area = filter_polygons(request_area)
            # shape_gdf = gpd.GeoDataFrame(geometry=[filtered_request_area], crs="EPSG:4326")
            # shape_gdf.to_file(buffer_gpkg, driver="GPKG", layer=shape_layer_name)
            # logger.info(f"Wrote zone_shape to layer '{shape_layer_name}'")

        # compute hex coverage
        hex_cells = cover_buffer_zone(request_area, zone["resolution"])  # returns a set of hex cells
        del request_area
        if hex_cells:
            h3_shape = shape(h3.cells_to_h3shape(list(hex_cells)).__geo_interface__)
        else:
            h3_shape = shapely.geometry.GeometryCollection()

        if i < len(sea_zones) - 1:  # avoid last zone
            covered_area = chunked_unary_union([covered_area, h3_shape])
            buffer_value = h3.average_hexagon_edge_length(zone["resolution"] + 2, unit='m')
            covered_area = buffer_geodesic(covered_area, buffer_value, aoi_polygon, simplify=False)
            del h3_shape
            # Save the covered area to buffer_gpkg
            covered_gdf = gpd.GeoDataFrame(geometry=[covered_area], crs="EPSG:4326")
            covered_gdf.to_file(buffer_gpkg, driver="GPKG",
                                layer=f"covered_area_r{zone['resolution']}_b{zone['buffer'] // 1000}km")
            del covered_gdf
            logger.info(f"Saved covered area for zone {i + 1}/{len(sea_zones)}")

        hexes = []

        for cell in tqdm(list(hex_cells), desc="Processing hex cells"):
            neighbours = h3.grid_ring(cell, 1)
            is_periphery = any(neighbour not in hex_cells for neighbour in neighbours)

            boundary = shape(h3.cells_to_h3shape([cell]).__geo_interface__)
            if not boundary.is_valid or boundary.is_empty:
                continue

            hexes.append({
                "h3_id": cell,
                "geometry": boundary,
                "periphery": is_periphery
            })

        del hex_cells

        if hexes:
            gdf = gpd.GeoDataFrame(hexes, geometry="geometry", crs="EPSG:4326")
            gdf.to_file(output_gpkg, driver="GPKG", layer=hex_layer_name)
            del hexes
            logger.info(f"Wrote {len(gdf)} hexes to layer zone {i + 1}/{len(sea_zones)}: {hex_layer_name}")
            del gdf
        else:
            logger.warning(f"No valid hexes found for zone {i + 1}, skipping layer creation.")


def main():
    """Main function to run the sea zone and graph generation process."""
    get_zoned_hex_graph()


if __name__ == "__main__":
    main()
