import logging
import math
import pickle
import sys
import traceback
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import islice
from math import radians, sin, cos, atan2, sqrt
from multiprocessing import Pool, cpu_count
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio import features
from shapely import GEOSException
from shapely.geometry import LineString
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.geometry.geo import mapping, box
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.point import Point
from shapely.ops import split, snap
from shapely.ops import transform
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid
from skan import csr
from skimage.morphology import skeletonize
from tqdm import tqdm

# === Configuration for Logging ===
# Use a named logger in utilities to avoid interfering with root logger of main script
logger = logging.getLogger(__name__)
tqdm.pandas()

docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / "geo"


def tile_bounds(bounds, tile_size, padding=0.0):
    minx, miny, maxx, maxy = bounds
    x_steps = np.arange(minx, maxx, tile_size)
    y_steps = np.arange(miny, maxy, tile_size)
    tiles = []
    for x in x_steps:
        for y in y_steps:
            tiles.append(box(x - padding, y - padding, x + tile_size + padding, y + tile_size + padding))
    return tiles


def fix_geometry(geom: BaseGeometry, method: str = "auto") -> BaseGeometry:
    """
    Attempt to fix invalid geometries using the specified method.

    Parameters:
        geom (BaseGeometry): The input geometry.
        method (str): Fixing strategy: "buffer", "make_valid", or "auto".

    Returns:
        BaseGeometry: A valid geometry if possible, or the original.
    """
    if geom.is_valid:
        return geom

    try:
        if method == "buffer":
            fixed = geom.buffer(0)
        elif method == "make_valid":
            fixed = make_valid(geom)
        elif method == "auto":
            fixed = geom.buffer(0)
            if not fixed.is_valid:
                fixed = make_valid(geom)
        else:
            raise ValueError(f"Unknown fix method: {method}")

        return fixed if fixed.is_valid else geom

    except Exception as e:
        logger.warning(f"Failed to fix geometry: {e}")
        return geom


def _diff_tile(tile, a_geom, b_geom, a_prepared, b_prepared=None):
    try:
        if not a_prepared.intersects(tile):
            return None

        clipped_a = a_geom.intersection(tile)
        if clipped_a.is_empty:
            return None

        if b_prepared and not b_prepared.intersects(tile):
            return clipped_a  # b doesn't touch this tile

        clipped_b = b_geom.intersection(tile)
        diff = clipped_a.difference(clipped_b)
        return diff if not diff.is_empty else None

    except GEOSException:
        logger.warning("Tile difference failed; attempting to fix geometries")
        try:
            fixed_a = fix_geometry(clipped_a)
            fixed_b = fix_geometry(clipped_b)
            diff = fixed_a.difference(fixed_b)
            return diff if not diff.is_empty else None
        except Exception as e2:
            logger.error(f"Fixed tile difference also failed: {e2}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error in _diff_tile: {e}")
        return None


def tilewise_difference(a, b, tile_size=1.0, padding=0.1, use_parallel=False, union_chunk_size=100):
    tiles = tile_bounds(a.bounds, tile_size, padding)
    a_prepared = prep(a)
    b_prepared = prep(b)

    def chunked(iterable, size):
        it = iter(iterable)
        while (chunk := list(islice(it, size))):
            yield chunk

    def result_generator():
        if use_parallel:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(_diff_tile, tile, a, b, a_prepared, b_prepared): tile
                    for tile in tiles
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Tilewise difference (parallel)"):
                    result = future.result()
                    if result:
                        yield result
        else:
            for tile in tqdm(tiles, desc="Tilewise difference (serial)"):
                result = _diff_tile(tile, a, b, a_prepared, b_prepared)
                if result:
                    yield result

    geometries = []
    for chunk in chunked(result_generator(), union_chunk_size):
        geometries.append(unary_union(chunk))

    return chunked_unary_union(geometries) if geometries else GeometryCollection()


def tile_geometry(geom, tile_size_deg=1.0):
    """Yield clipped tiles of the input geometry."""
    minx, miny, maxx, maxy = geom.bounds

    for x0 in frange(minx, maxx, tile_size_deg):
        for y0 in frange(miny, maxy, tile_size_deg):
            tile = box(x0, y0, x0 + tile_size_deg, y0 + tile_size_deg)
            clipped = geom.intersection(tile)
            if not clipped.is_empty and clipped.is_valid:
                yield clipped


def frange(start, stop, step):
    """Range for floats"""
    while start < stop:
        yield start
        start += step


def cover_buffer_zone(zone_geom, resolution, tile_size_deg=1.0):
    hex_cells = set()

    if isinstance(zone_geom, gpd.GeoDataFrame):
        geoms = [g for g in zone_geom.geometry]
        logger.info(f"Found {len(geoms)} geometries in GeoDataFrame.")
    elif isinstance(zone_geom, gpd.GeoSeries):
        geoms = [g for g in zone_geom if g.is_valid and not g.is_empty]
        logger.info(f"Found {len(geoms)} geometries in GeoSeries.")
    elif zone_geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
        geoms = [g for g in zone_geom.geoms if g.is_valid and not g.is_empty]
    else:
        geoms = [zone_geom]

    # Tile the geometries to avoid memory issues with large polygons, and for speed
    for geom in geoms:
        try:
            minx, miny, maxx, maxy = geom.bounds
            est_tiles_x = int((maxx - minx) / tile_size_deg) + 1
            est_tiles_y = int((maxy - miny) / tile_size_deg) + 1
            est_total_tiles = est_tiles_x * est_tiles_y
        except Exception as e:
            logger.warning(f"Could not estimate tile count: {e}")
            est_total_tiles = None

        tile_iter = tile_geometry(geom, tile_size_deg=tile_size_deg)
        for tile_geom in tqdm(tile_iter, desc="Generating H3 cells from tiles", total=est_total_tiles):
            try:
                h3_shape = h3.geo_to_h3shape(mapping(tile_geom))
                cells = h3.h3shape_to_cells(h3_shape, resolution)
                hex_cells.update(cells)
            except Exception as e:
                logger.warning(f"Failed to generate H3 cells for tile: {e}")
                continue

    return hex_cells


def safe_difference(
        a,
        b,
        *,
        simplify_tolerance=0.0001,
        attempt_fix=True,
        tile_threshold_area=1.0,
        tile_size=1.0,
        tile_padding=0.1,
        use_parallel=True,
):
    """
    Compute a.difference(b) robustly.
    If geometry area exceeds `tile_threshold_area`, use tiling directly.
    """
    a_area = (a.bounds[2] - a.bounds[0]) * (a.bounds[3] - a.bounds[1])
    if a_area > tile_threshold_area:
        logger.info(f"Using tilewise difference (area={a_area:.2f} > threshold={tile_threshold_area})")
        return tilewise_difference(
            fix_geometry(a) if attempt_fix else a,
            fix_geometry(b) if attempt_fix else b,
            tile_size=tile_size,
            padding=tile_padding,
            use_parallel=use_parallel,
        )

    try:
        return a.difference(b)
    except GEOSException as e:
        logger.warning(f"Initial difference failed: {e}")

        if attempt_fix:
            logger.info("Attempting to clean geometries and retry")
            a_fixed = fix_geometry(a)
            b_fixed = fix_geometry(b)

            try:
                return a_fixed.difference(b_fixed)
            except GEOSException as e2:
                logger.warning(f"Fixed difference still failed: {e2}")

                try:
                    a_simpl = a_fixed.simplify(simplify_tolerance)
                    b_simpl = b_fixed.simplify(simplify_tolerance)
                    return a_simpl.difference(b_simpl)
                except GEOSException as e3:
                    logger.warning(f"Simplified difference also failed: {e3}")
                    logger.info("Falling back to tilewise difference (post-failure)")
                    return tilewise_difference(
                        a_fixed, b_fixed,
                        tile_size=tile_size,
                        padding=tile_padding,
                        use_parallel=use_parallel,
                    )

    return GeometryCollection()


def fix_geometry(geom):
    """Try to fix invalid geometry using buffer(0) and cleanup."""
    if geom.is_valid:
        return geom
    try:
        repaired = geom.buffer(0)
        if repaired.is_valid:
            return repaired
    except Exception as e:
        logger.warning(f"Buffer fix failed: {e}")
    return geom  # return as-is if fix fails


def repair_geometries_in_layer(gpkg_path: str, layer_name: str, tolerance: float = 0.0000001):
    """
    Load a GPKG layer, validate and repair each geometry individually,
    and overwrite the layer with corrected geometries (without merging).
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        logger.error(f"GPKG file does not exist: {gpkg_path}")
        return

    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
    except Exception as e:
        logger.error(f"Error reading layer '{layer_name}' from {gpkg_path}: {e}")
        return

    logger.info(f"Loaded {len(gdf)} geometries from layer '{layer_name}'")

    def clean_geometry(geom):
        if geom is None or geom.is_empty:
            return None
        try:
            if not geom.is_valid:
                geom = make_valid(geom)
            if not geom.is_valid:
                # Optional extra snap-to-self cleanup
                geom = snap(geom, geom, tolerance)
            return geom
        except GEOSException as e:
            logger.warning(f"GEOSException while cleaning geometry: {e}")
            return None

    gdf["geometry"] = gdf["geometry"].progress_apply(clean_geometry)
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid & ~gdf.geometry.is_empty]

    if gdf.empty:
        logger.warning(f"All geometries were invalid or empty after repair in layer '{layer_name}'")
        return

    try:
        gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
        logger.info(f"Saved repaired geometries to layer '{layer_name}' in {gpkg_path}")
    except Exception as e:
        logger.error(f"Failed to write repaired layer '{layer_name}': {e}")


# Helper function to buffer a single Polygon part
def _buffer_single_polygon_part(polygon: Polygon, radius_m: float, area_threshold, tolerance) -> BaseGeometry:
    """
    Buffers a single valid Polygon by the given radius (in metres) using an
    Azimuthal Equidistant projection centered on its centroid.
    This function is designed to be parallelized.
    """

    if not polygon.is_valid:
        polygon = make_valid(polygon)

    if not isinstance(polygon, Polygon) or not polygon.is_valid or polygon.is_empty:
        logger.warning(
            f"Polygon is not a valid Polygon or is empty after make_valid: {type(polygon)} - {polygon.geom_type if hasattr(polygon, 'geom_type') else 'No geom_type'}. Returning empty.")
        return Polygon()

    # If Polygon is a rectangle identical to its bounding box, return it directly
    if area_threshold and polygon.equals(
            polygon.envelope):  # No need to buffer on first iteration if it's a solid rectangle (sliced OSM shapes)
        return polygon

    centroid = polygon.centroid
    lon, lat = centroid.x, polygon.centroid.y

    aeqd_crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    wgs84_crs = CRS.from_epsg(4326)

    project_to_aeqd = Transformer.from_crs(wgs84_crs, aeqd_crs, always_xy=True).transform
    project_to_wgs84 = Transformer.from_crs(aeqd_crs, wgs84_crs, always_xy=True).transform

    try:
        pol_proj = transform(project_to_aeqd, polygon)

        if area_threshold and pol_proj.area > area_threshold:
            print("Simplifying polygon part to reduce complexity...", flush=True)
            pol_proj = pol_proj.simplify(tolerance, preserve_topology=True)
            print("Polygon part simplified.", flush=True)

        buffered_proj = pol_proj.buffer(radius_m)
        buffered_wgs84 = transform(project_to_wgs84, buffered_proj)

    except Exception as e:
        logger.warning(
            f"Error during transform/buffer for polygon part (centroid: {lon},{lat}) by {radius_m}m: {e}. Type of error: {type(e).__name__}")
        logger.error(traceback.format_exc())
        return Polygon()

    # Final validation of the buffered geometry
    if not buffered_wgs84.is_valid:
        buffered_wgs84 = make_valid(buffered_wgs84)
        if not buffered_wgs84.is_valid or buffered_wgs84.is_empty:
            logger.warning(
                f"Buffered geometry invalid after final make_valid for polygon part (centroid: {lon},{lat}). Returning empty.")
            return Polygon()

    return buffered_wgs84


def buffer_geodesic(pol: BaseGeometry, radius_m: float, aoi_polygon=None, simplify=False) -> BaseGeometry:
    """
    Buffers a geometry (Polygon, MultiPolygon, or GeometryCollection)
    by the given radius (in metres) using an Azimuthal Equidistant projection
    centered on the centroid of each individual Polygon part.
    Parallelizes buffering for MultiPolygons.
    """

    # # 1. Initial Validation and Normalization
    # if not pol.is_valid:
    #     pol = make_valid(pol)
    #     if not pol.is_valid:  # Still invalid after make_valid
    #         logger.warning(f"Input geometry remains invalid after make_valid: {pol.geom_type}")
    #         return pol.buffer(0)  # Attempt a 0-buffer to fix, or return empty
    #
    # # 2. Filter polygons (filter_polygons handles MultiPolygons too)
    # pol = filter_polygons(pol)
    #
    # if pol.is_empty:
    #     return pol

    # 3. Extract individual Polygons for buffering
    if isinstance(pol, Polygon):
        if pol.is_empty:
            logger.info("Input Polygon is empty. Returning empty MultiPolygon.")
            return MultiPolygon()
        polygons_to_buffer = [pol]
        count = 1
    elif isinstance(pol, MultiPolygon):
        # Avoid building a potentially-large list in memory
        count = len(pol.geoms)  # May be an overestimate if some are empty, but this is fine for parallel processing
        if count == 0:
            logger.info("No valid polygons found in MultiPolygon. Returning empty MultiPolygon.")
            return MultiPolygon()

        def polygon_generator():
            for p in pol.geoms:
                if isinstance(p, Polygon) and not p.is_empty:
                    yield p

        polygons_to_buffer = polygon_generator()
    else:
        logger.warning(f"Unsupported geometry type for geodesic buffering: {pol.geom_type}. Returning original.")
        return pol  # Return original if type is not handled

    if count == 0:
        logger.info("No valid polygons extracted for buffering. Returning empty MultiPolygon.")
        return MultiPolygon()

    # 4. Parallelize buffering of individual parts
    num_processes = max(1, cpu_count() - 1)
    logger.info(f"Geodesic-buffering {count} polygon parts by {radius_m}m with {num_processes} parallel processes...")
    sys.stdout.flush()

    area_threshold = int(10 * math.pi * simplify ** 2) if simplify else False
    tolerance = int(simplify / 10) if simplify else False

    # Use functools.partial to pass arguments to the worker function
    buffer_worker = partial(_buffer_single_polygon_part, radius_m=radius_m, area_threshold=area_threshold,
                            tolerance=tolerance)

    with Pool(processes=num_processes) as pool, tqdm(total=count, desc="Buffering polygons") as pbar:
        buffered_parts = []
        for result in pool.imap(buffer_worker, polygons_to_buffer, chunksize=1):
            clipped_result = result.intersection(aoi_polygon)
            if clipped_result and not clipped_result.is_empty:
                buffered_parts.append(result.intersection(aoi_polygon))
            pbar.update(1)
            sys.stdout.flush()

    if not buffered_parts:
        logger.info("All buffered parts are empty after parallel processing. Returning empty MultiPolygon.")
        return MultiPolygon()

    # 5. Union the results
    logger.info(f"Unioning {len(buffered_parts)} buffered parts.")
    union_result = chunked_unary_union(buffered_parts)

    # Final filtering and validation of the union result
    union_result = filter_polygons(union_result)
    if not union_result.is_valid:
        union_result = make_valid(union_result)
        if not union_result.is_valid:
            logger.warning(f"Union result remains invalid after make_valid: {union_result.geom_type}")
            union_result = union_result.buffer(0)
            if not union_result.is_valid:
                logger.warning("Union result is still invalid after 0-buffer. Returning empty.")
                return MultiPolygon()

    return union_result


def swap_geojson_coords_in_place(geojson_geom_dict):
    """
    Swaps the coordinate order (e.g., from [lon, lat] to [lat, lon] or vice-versa)
    for all coordinates within a GeoJSON Polygon or MultiPolygon dictionary.
    Modifies the dictionary in place.

    Args:
        geojson_geom_dict (dict): A GeoJSON dictionary representing a Polygon or MultiPolygon.

    Returns:
        dict: The modified GeoJSON dictionary with swapped coordinates.
    """
    geom_type = geojson_geom_dict.get('type')
    coordinates = geojson_geom_dict.get('coordinates')

    if not coordinates:
        return geojson_geom_dict  # Return as is if no coordinates

    if geom_type == 'Polygon':
        # A Polygon's coordinates are [[outer_ring], [hole_ring1], ...]
        # Each ring is a list of [lon, lat] or [lat, lon] pairs.
        new_rings = []
        for ring in coordinates:
            new_rings.append([[coord[1], coord[0]] for coord in ring])
        geojson_geom_dict['coordinates'] = new_rings
    elif geom_type == 'MultiPolygon':
        # A MultiPolygon's coordinates are [[[poly1_outer_ring], [poly1_hole_ring1], ...], [[poly2_outer_ring], ...]]
        new_polygons = []
        for poly_coords in coordinates:
            new_rings = []
            for ring in poly_coords:
                new_rings.append([[coord[1], coord[0]] for coord in ring])
            new_polygons.append(new_rings)
        geojson_geom_dict['coordinates'] = new_polygons
    else:
        # Handle other geometry types if necessary, or raise an error
        raise ValueError(
            f"Unsupported geometry type '{geom_type}' for coordinate swapping. Only Polygon and MultiPolygon are supported.")

    return geojson_geom_dict


def unique_edges(hex_cells):
    cellLatlng = {cell: h3.cell_to_latlng(cell) for cell in hex_cells}
    neighbours = {}
    for cell in hex_cells:
        neighbours[cell] = h3.grid_ring(cell, 1)
    edges = set()
    for cell, nbrs in neighbours.items():
        for nbr in nbrs:
            if cell < nbr and nbr in cellLatlng:  # Ensure each edge is only counted once
                edges.add((cell, nbr))
    # Generate Line for each edge (these will later be combined into a MultiLineString)
    lines = []
    lengths = []
    for cell1, cell2 in edges:
        latlng1 = cellLatlng[cell1]
        latlng2 = cellLatlng[cell2]
        lines.append([(latlng1[0], latlng1[1]), (latlng2[0], latlng2[1])])
        lengths.append(h3.great_circle_distance(latlng1, latlng2, unit='m'))
    # Compute average length of edges
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        print(f"Number of edges: {len(edges)}")
        print(f"Average edge length: {avg_length:.2f} metres")
    else:
        print("No edges found.")
    return {
        "type": "MultiLineString",
        "coordinates": lines,
        "lengths": lengths,
    }


def polygon_to_edges(shapely_polygon_input, resolution=6):
    """Convert a Shapely polygon to H3 hexagon points."""
    geojson_dict = mapping(shapely_polygon_input)
    h3_shape = h3.geo_to_h3shape(swap_geojson_coords_in_place(geojson_dict))
    hex_cells = h3.h3shape_to_cells(h3_shape, resolution)
    print(f"Number of hex cells: {len(hex_cells)}")
    return unique_edges(hex_cells)


# Helper for Haversine Distance (accurately calculates distance between lon/lat points)
def haversineDistance(coords1_lonlat, coords2_lonlat):
    """
    Calculates the Haversine distance between two [lon, lat] points in meters.
    """
    R = 6371e3  # metres (Earth's radius)

    lon1, lat1 = radians(coords1_lonlat[0]), radians(coords1_lonlat[1])
    lon2, lat2 = radians(coords2_lonlat[0]), radians(coords2_lonlat[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a));

    return R * c


def extract_valid_polygons(geometry):
    """Return a MultiPolygon made of all polygonal parts in the input geometry."""
    if geometry.geom_type == 'Polygon':
        return MultiPolygon([geometry])
    elif geometry.geom_type == 'MultiPolygon':
        return geometry
    elif geometry.geom_type == 'GeometryCollection':
        polys = [g for g in geometry.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        return MultiPolygon([p for g in polys for p in (g.geoms if isinstance(g, MultiPolygon) else [g])])
    else:
        return None


def fill_holes(geom):
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    elif isinstance(geom, MultiPolygon):
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    else:
        raise ValueError("Input geometry must be a Polygon or MultiPolygon")


def flip_coords(geom):
    if geom.is_empty:
        return geom
    return transform(lambda x, y: (y, x), geom)


def chunked_unary_union(geoms, chunk_size=1000, prefilter_polygons=True):

    geoms = list(geoms)
    filtered_geoms = []
    for geom in geoms:
        if geom.is_empty:
            continue
        if not isinstance(geom, (Polygon, MultiPolygon, GeometryCollection)):
            continue
        if prefilter_polygons:
            geom = filter_polygons(geom)
            if geom.is_empty:
                continue
        filtered_geoms.append(geom)

    unions = []
    for i in tqdm(range(0, len(filtered_geoms), chunk_size), desc="Unioning chunks"):
        chunk = filtered_geoms[i:i + chunk_size]
        unions.append(unary_union(chunk))
    if not unions:
        logger.warning("No valid geometries to union. Returning empty GeometryCollection.")
        return GeometryCollection()
    if len(unions) == 1:
        logger.info("Only one union found, returning it directly.")
        return unions[0]
    logger.info("Now combining all unions into a single geometry...")
    return unary_union(unions)


def geometries_to_MultiPolygon(geometries):
    """
    Convert a list of Shapely geometries to a MultiPolygon.
    Handles Polygons, MultiPolygons, and GeometryCollections.
    Returns an empty MultiPolygon if no valid polygonal geometries are found.
    """

    def extract_polygons(geom):
        if isinstance(geom, Polygon):
            return [geom]
        elif isinstance(geom, MultiPolygon):
            return list(geom.geoms)
        elif isinstance(geom, GeometryCollection):
            return [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        else:
            return []

    geometries = list(geometries)  # ensure it's iterable and safe to process

    polygons = []
    for geom in geometries:
        extracted = extract_polygons(geom)
        for g in extracted:
            if isinstance(g, Polygon):
                polygons.append(g)
            elif isinstance(g, MultiPolygon):
                polygons.extend(g.geoms)

    return MultiPolygon(polygons) if polygons else MultiPolygon()


def filter_polygons(geom):
    # Accept only Polygon or MultiPolygon geometries (including GeometryCollections containing them)
    if geom.is_empty:
        return geom  # empty geometry, keep as is
    if geom.geom_type in ['Polygon', 'MultiPolygon']:
        return geom
    if geom.geom_type == 'GeometryCollection':
        filtered = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
        if not filtered:
            return Polygon()  # empty polygon if none left
        elif len(filtered) == 1:
            return filtered[0]
        else:
            return MultiPolygon(filtered)
    # For any other geometry (Point, LineString, etc.) return empty polygon to exclude it
    return Polygon()


def polygons_from_gdf(gdf):
    """Convert GeoSeries geometries to a list of Polygons."""
    unioned = gdf.unary_union
    if isinstance(unioned, Polygon):
        return [unioned]
    elif isinstance(unioned, MultiPolygon):
        return list(unioned.geoms)
    else:
        return []


def save_geopackage(geom, name):
    """
    Save a Shapely geometry or GeoDataFrame to a GeoPackage file.
    """
    output_path = geo_output_directory / f"{name}.gpkg"

    if not isinstance(geom, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(
            {"name": [name]},
            geometry=[geom],
            crs="EPSG:4326"
        )
    else:
        gdf = geom.copy()
        if 'name' not in gdf.columns:
            gdf['name'] = name
        gdf = gdf.set_geometry('geometry', inplace=False)

    gdf.to_file(output_path, driver="GPKG", layer=name)
    logger.info(f"Saved {name} to {output_path}")


def save_shapely_to_geopackage_4326(geometry_3857, filename_stem, layer_name, geo_output_directory,
                                    transformer_3857_to_4326, properties=None, clip_to_aoi_3857=None):
    """
    Reprojects a Shapely geometry from EPSG:3857 to EPSG:4326 and saves it
    as a GeoPackage (.gpkg) file with the given layer name.

    Args:
        geometry_3857 (shapely.geometry.BaseGeometry): The geometry in EPSG:3857.
        filename_stem (str): Base name of the output .gpkg file (no extension).
        layer_name (str): Name of the layer within the GeoPackage.
        geo_output_directory (Path): Output directory.
        transformer_3857_to_4326 (pyproj.Transformer): Transformer for reprojection.
        properties (dict, optional): Attributes to store with the geometry.
        clip_to_aoi_3857 (shapely.geometry.BaseGeometry, optional): Optional AOI clipping.
    """
    if clip_to_aoi_3857 is not None:
        geometry_3857 = geometry_3857.intersection(clip_to_aoi_3857)

    geometry_3857 = extract_valid_polygons(geometry_3857)

    if geometry_3857 is None or geometry_3857.is_empty:
        logger.warning(f"Skipping export of empty geometry for {layer_name}.")
        return

    geometry_4326 = transform(transformer_3857_to_4326.transform, geometry_3857)

    if geometry_4326.geom_type == 'Polygon':
        geometry_4326 = gpd.GeoSeries([geometry_4326]).unary_union  # Promote to MultiPolygon if needed

    if properties is None:
        properties = {}

    if 'name' not in properties:
        properties['name'] = layer_name.replace('_', ' ').title()

    gdf = gpd.GeoDataFrame([properties], geometry=[geometry_4326], crs="EPSG:4326")

    output_path = geo_output_directory / f"{filename_stem}.gpkg"

    try:
        gdf.to_file(output_path, layer=layer_name, driver="GPKG")
        logger.info(f"'{layer_name}' exported to {output_path}.")
    except Exception as e:
        logger.error(f"Error exporting '{layer_name}' GeoPackage: {e}")


def resample_line(line, spacing):
    """
    Resample a LineString to points approximately every `spacing` units.
    Returns a new LineString with those points.
    """
    if line.length == 0:
        return line

    distances = np.arange(0, line.length, spacing)
    distances = np.append(distances, line.length)  # ensure last point included

    points = [line.interpolate(distance) for distance in distances]
    return LineString(points)


def simplify_outline(outline_geom, spacing=5000):
    """
    Accepts a LineString or MultiLineString, resamples each LineString to
    ~spacing point distance, returns a combined MultiLineString or LineString.
    """
    if isinstance(outline_geom, LineString):
        return resample_line(outline_geom, spacing)
    elif isinstance(outline_geom, MultiLineString):
        resampled_lines = [resample_line(line, spacing) for line in outline_geom.geoms]
        # If only one line after resampling, return LineString, else MultiLineString
        if len(resampled_lines) == 1:
            return resampled_lines[0]
        else:
            return MultiLineString(resampled_lines)
    else:
        raise ValueError("Outline geometry must be LineString or MultiLineString")


def cut_lines_at_intersection(lines, cutter_geom, cutter_polygon_union):
    """
    Split input lines at intersections with cutter_geom (boundary line),
    then keep parts of lines that do NOT intersect the cutter polygon (i.e. outside open sea area).

    Returns:
        result_lines (list of LineString): segments outside the cutter polygon
        intersection_points (list of (x, y) tuples): where lines intersect the cutter_geom
    """
    result_lines = []
    intersection_points = []

    for line in lines:
        # Find intersection points with cutter_geom
        intersection = line.intersection(cutter_geom)

        # Extract points from the intersection geometry
        if intersection.is_empty:
            pass
        elif intersection.geom_type == 'Point':
            intersection_points.append((intersection.x, intersection.y))
        elif intersection.geom_type == 'MultiPoint':
            intersection_points.extend([(pt.x, pt.y) for pt in intersection.geoms])
        elif intersection.geom_type == 'GeometryCollection':
            for geom in intersection.geoms:
                if isinstance(geom, Point):
                    intersection_points.append((geom.x, geom.y))

        # Now split the line and keep valid segments
        split_result = split(line, cutter_geom)
        for segment in split_result.geoms:
            if not segment.intersects(cutter_polygon_union):
                result_lines.append(segment)

    return result_lines, intersection_points


def skeleton_to_lines(skeleton, transform):
    skeleton_graph = csr.Skeleton(skeleton)
    lines = []

    for i, _ in enumerate(skeleton_graph.paths_list()):
        pixel_coords = skeleton_graph.path_coordinates(i)
        geo_coords = [transform * (x, y) for y, x in pixel_coords]
        if len(geo_coords) >= 2:
            lines.append(LineString(geo_coords))

    return lines


def split_large_polygon(poly, tile_size=500000):
    """
    Splits a large Shapely polygon into smaller pieces using a square tiling grid.

    Parameters:
        poly (Polygon or MultiPolygon): The input geometry in EPSG:3857.
        tile_size (float): Tile width/height in metres (default: 500 km).

    Returns:
        List[Polygon]: List of valid, non-empty Polygon objects.
    """
    minx, miny, maxx, maxy = poly.bounds
    tiles = []

    nx = math.ceil((maxx - minx) / tile_size)
    ny = math.ceil((maxy - miny) / tile_size)

    for i in range(nx):
        for j in range(ny):
            tile = box(
                minx + i * tile_size,
                miny + j * tile_size,
                minx + (i + 1) * tile_size,
                miny + (j + 1) * tile_size,
            )
            clipped = poly.intersection(tile)
            if not clipped.is_empty:
                if clipped.geom_type == "Polygon":
                    tiles.append(clipped)
                elif clipped.geom_type == "MultiPolygon":
                    tiles.extend(
                        g for g in clipped.geoms if g.is_valid and not g.is_empty
                    )

    return tiles


def skeletonise_polygon(geom, resolution=5, workers=4, split_threshold=1e10):
    """
    Skeletonise a Shapely Polygon or MultiPolygon, splitting large ones into grid tiles.
    """
    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = [g for g in geom.geoms if g.is_valid and not g.is_empty]
    else:
        raise ValueError("Expected Polygon or MultiPolygon")

    split_polygons = []
    for i, poly in enumerate(polygons):
        if poly.area > split_threshold:
            print(f"🪓 Splitting large polygon {i + 1} (area={poly.area:.0f}) into grid tiles...")
            split_polygons.extend(split_large_polygon(poly))
        else:
            split_polygons.append(poly)

    split_polygons.sort(key=lambda g: g.area)

    return parallel_skeletonise(split_polygons, resolution=resolution, workers=workers)


def skeletonise_single(pickled_geom_and_resolution_and_index):
    try:
        poly, resolution, idx = pickle.loads(pickled_geom_and_resolution_and_index)
        print(f"Processing polygon {idx} (area={poly.area:.2f})")

        if not poly.is_valid or poly.is_empty or poly.area < resolution ** 2:
            print(f"Skipping polygon {idx} due to invalidity or small area.")
            return idx, []

        minx, miny, maxx, maxy = poly.bounds
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)
        if width == 0 or height == 0:
            print(f"Skipping polygon {idx} due to zero width or height.")
            return idx, []

        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

        raster = features.rasterize(
            [(poly, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        skeleton = skeletonize(raster).astype(np.uint8)

        if skeleton.sum() == 0 or np.count_nonzero(skeleton) < 2:
            print(f"Skipping polygon {idx} due to empty skeleton.")
            return idx, []

        try:
            lines = skeleton_to_lines(skeleton, transform)

            # Ensure output is a list of LineStrings, even if one or iterable of wrong type
            if isinstance(lines, LineString) or not isinstance(lines, Iterable):
                lines = [lines]
            else:
                lines = list(lines)  # Ensure it's materialised and indexable

            # Sanity check: all elements should be LineStrings
            assert all(isinstance(l, LineString) for l in lines), \
                f"Output of polygon {idx} contains non-LineStrings: {lines}"

            print(f"Polygon {idx} processed successfully, got {len(lines)} lines.")
            return idx, lines

        except ValueError as e:
            print(f"⚠️ Skipping polygon {idx} due to skeleton conversion error: {e}")
            return idx, []

    except Exception as e:
        print(f"⚠️ Exception in worker for polygon {idx}: {e}")
        print(traceback.format_exc())
        return idx, []


def parallel_skeletonise(polygons, resolution=5, workers=4):
    print(f"Using {workers} parallel workers...")
    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, poly in enumerate(polygons, 1):
            data = pickle.dumps((poly, resolution, idx))
            futures.append(executor.submit(skeletonise_single, data))

        for idx, future in enumerate(as_completed(futures), 1):
            try:
                poly_idx, lines = future.result()

                # 🛡️ Defensive: ensure result is a list of LineStrings
                if isinstance(lines, LineString):
                    lines = [lines]
                elif not isinstance(lines, (list, tuple)):
                    lines = list(lines)  # in case it's a generator

                results.append((poly_idx, lines))
                print(f"Completed polygon {poly_idx}/{len(polygons)}")
            except Exception as e:
                print(f"⚠️ Exception retrieving result for polygon {idx}: {e}")

    # Sort by polygon index
    results.sort(key=lambda x: x[0])
    return [line for _, lines in results for line in lines]
