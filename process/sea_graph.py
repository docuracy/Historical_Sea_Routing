# sea_graph.py

import gc
import logging
import pickle
import re
import subprocess
from multiprocessing import Pool
from pathlib import Path

import fiona
import geopandas as gpd
import h3
import pandas as pd
from cltk.utils import file_exists
from shapely.geometry import Polygon
from shapely.geometry.geo import shape, mapping
from shapely.geometry.linestring import LineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.ops import nearest_points
from shapely.prepared import prep
from shapely.strtree import STRtree
from shapely.validation import make_valid, explain_validity
from shapely.wkb import dumps as wkb_dumps
from shapely.wkt import loads as wkt_loads
from tqdm import tqdm

from process.config import AOIS, COASTAL_SEA_RESOLUTION
from process.utils import filter_polygons, geometries_to_MultiPolygon, chunked_unary_union, safe_difference, \
    cover_buffer_zone

tqdm.pandas()

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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
AOI = AOIS[0]

docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / AOI["name"]
geo_output_directory.mkdir(parents=True, exist_ok=True)

output_gpkg = geo_output_directory / "graph.gpkg"
buffer_gpkg = geo_output_directory / "land_buffered.gpkg"
pickle_gpkg = geo_output_directory / "pickles.gpkg"
edge_gpkg = geo_output_directory / "edges.gpkg"
graph_output_path = geo_output_directory / "sea_graph.json.gz"

viabundus_geojson = docs_directory / "data" / "Viabundus-2-water-1500.geojson"

# === PARAMETERS ===

# COASTAL_SEA_RESOLUTION = 7  # H3 resolution for land-adjacent zones

COASTAL_LAND_DEM_RESOLUTION = 5  # H3 resolution for coastal land DEM
COASTAL_LAND_DEM_BUFFER = 50000 + h3.average_hexagon_edge_length(COASTAL_LAND_DEM_RESOLUTION,
                                                                 unit='m')  # Buffer for coastal land DEM in metres

aoi_polygon = Polygon  # This will be populated by get_land()

# === Global Spatial Trees (Populated in get_land) ===
land_geoms = list  # List of geometries from land_buffered_0km (Polygons)
prepared_land = list  # Prepared geometries for land_buffered_0km (Polygons)
land_tree = STRtree  # For land_buffered_0km (Polygons)
land_geom_index_map = dict  # Maps WKB hex dumps to indices in land_geoms
flatland_geoms = list  # List of geometries from land_flattened (Polygons)
prepared_flatland_list = list  # Prepared geometries for land_flattened (Polygons)
flatland_tree = STRtree  # For land_flattened (Polygons)
coastline_lines = gpd.GeoDataFrame  # For coastline boundaries (LineStrings)
coastline_tree = STRtree  # For land_flattened boundaries (LineStrings)


def build_trees(unified_land_gdf, flattened_land_gdf):
    global land_tree, flatland_tree, coastline_lines, coastline_tree
    global prepared_land, flatland_geoms, prepared_flatland_list, land_geoms, land_geom_index_map

    # Explode multi-geometries into single parts
    land_polygons = unified_land_gdf.explode(index_parts=False).reset_index(drop=True)
    flatland_parts = flattened_land_gdf.explode(index_parts=False).reset_index(drop=True)

    # Prepare land geometry tree
    land_geoms = list(land_polygons.geometry)
    prepared_land = [prep(g) for g in land_geoms]
    land_tree = STRtree(land_geoms)
    land_geom_index_map = {wkb_dumps(geom, hex=True): i for i, geom in enumerate(land_geoms)}

    # Prepare flatland geometry tree
    flatland_geoms = list(flatland_parts.geometry)
    prepared_flatland_list = [prep(g) for g in flatland_geoms]
    flatland_tree = STRtree(flatland_geoms)

    # Prepare coastlines (just boundaries of polygons)
    coastline_lines = [geom.boundary for geom in flatland_geoms]
    coastline_tree = STRtree(coastline_lines)

    logger.info("Spatial trees and prepared geometries built successfully.")


def cells_to_gdf(cells):
    """Convert H3 cells to a GeoDataFrame."""
    return gpd.GeoDataFrame(
        {'h3_id': list(cells),
         'centroid': [Point(lon, lat) for lat, lon in (h3.cell_to_latlng(cell) for cell in cells)]},
        geometry=[shape(h3.cells_to_h3shape([cell]).__geo_interface__) for cell in cells],
        crs="EPSG:4326"
    )


def get_land():
    logger.info("Loading and processing country geometries...")

    global aoi_polygon  # , flatland_gdf

    # Check if AOI and land files already exist
    if file_exists(str(buffer_gpkg)):
        aoi_polygon = gpd.read_file(buffer_gpkg, layer="aoi").geometry.iloc[0]
        land_gdf = gpd.read_file(buffer_gpkg, layer='land_buffered_0km')
        flatland_gdf = gpd.read_file(buffer_gpkg, layer='land_flattened')
        logger.info("Loaded existing AOI and land geometries.")
        build_trees(land_gdf, flatland_gdf)
        return
    else:
        aoi_polygon = Polygon.from_bounds(*AOI["bounds"])
        gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs="EPSG:4326")
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
            land_gdf = land_gdf[land_gdf.geometry.intersects(aoi_polygon)]

            if land_gdf.empty:
                logger.error("No OSM land geometries intersect the AOI.")
                exit()

            # Clip to AOI
            land_gdf["geometry"] = land_gdf.intersection(aoi_polygon)

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

        build_trees(unified_land_gdf, flattened_land_gdf)

        return


def trim_to_aoi():
    # Tidy up by removing hexes outside the AOI
    logger.info("Tidying up hexes by AOI intersection...")
    layers_in_gpkg = fiona.listlayers(output_gpkg)
    layer = 0
    started = False
    stop = False
    while layer <= COASTAL_SEA_RESOLUTION and not (started and stop):
        layer_name = f"hexes_r{layer}"
        if layer_name in layers_in_gpkg:
            hexes_gdf = gpd.read_file(output_gpkg, layer=layer_name)
            hexes_gdf = hexes_gdf[hexes_gdf.intersects(aoi_polygon)]
            if hexes_gdf.empty:
                subprocess.run(["ogrinfo", output_gpkg, "-sql", f"DROP TABLE {layer_name}"], check=True)
                logger.info(f"Removed non-intersecting layer {layer_name} from {output_gpkg}.")
                stop = started
            else:
                started = True
                hexes_gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")
                logger.info(f"Tidied up {layer_name} by AOI intersection.")
        else:
            logger.warning(f"Layer {layer_name} not found in {output_gpkg}.")
            stop = started
        layer += 1


def within_land(geom):
    """
    Check if the given geometry is entirely contained by land.
    Returns True if contained, False otherwise.
    """
    idx = flatland_tree.query(geom, predicate='within')
    return len(idx) > 0


def stats(geom, centroid, hex_diameter_m, get_closest=False):
    def point_dist_to_coast(pt):
        idx = coastline_tree.nearest(pt)
        nearest_line = coastline_lines[idx]
        nearest_pt = nearest_points(pt, nearest_line)[1]
        return h3.great_circle_distance((pt.y, pt.x), (nearest_pt.y, nearest_pt.x), unit='m')

    dist = point_dist_to_coast(centroid)

    if hex_diameter_m == 0:
        return True, dist, None

    intersects = any(prepared_land[i].intersects(geom) for i in land_tree.query(geom))

    closest = None
    if intersects and get_closest:
        overlapping = False
        for candidate in land_tree.query(geom):
            key = wkb_dumps(candidate, hex=True)
            idx = land_geom_index_map.get(key)
            if idx is not None:
                if prepared_land[idx].overlaps(geom):
                    overlapping = True
                    break
        if overlapping:  # but not entirely
            closest = 0.0
        else:
            coords = list(geom.exterior.coords)
            closest = min(point_dist_to_coast(Point(c)) for c in coords) if coords else None

    only_sea = ((closest is not None and closest > hex_diameter_m) or dist > hex_diameter_m) and not intersects

    return only_sea, dist, closest


def get_coastal_hexes(cells):
    logger.info("Generating coastal sea hexes...")
    littoral_area = shape(h3.cells_to_geo(cells))
    land = gpd.read_file(buffer_gpkg, layer='land_flattened').geometry.iloc[0]
    logger.info(f"Computing coastal sea area by subtracting land from littoral area...")
    coastal_sea_area = safe_difference(littoral_area, land, attempt_fix=False, use_parallel=False)

    del littoral_area
    del land
    gc.collect()

    coastal_sea_area = filter_polygons(coastal_sea_area)

    coastal_sea_gdf = cells_to_gdf(cover_buffer_zone(coastal_sea_area, COASTAL_SEA_RESOLUTION))

    del coastal_sea_area
    gc.collect()

    tqdm.pandas(desc=f"Computing stats for coastal (r{COASTAL_SEA_RESOLUTION}) hexes")

    results = list(coastal_sea_gdf.progress_apply(
        lambda row: stats(
            row.geometry, row.centroid,
            hex_diameter_m=0
        ), axis=1
    ))  # Only dist_m required here
    results_df = pd.DataFrame(results, columns=['sea_only', 'dist_m', 'closest_m'], index=coastal_sea_gdf.index)
    hexes_gdf = pd.concat([coastal_sea_gdf, results_df], axis=1)
    hexes_gdf.to_file(output_gpkg, layer=f"hexes_r{COASTAL_SEA_RESOLUTION}", driver="GPKG")
    logger.info(f"Saved {len(hexes_gdf)} coastal sea hexes to {output_gpkg} layer 'hexes_r{COASTAL_SEA_RESOLUTION}'.")

    # Pickle the cells for reload after interruption
    cells = set(hexes_gdf['h3_id'])
    with open(geo_output_directory / f"cells_r{COASTAL_SEA_RESOLUTION + 1}.pkl", "wb") as f:
        pickle.dump(cells, f)

    del results
    del results_df
    del hexes_gdf
    del cells
    gc.collect()


def draw_pickled_cells(pickle_dir: Path = geo_output_directory, gpkg_path: Path = pickle_gpkg):
    """
    Write the outlines of all hexes in each cells_rX.pkl file to a corresponding layer in the provided GeoPackage.

    :param pickle_dir: Directory containing `cells_rX.pkl` files.
    :param gpkg_path: Path to the output GeoPackage (e.g., pickle_gpkg).
    """
    logger.info(f"Processing pickle files in {pickle_dir} to extract hex outlines...")

    pkl_files = sorted(pickle_dir.glob("cells_r[0-9]*.pkl"))
    if not pkl_files:
        logger.warning("No matching pickle files found.")
        return

    for pkl_file in pkl_files:
        resolution_match = re.search(r"r(\d+)", pkl_file.name)
        if not resolution_match:
            logger.warning(f"Could not determine resolution from filename: {pkl_file.name}")
            continue

        resolution = resolution_match.group(1)
        layer_name = f"hex_outlines_r{resolution}"
        logger.info(f"Processing {pkl_file.name} → Layer: {layer_name}")

        try:
            with open(pkl_file, "rb") as f:
                h3_cells = pickle.load(f)

            if not h3_cells:
                logger.warning(f"No cells found in {pkl_file.name}")
                continue

            gdf = gpd.GeoDataFrame(
                {"h3_id": list(h3_cells)},
                geometry=[
                    shape(h3.cells_to_h3shape([cell]).__geo_interface__) for cell in h3_cells
                ],
                crs="EPSG:4326"
            )

            gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")
            logger.info(f"Wrote {len(gdf)} hex outlines to {layer_name}")

        except Exception as e:
            logger.error(f"Error processing {pkl_file.name}: {e}")


def get_pickled_cells(resolution=None):
    pattern = re.compile(r"cells_r(\d+)\.pkl")

    # Find all matching files and extract resolution values
    saved_files = list(geo_output_directory.glob("cells_r*.pkl"))
    resolutions = [
        (int(match.group(1)), file)
        for file in saved_files
        if (match := pattern.match(file.name))
    ]

    if resolution:
        # Filter for any specified resolution
        resolutions = [(res, file) for res, file in resolutions if res == resolution]
        if not resolutions:
            logger.warning(f"No pickled cells found for resolution r{resolution}.")
            return 0, set()

    # Get the one with the highest resolution
    if resolutions:
        max_res, max_res_file = max(resolutions, key=lambda x: x[0])

        with open(max_res_file, "rb") as f:
            cells = pickle.load(f)

        logger.info(f"Loaded {len(cells)} pickled cells for r{max_res}.")
        return max_res, cells
    else:
        return 0, set()


def get_zoned_hex_graph(resolution=None):
    """
    Create hexagonal zones based on the provided land GeoDataFrame and AOI polygon.
    Returns a GeoDataFrame of hexagonal zones and a NetworkX graph of the sea zones.
    """

    # Start by covering the AOI with a hexagonal grid: step up from r0 until hexes are found
    h3_aoi = h3.geo_to_h3shape(mapping(aoi_polygon))

    hex_resolution, cells = get_pickled_cells(resolution)

    if not cells:
        while hex_resolution < 15 and not cells:
            logger.info(f"No hexes found yet. Trying r{hex_resolution}...")
            cells.update(h3.h3shape_to_cells(h3_aoi, hex_resolution))
            hex_resolution += 1

        hex_resolution -= 1
        logger.info(f"Found {len(cells)} hexes at r{hex_resolution}.")

        # Add neighbours to ensure full AOI coverage
        for cell in list(cells):
            cells.update(h3.grid_ring(cell, 1))

        logger.info(f"Expanded to {len(cells)} hexes with neighbours.")

    tqdm.pandas(desc=f"Computing stats for r{hex_resolution} hexes")
    while cells and hex_resolution < COASTAL_SEA_RESOLUTION:  # Different treatment for coastal sea

        if resolution is not None and hex_resolution != resolution:
            logger.info(f"Skipping resolution r{hex_resolution} as it does not match requested r{resolution}.")
            hex_resolution += 1
            continue

        layer_name = f"hexes_r{hex_resolution}"

        # Make penultimate band wider
        bandwidth_multiplier = 1.5 if hex_resolution == COASTAL_SEA_RESOLUTION - 1 else 2

        hexes_gdf = cells_to_gdf(cells)

        results = list(hexes_gdf.progress_apply(
            lambda row: stats(
                row.geometry, row.centroid,
                hex_diameter_m=h3.average_hexagon_edge_length(hex_resolution, unit='m') * bandwidth_multiplier,
                get_closest=hex_resolution <= COASTAL_LAND_DEM_RESOLUTION
            ), axis=1
        ))
        results_df = pd.DataFrame(results, columns=['sea_only', 'dist_m', 'closest_m'], index=hexes_gdf.index)
        hexes_gdf = pd.concat([hexes_gdf, results_df], axis=1)

        if hexes_gdf['sea_only'].sum() > 0:
            hexes_gdf[hexes_gdf['sea_only']].to_file(output_gpkg, layer=layer_name, driver="GPKG")
            logger.info(f"Saved hexes to {output_gpkg} layer '{layer_name}'.")
        else:
            logger.info(f"No entirely-sea hexes found yet.")

        if hex_resolution == COASTAL_LAND_DEM_RESOLUTION:
            land_dem_mask = (
                (~hexes_gdf['sea_only'] & (hexes_gdf['closest_m'] <= COASTAL_LAND_DEM_BUFFER))
            )
            hexes_gdf[land_dem_mask].to_file(output_gpkg, layer=f'land_hexes_r{COASTAL_LAND_DEM_RESOLUTION}',
                                             driver="GPKG")
            logger.info(f"Saved land hexes to {output_gpkg} layer 'land_hexes_r{COASTAL_LAND_DEM_RESOLUTION}'.")

            land_dem_gdf = hexes_gdf.loc[land_dem_mask]

            land_only_ids = set(
                hexes_gdf.loc[~hexes_gdf['sea_only'] & (hexes_gdf['closest_m'] >= 0), 'h3_id']
            )

            littoral_ids = set(
                land_dem_gdf.loc[~land_dem_gdf['geometry'].apply(within_land), 'h3_id']
            )

            # We want the original hexes, minus the sea-only hexes, minus the land hexes, plus the littoral hexes
            intersecting_ids = (set(hexes_gdf.loc[~hexes_gdf['sea_only'], 'h3_id']) - land_only_ids) | littoral_ids

            del land_dem_gdf

        else:
            intersecting_ids = set(hexes_gdf.loc[~hexes_gdf['sea_only'], 'h3_id'])

        del hexes_gdf
        gc.collect()

        hex_resolution += 1

        # Get all of the children of the remaining land-intersecting hexes
        logger.info(f"Finding children of {len(intersecting_ids)} hexes at r{hex_resolution}...")
        with Pool() as pool:
            children = pool.starmap(h3.cell_to_children, [(cell, hex_resolution) for cell in intersecting_ids])
        cells = set().union(*children)
        logger.info(f"Found {len(cells)} hexes for r{hex_resolution}.")

        del pool
        del intersecting_ids
        gc.collect()

        # Pickle the cells for reload after interruption
        with open(geo_output_directory / f"cells_r{hex_resolution}.pkl", "wb") as f:
            pickle.dump(cells, f)
        logger.info(f"Pickled {len(cells)} cells for r{hex_resolution} to 'cells_r{hex_resolution}.pkl'.")

    if resolution is not None and hex_resolution != resolution:
        logger.info(f"Skipping resolution r{hex_resolution} as it does not match requested r{resolution}.")
    elif hex_resolution == COASTAL_SEA_RESOLUTION:
        get_coastal_hexes(cells)
    else:
        logger.info(f"Nothing more to compute.")


def get_edges():
    logger.info("Generating edges for hexagonal zones...")

    layers_in_gpkg = fiona.listlayers(output_gpkg)
    layers_in_gpkg = sorted(
        [layer for layer in layers_in_gpkg if layer.startswith("hexes_r")],
        key=lambda x: int(x.split('_r')[1])
    )

    existing_edge_layers = fiona.listlayers(edge_gpkg) if file_exists(edge_gpkg) else []

    logger.info(f"Found hex layers: {layers_in_gpkg}")
    logger.info(f"Found existing edge layers: {existing_edge_layers}")

    def intersects_land(edge_line):
        for idx in land_tree.query(edge_line):
            try:
                candidate_geom = land_geoms[idx]  # Get land geometry
                key = wkb_dumps(candidate_geom, hex=True)
                map_idx = land_geom_index_map.get(key)

                if map_idx is not None:
                    if prepared_land[map_idx].intersects(edge_line):
                        return True
                else:
                    logger.warning(f"Land candidate geometry not found in index map.")

            except Exception as e:
                logger.error(f"Error processing candidate geometry: {e}")
        return False

    # Cache of {resolution: {h3_id: Point}}
    h3_centroids_by_resolution = {}

    for idx, layer in enumerate(layers_in_gpkg):
        resolution = int(layer.split('_r')[1])

        # Skip if edges for this resolution already exist
        edge_layer_name = f"edges_r{resolution}"
        if edge_layer_name in existing_edge_layers:
            logger.info(f"Skipping resolution r{resolution} as edges already exist in layer '{edge_layer_name}'.")
            continue

        logger.info(f"Processing layer {layer} at resolution r{resolution}...")

        hexes_gdf = gpd.read_file(output_gpkg, layer=layer)
        hexes_gdf['centroid_geom'] = hexes_gdf['centroid'].apply(wkt_loads)

        centroid_lookup = dict(zip(hexes_gdf['h3_id'].astype(str), hexes_gdf['centroid_geom']))
        h3_centroids_by_resolution[resolution] = centroid_lookup

        h3_ids = set(centroid_lookup.keys())
        edge_records = []
        seen_edges = set()

        for h3_id in tqdm(h3_ids,
                          desc=f"Processing r{resolution} H3 IDs{" with land-intersection" if resolution == COASTAL_SEA_RESOLUTION else ""}"):
            neighbours = h3.grid_disk(h3_id, 1)
            for neighbour in neighbours:
                if neighbour in h3_ids:
                    edge_key = tuple(sorted([h3_id, neighbour]))
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    edge_line = LineString([centroid_lookup[h3_id], centroid_lookup[neighbour]])
                    if not (resolution == COASTAL_SEA_RESOLUTION and intersects_land(edge_line)):
                        edge_records.append({"source": h3_id, "target": neighbour, "geometry": edge_line})
                else:
                    # Missing neighbour → attempt connection to parent/grandparent
                    parent_res = resolution - 1
                    grandparent_res = resolution - 2

                    for higher_res in (parent_res, grandparent_res):
                        if higher_res in h3_centroids_by_resolution:
                            higher_lookup = h3_centroids_by_resolution[higher_res]
                            target_id = h3.cell_to_parent(neighbour, higher_res)
                            if target_id in higher_lookup:
                                edge_key = (h3_id, target_id)
                                if edge_key in seen_edges:
                                    continue
                                seen_edges.add(edge_key)
                                edge_line = LineString([centroid_lookup[h3_id], higher_lookup[target_id]])
                                edge_records.append({
                                    "source": h3_id,
                                    "target": target_id,
                                    "geometry": edge_line,
                                    "cross_resolution": True
                                })
                                break  # Only link to the nearest existing ancestor

        # Save edge layer
        edges_gdf = gpd.GeoDataFrame(edge_records, crs="EPSG:4326")
        edges_gdf.to_file(edge_gpkg, layer=edge_layer_name, driver="GPKG")
        logger.info(f"Saved {len(edges_gdf)} edges to {edge_gpkg} layer '{edge_layer_name}'.")


def main():
    """Main function to run the sea zone and graph generation process."""
    get_land()
    # get_zoned_hex_graph()  # Add 5 for debugging
    # trim_to_aoi()
    # draw_pickled_cells()
    get_edges()


if __name__ == "__main__":
    main()
