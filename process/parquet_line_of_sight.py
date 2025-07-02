import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import fastparquet
import fiona
import geopandas as gpd
import h3
import pandas as pd
from shapely import wkt
from shapely.geometry.point import Point
from tqdm import tqdm

from process.config import AOIS

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
AOI = AOIS[0]

# --- Constants ---
R_EARTH = 6_371_000  # metres
DECK_HEIGHT_M = 3.5  # Estimated height of a ship's deck plus observer's eye level in metres


def arc_horizon_distance(height_m: float) -> float:
    """
    Computes arc distance to horizon from a given height above Earth's surface.
    """
    return R_EARTH * math.acos(R_EARTH / (R_EARTH + height_m))


OBSERVER_HORIZON_DISTANCE = arc_horizon_distance(DECK_HEIGHT_M)


def get_deck_visibility_distance(elevation_m: float) -> float:
    return arc_horizon_distance(elevation_m) + OBSERVER_HORIZON_DISTANCE


def find_first_visible_land(
        sea_hex_id: str,
        r5_hex_id: str,
        sea_geom: Point,
        land_lookup: dict[str, Tuple[Point, float]],
        max_deck_visibility_distance: float
) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Find first visible land hex from a given sea hex by increasing H3 radius,
    stopping early if visibility is found or horizon exceeded.
    """
    sea_latlon = (sea_geom.y, sea_geom.x)
    seen: set[str] = set()
    k = 1
    max_d = 0
    best_result = (sea_hex_id, None, None)

    while k < 14 and max_d < max_deck_visibility_distance:
        if k == 1:
            ring = h3.grid_disk(r5_hex_id, 1)  # Include the center hex
        else:
            ring = h3.grid_ring(r5_hex_id, k)
        new_hexes = [h for h in ring if h not in seen]
        if not new_hexes:
            break
        seen.update(new_hexes)

        for land_hex_id in new_hexes:
            if land_hex_id not in land_lookup:
                continue
            land_point, deck_visibility_distance = land_lookup[land_hex_id]

            land_latlon = (land_point.y, land_point.x)

            d = h3.great_circle_distance(sea_latlon, land_latlon, unit='m')

            if d < deck_visibility_distance and (best_result[1] is None or d < best_result[1]):
                best_result = (sea_hex_id, d, land_hex_id)

            max_d = max(max_d, d)

        k += 1

    return best_result


def main():
    # --- Filepaths ---
    docs_directory = Path(__file__).resolve().parent.parent / "docs"
    geo_output_directory = docs_directory / "data" / AOI["name"]
    geo_output_directory.mkdir(parents=True, exist_ok=True)

    output_gpkg = geo_output_directory / "graph.gpkg"
    dem_parquet = geo_output_directory / "dem_land_r5.parquet"
    visibility_parquet = geo_output_directory / "visibility_distance.parquet"
    land_lookup_path = geo_output_directory / "land_lookup.parquet"

    # --- Load Land Hex Data ---
    logger.info("Loading elevation and geometry data...")
    gdf_land_nodes = gpd.read_file(output_gpkg, layer="land_hexes_r5")
    gdf_land_nodes["geometry"] = gdf_land_nodes["centroid"].apply(wkt.loads)
    df_elev = pd.read_parquet(dem_parquet)
    gdf_land_nodes = gdf_land_nodes.merge(df_elev, on="h3_id")

    if land_lookup_path.exists():
        df_lookup = pd.read_parquet(land_lookup_path)
        df_lookup["geometry"] = df_lookup["geometry"].apply(wkt.loads)
        land_lookup = {
            row.h3_id: (row.geometry, row.deck_visibility_distance)
            for row in df_lookup.itertuples()
        }
        logger.info(f"Loaded {len(land_lookup)} land hexes from precomputed file.")
    else:
        land_lookup = {
            row.h3_id: (
                row.geometry,
                get_deck_visibility_distance(row.max_elevation_m)
            )
            for row in gdf_land_nodes.itertuples()
        }
        logger.info(f"Computed {len(land_lookup)} land hexes. Saving for reuse...")
        pd.DataFrame([
            {
                "h3_id": k,
                "geometry": v[0].wkt,
                "deck_visibility_distance": v[1]
            } for k, v in land_lookup.items()
        ]).to_parquet(land_lookup_path, index=False)

    # --- Horizon Parameters ---
    max_deck_visibility_distance = max(d for _, d in land_lookup.values())

    # --- Loop through layers ---
    layers_in_gpkg = sorted(
        [layer for layer in fiona.listlayers(output_gpkg) if layer.startswith("hexes_r")],
        key=lambda x: int(x.split('_r')[1])
    )

    for layer in layers_in_gpkg:
        resolution = int(layer.split('_r')[1])
        logger.info(f"Processing layer {layer} (r{resolution})...")

        gdf_nodes = gpd.read_file(output_gpkg, layer=layer)
        sea_hexes = []
        for node in gdf_nodes.itertuples():
            if node.dist_m > max_deck_visibility_distance:
                # Skip nodes that are too far from land
                continue

            hex_id = str(node.h3_id)
            r5_hex_id = str(
                h3.cell_to_parent(hex_id, 5) if resolution > 5 else
                h3.cell_to_center_child(hex_id, 5) if resolution < 5 else
                hex_id
            )
            point_geom = wkt.loads(node.centroid)

            sea_hexes.append((hex_id, r5_hex_id, point_geom, node.dist_m))

        if not sea_hexes:
            logger.info(f"No sea nodes found for layer {layer}. Skipping...")
            continue

        # --- Compute Visibility in Parallel ---
        # logger.info(f"Computing visibility for {len(sea_hexes)} sea nodes...")
        # with ProcessPoolExecutor() as executor:
        #     futures = [
        #         executor.submit(find_first_visible_land, hex_id, r5_hex_id, geom, land_lookup, max_deck_visibility_distance)
        #         for hex_id, r5_hex_id, geom, _ in sea_hexes
        #     ]
        #     results = [f.result() for f in futures]
        #
        # logger.info(f"Computing visibility for {len(sea_hexes)} sea nodes...")

        results = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    find_first_visible_land,
                    hex_id, r5_hex_id, geom, land_lookup, max_deck_visibility_distance
                ): hex_id
                for hex_id, r5_hex_id, geom, _ in sea_hexes
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Visibility"):
                results.append(future.result())

        substantially_visible = sum(1 for res in results if res[1] is not None)
        logger.info(f"Substantially visible land found for {substantially_visible} of {len(results)} sea nodes.")

        df = pd.DataFrame(results, columns=["hex_id", "distance_to_visible_land_m", "visible_land_hex_id"])
        df["hex_id"] = df["hex_id"].astype(str)
        df["visible_land_hex_id"] = df["visible_land_hex_id"].astype("string")

        # --- Save (Append if file exists) ---
        if visibility_parquet.exists():
            existing = fastparquet.ParquetFile(visibility_parquet)
            df_existing = existing.to_pandas()
            df_combined = pd.concat([df_existing, df], ignore_index=True).drop_duplicates(subset=["hex_id"])
            fastparquet.write(visibility_parquet, df_combined, compression='snappy')
        else:
            fastparquet.write(visibility_parquet, df, compression='snappy')

        logger.info(f"Saved results to {visibility_parquet}")


if __name__ == "__main__":
    main()
