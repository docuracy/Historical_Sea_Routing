import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from process.config import AOIS
from process.dem import ensure_cache_dir, h3_max_elevation

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
AOI = AOIS[0]


def main():
    docs_directory = Path(__file__).resolve().parent.parent / "docs"
    geo_output_directory = docs_directory / "data" / AOI["name"]
    geo_output_directory.mkdir(parents=True, exist_ok=True)

    output_gpkg = geo_output_directory / "graph.gpkg"

    dem_parquet = geo_output_directory / "dem_land_r5.parquet"

    print("Reading land_hexes_r5 layer...")
    gdf = gpd.read_file(output_gpkg, layer="land_hexes_r5")
    gdf["h3_id"] = gdf["h3_id"].astype(str)
    gdf["hex_shape"] = gdf.geometry

    ensure_cache_dir()

    # Load existing results
    processed_ids = set()
    if dem_parquet.exists():
        done_df = pd.read_parquet(dem_parquet)
        processed_ids = set(done_df["h3_id"])
        print(f"Skipping {len(processed_ids)} already processed cells.")

    gdf = gdf[~gdf["h3_id"].isin(processed_ids)]
    if gdf.empty:
        print("All cells already processed.")
        return

    # Process in batches
    batch = []
    batch_size = 10
    engine = "fastparquet"

    for row in tqdm(gdf.itertuples(index=False), total=len(gdf), desc="Elevation"):
        max_elev = h3_max_elevation(row.h3_id, hex_shape=row.hex_shape)
        batch.append({"h3_id": row.h3_id, "max_elevation_m": max_elev})

        if len(batch) >= batch_size:
            df = pd.DataFrame(batch)
            df.to_parquet(dem_parquet, index=False, engine=engine, append=dem_parquet.exists())
            batch.clear()

    # Write final batch
    if batch:
        df = pd.DataFrame(batch)
        df.to_parquet(dem_parquet, index=False, engine=engine, append=dem_parquet.exists())

    print(f"Completed and saved to {dem_parquet}")


if __name__ == "__main__":
    main()
