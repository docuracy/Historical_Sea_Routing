import logging
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from process.config import AOIS
from process.copernicus_query import query_all_months
from process.sea_graph_v3 import COASTAL_SEA_RESOLUTION

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"The codec `vlen-utf8` is currently not part.*",
    category=UserWarning,
    module=r"zarr\.codecs\.vlen_utf8"
)

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
AOI = AOIS[0]


def main():
    docs_directory = Path(__file__).resolve().parent.parent / "docs"
    geo_output_directory = docs_directory / "data" / AOI["name"]
    geo_output_directory.mkdir(parents=True, exist_ok=True)

    output_gpkg = geo_output_directory / "graph.gpkg"
    copernicus_parquet = geo_output_directory / "copernicus.parquet"

    # Load existing results
    processed_ids = set()
    if copernicus_parquet.exists():
        done_df = pd.read_parquet(copernicus_parquet)
        processed_ids = set(done_df["h3_id"])
        print(f"Skipping {len(processed_ids)} already processed cells.")

    # Get the names of the layers in the geopackage
    layers_in_gpkg = fiona.listlayers(output_gpkg)
    layer = 0
    started = False
    stop = False
    while layer <= COASTAL_SEA_RESOLUTION and not (started and stop):
        layer_name = f"hexes_r{layer}"
        if layer_name in layers_in_gpkg:
            started = True
            gdf = gpd.read_file(output_gpkg, layer=layer_name)
            gdf["h3_id"] = gdf["h3_id"].astype(str)

            gdf = gdf[~gdf["h3_id"].isin(processed_ids)]
            if gdf.empty:
                print(f"All r{layer} cells already processed.")
                layer += 1
                continue

            # `centroid` is stored as "POINT (0.7351256407309354 60.36727256077856)", so extract to lat and lon
            gdf["lon"] = gdf["centroid"].str.extract(r"\(\s*([^\s]+)")[0].astype(float)
            gdf["lat"] = gdf["centroid"].str.extract(r"\s([^\s]+)\)")[0].astype(float)

            # Process in batches
            batch = []
            batch_size = 250
            engine = "fastparquet"

            for row in tqdm(gdf.itertuples(index=False), total=len(gdf), desc=f"Copernicus r{layer}"):
                monthly_data = query_all_months((row.lat, row.lon))
                if monthly_data[0]["deptho"] is None:
                    logger.warning(f"{row.h3_id} at ({row.lat}, {row.lon}) is missing depth data.")
                for month_idx, stats in enumerate(monthly_data, start=1):
                    record = {
                        "h3_id": row.h3_id,
                        "month": month_idx,
                        **stats,
                    }
                    batch.append(record)

                if len(batch) >= batch_size * 12:  # 12 months
                    df = pd.DataFrame(batch)
                    df.to_parquet(copernicus_parquet, index=False, engine=engine, append=copernicus_parquet.exists())
                    batch.clear()

            # Write final batch
            if batch:
                df = pd.DataFrame(batch)
                df.to_parquet(copernicus_parquet, index=False, engine=engine, append=copernicus_parquet.exists())

            print(f"Completed r{layer} and added to {copernicus_parquet}")

        else:
            logger.warning(f"Layer {layer_name} not found in {output_gpkg}.")
            stop = started
        layer += 1


if __name__ == "__main__":
    main()
