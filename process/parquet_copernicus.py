import logging
import warnings
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from dask.array.core import xr
from tqdm import tqdm

from process.config import AOIS, copernicus_data_directory
from process.copernicus_query import query_all_months, daylight_zarr_path, DatasetCache
from process.sea_graph import COASTAL_SEA_RESOLUTION

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

    dataset_cache = DatasetCache(
        modal={
            "wind": xr.open_zarr(copernicus_data_directory / "zarr" / "wind_modal_monthly.zarr", consolidated=True),
            "current": xr.open_zarr(copernicus_data_directory / "zarr" / "current_modal_monthly.zarr",
                                    consolidated=True)
        },
        weather=xr.open_zarr(copernicus_data_directory / "zarr" / "weather.zarr", consolidated=True),
        bathymetry=xr.open_zarr(copernicus_data_directory / "zarr" / "bathymetry.zarr", consolidated=True),
        daylight=xr.open_zarr(daylight_zarr_path, consolidated=True)
    )

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
                monthly_data = query_all_months((row.lat, row.lon), h3id=row.h3_id, cache=dataset_cache)
                if monthly_data[0]["deptho"] is None:
                    logger.warning(f"{row.h3_id} at ({row.lat}, {row.lon}) is missing depth data.")
                for month_idx, stats in enumerate(monthly_data, start=1):
                    record = {
                        "h3_id": row.h3_id,
                        "month": month_idx,
                        "dist_m": row.dist_m,
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
