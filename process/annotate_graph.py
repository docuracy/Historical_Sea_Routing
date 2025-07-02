import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from shapely.geometry import Point
from pathlib import Path

# === Configuration ===
AOI = {"name": "your_aoi_name"}  # Replace or pass via CLI
docs_directory = Path(__file__).resolve().parent.parent / "docs"
geo_output_directory = docs_directory / "data" / AOI["name"]
geo_output_directory.mkdir(parents=True, exist_ok=True)

output_gpkg = geo_output_directory / "graph.gpkg"
visibility_nc = Path("/path/to/mean_monthly_visibility.nc")  # update this
output_parquet_dir = geo_output_directory / "visibility_parquets"
output_parquet_dir.mkdir(exist_ok=True)

# === Functions ===
def load_visibility_dataset(path):
    return xr.open_dataset(path)

def get_monthly_visibilities(ds, lat, lon):
    """Returns a list of 12 monthly visibility values at lat/lon."""
    try:
        vals = ds['visibility'].interp(lat=lat, lon=lon, method="nearest")
        return [float(vals.isel(time=month).item()) for month in range(12)]
    except Exception:
        return [np.nan] * 12

def process_layer(layername, ds, output_dir):
    gdf = gpd.read_file(output_gpkg, layer=layername)
    gdf["centroid"] = gdf.geometry.centroid

    vis_cols = [f"vis_{str(i+1).zfill(2)}" for i in range(12)]
    vis_values = [get_monthly_visibilities(ds, pt.y, pt.x) for pt in gdf["centroid"]]

    vis_df = pd.DataFrame(vis_values, columns=vis_cols)
    result = pd.concat([gdf.drop(columns="centroid"), vis_df], axis=1)

    output_path = output_dir / f"graph_{layername}.parquet"
    result.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(result)} rows to {output_path.name}")

# === Main ===
def main():
    ds = load_visibility_dataset(visibility_nc)
    layers = fiona.listlayers(output_gpkg)

    for layername in layers:
        print(f"▶ Processing layer: {layername}")
        process_layer(layername, ds, output_parquet_dir)

if __name__ == "__main__":
    import fiona  # ensure fiona is available for listing GPKG layers
    main()
