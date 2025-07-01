import shutil
import sys
import zipfile
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr
import zarr
from copernicusmarine import subset
from numcodecs import Blosc
from scipy.spatial import cKDTree

from process.config import datasets, AOIS, copernicus_data_directory


def download_marine_dataset(dataset_name, dataset_info, bbox):
    dataset_id = dataset_info["dataset_id"]
    variable_list = dataset_info["nc_variables"]
    date_range = dataset_info.get("date_range", None)

    output_file = copernicus_data_directory / f"{dataset_name.lower().replace(' ', '_')}_subset.nc"

    if output_file.exists():
        print(f"✔ {dataset_name} already exists at {output_file}, skipping download.")
        return output_file

    print(f"⬇ Processing {dataset_name}...")
    try:
        subset(
            dataset_id=dataset_id,
            variables=variable_list,
            minimum_longitude=bbox[0],
            maximum_longitude=bbox[2],
            minimum_latitude=bbox[1],
            maximum_latitude=bbox[3],
            start_datetime=date_range[0] if date_range else None,
            end_datetime=date_range[1] if date_range else None,
            output_filename=str(output_file)
        )
        print(f"✅ Successfully downloaded {dataset_name} to {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Failed to process {dataset_name}: {e}", file=sys.stderr)
        return None


def compute_temporal_averages(output_file: Path, dataset_name: str, time_dim: str = None, var_name: str = None):
    if not output_file.exists():
        print(f"⚠ Cannot compute averages: {output_file} not found.")
        return

    monthly_avg_file = copernicus_data_directory / f"{dataset_name.lower().replace(' ', '_')}_monthly_avg.nc"
    annual_avg_file = copernicus_data_directory / f"{dataset_name.lower().replace(' ', '_')}_annual_avg.nc"

    try:
        with xr.open_dataset(output_file) as ds:
            # Determine time dimension
            if time_dim is None:
                if 'time' in ds.dims:
                    time_dim = 'time'
                elif 'valid_time' in ds.dims:
                    time_dim = 'valid_time'
                else:
                    print(f"⚠ Dataset {dataset_name} has no recognized time dimension; skipping averages.")
                    return

            # Select variable(s) if specified
            ds_to_process = ds[var_name] if var_name else ds

            # Compute monthly averages if not present
            if not monthly_avg_file.exists():
                print(f"⏳ Computing monthly averages for {dataset_name}...")
                ds_to_process = ds_to_process.assign_coords(month=ds[time_dim].dt.month)
                monthly_avg = ds_to_process.groupby('month').mean(dim=time_dim)
                monthly_avg.to_netcdf(monthly_avg_file)
                print(f"✅ Monthly averages saved to {monthly_avg_file}")
            else:
                print(f"✔ Monthly averages already exist for {dataset_name}")

            # Compute annual averages from the monthly averages file
            if not annual_avg_file.exists():
                print(f"⏳ Computing annual averages for {dataset_name} from monthly averages...")
                with xr.open_dataset(monthly_avg_file) as ds_monthly:
                    annual_avg = ds_monthly.mean(dim='month')
                    annual_avg.to_netcdf(annual_avg_file)
                print(f"✅ Annual averages saved to {annual_avg_file}")
            else:
                print(f"✔ Annual averages already exist for {dataset_name}")

    except Exception as e:
        print(f"❌ Error computing averages for {dataset_name}: {e}", file=sys.stderr)


def download_era5_dataset(dataset_name, bbox):
    dataset = datasets[dataset_name]
    dataset_lower = dataset_name.lower().replace(" ", "_")

    zip_path = copernicus_data_directory / f"{dataset_lower}_subset.zip"
    extract_dir = copernicus_data_directory / f"{dataset_lower}_extracted"

    if extract_dir.exists() and any(extract_dir.glob("*.nc")):
        print(f"✔ ERA5 {dataset_lower} data already extracted in {extract_dir}")
        return

    c = cdsapi.Client()

    print(f"⬇ Downloading ERA5 {dataset_lower} data from {dataset['dataset_id']}...")

    # Build request parameters conditionally
    request_params = {
        "product_type": [dataset["product_type"]],
        "variable": dataset["nc_variables"],
        "year": dataset["years"],
        "month": dataset["months"],
        "time": dataset["hours"],
        "format": "netcdf",
        "download_format": "zip",
        "area": [bbox[3], bbox[0], bbox[1], bbox[2]],  # [north, west, south, east]
    }

    if not dataset['dataset_id'].endswith("monthly-means"):
        request_params["day"] = dataset["days"]

    try:
        c.retrieve(
            dataset["dataset_id"],
            request_params,
            str(zip_path)
        )
        print(f"✅ ERA5 zip data saved to {zip_path}")

        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        nc_files = list(extract_dir.glob("*.nc"))
        if nc_files:
            print(f"✅ Extracted {len(nc_files)} .nc files to {extract_dir}")
        else:
            print(f"⚠ No .nc files found in {extract_dir}", file=sys.stderr)

        # Delete the zip file after extraction
        zip_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"❌ ERA5 download failed: {e}", file=sys.stderr)


# Helper class for fast nearest-neighbour lookup
class LatLonIndexer:
    def __init__(self, lats, lons):
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        self.points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        self.kdtree = cKDTree(self.points)
        self.shape = lat_grid.shape

    def query(self, lat, lon):
        _, idx = self.kdtree.query([lat, lon])
        return np.unravel_index(idx, self.shape)

    def save(self, path):
        np.savez_compressed(path, points=self.points)


# Compute chunk sizes based on AOI and resolution
def compute_chunk_sizes(aoi_bounds, resolution_degrees, time_len=12, max_chunk_size=100):
    """
    Compute chunk sizes for lat, lon, and time dimensions.

    Parameters:
    - aoi_bounds: tuple (min_lon, min_lat, max_lon, max_lat)
    - resolution_degrees: float, approximate size of one grid cell in degrees
    - time_len: int, length of time dimension (default 12 for months)
    - max_chunk_size: int, max number of points per chunk dimension to avoid huge chunks

    Returns:
    dict with keys: 'lat', 'lon', 'time'
    """
    min_lon, min_lat, max_lon, max_lat = aoi_bounds

    # Number of points in lat/lon direction based on AOI and resolution
    n_lat = int(np.ceil((max_lat - min_lat) / resolution_degrees))
    n_lon = int(np.ceil((max_lon - min_lon) / resolution_degrees))

    # Choose chunk size not bigger than max_chunk_size or dataset dimension
    chunk_lat = min(n_lat, max_chunk_size)
    chunk_lon = min(n_lon, max_chunk_size)
    chunk_time = min(time_len, max_chunk_size)

    return {"lat": chunk_lat, "lon": chunk_lon, "time": chunk_time}


# Main conversion function
def convert_netcdf_to_zarr(datasets, nc_root_dir, zarr_out_dir):
    zarr_out_dir.mkdir(parents=True, exist_ok=True)

    for name, info in datasets.items():
        dataset_dir = nc_root_dir / (name.lower().replace(" ", "_") + "_extracted")
        if dataset_dir.exists():
            nc_files = sorted(dataset_dir.glob("*.nc"))
            if not nc_files:
                print(f"Skipping {name}: no NetCDF files found.")
                continue

            print(f"Processing {name} from {len(nc_files)} files...")
        else:
            if name == "Bathymetry":
                nc_files = [nc_root_dir / "bathymetry_subset.nc"]
            else:
                nc_files = [nc_root_dir / (name.lower().replace(" ", "_") + "_monthly_avg.nc")]

        # Open each file separately and rechunk
        ds_list = []
        for nc_file in nc_files:
            ds_single = xr.open_dataset(nc_file)

            # Remove preset chunk encoding to avoid conflicts
            for var in ds_single.data_vars:
                ds_single[var].encoding.pop("chunks", None)

            # Determine chunk sizes for this dataset based on AOI and resolution
            lat_dim = [dim for dim in ds_single.dims if "lat" in dim][0]
            lon_dim = [dim for dim in ds_single.dims if "lon" in dim][0]
            lats = ds_single[lat_dim].values
            lons = ds_single[lon_dim].values

            chunk_sizes = compute_chunk_sizes(
                (lons.min(), lats.min(), lons.max(), lats.max()),
                info["resolution_degrees"],
                time_len=12 if "time" in ds_single.dims else 1
            )

            chunks = {
                lat_dim: chunk_sizes["lat"],
                lon_dim: chunk_sizes["lon"],
            }
            if "time" in ds_single.dims:
                chunks["time"] = chunk_sizes["time"]

            ds_single = ds_single.chunk(chunks)
            ds_list.append(ds_single)

        # Combine datasets along the time dimension or concatenate dims
        if len(ds_list) > 1:
            # Use concat if time dimension exists, else merge or first dataset
            if "time" in ds_list[0].dims:
                ds = xr.concat(ds_list, dim="time", data_vars="minimal", coords="minimal", compat="override")
            else:
                # If no time dimension, merge variables (adjust if needed)
                ds = xr.merge(ds_list)
        else:
            ds = ds_list[0]

        # If time dimension exists, convert to monthly means (optional)
        if "time" in ds.dims:
            ds["month"] = ds["time"].dt.month
            ds = ds.groupby("month").mean("time", keep_attrs=True).squeeze()
            ds = ds.reset_coords(drop=True)
            # Rechunk for monthly dimension
            chunks = {
                lat_dim: chunk_sizes["lat"],
                lon_dim: chunk_sizes["lon"],
                "month": 12,
            }
            ds = ds.chunk(chunks)

        # Remove any lingering chunk encodings before saving
        for var in ds.data_vars:
            ds[var].encoding.pop("chunks", None)

        # Write to Zarr
        zarr_path = zarr_out_dir / f"{name.lower().replace(' ', '_')}.zarr"
        if zarr_path.exists():
            shutil.rmtree(zarr_path)
        ds.to_zarr(zarr_path)

        # Save Lat/Lon index (if needed)
        indexer = LatLonIndexer(ds[lat_dim].values, ds[lon_dim].values)
        indexer.save(zarr_path / "latlon_index.npz")

        print(f"✓ {name} written to {zarr_path}")

    print("✅ All datasets converted to optimised Zarr.")


def main():
    AOI_index = 0  # Index of the AOI to use, can be changed or passed via CLI
    bbox = list(AOIS[AOI_index]["bounds"])  # Use Europe bounding box

    # downloaded_files = []
    # for dataset_name, dataset_info in datasets.items():
    #     if dataset_name in ["Weather"]:
    #         download_era5_dataset(dataset_name, bbox)
    #     else:
    #         output_file = download_marine_dataset(dataset_name, dataset_info, bbox)
    #         if output_file:
    #             downloaded_files.append((output_file, dataset_name))
    #
    # for output_file, dataset_name in downloaded_files:
    #     compute_temporal_averages(output_file, dataset_name)

    convert_netcdf_to_zarr(
        datasets=datasets,
        nc_root_dir=copernicus_data_directory,
        zarr_out_dir=copernicus_data_directory / "zarr"
    )


if __name__ == "__main__":
    main()
