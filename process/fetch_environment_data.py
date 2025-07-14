# fetch_environment_data.py

import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import copernicusmarine
import numpy as np
import xarray as xr
import zarr
from dask import delayed, compute
from scipy.spatial import cKDTree
from tqdm import tqdm

from process.bin_utils import compute_all_bins_to_json, get_binned_data_for_components_dask, assign_bin_index
from process.config import datasets, AOIS, copernicus_data_directory
from process.sea_graph import get_all_unique_h3_centroids_df

from confidence_analysis import main as analyse_confidence

# Suppress DEBUG logging from known libraries
logging.basicConfig(level=logging.WARNING)
for noisy in ["fsspec", "zarr", "asyncio", "concurrent.futures"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

warnings.filterwarnings("ignore", message="The codec `vlen-utf8`.*")
warnings.filterwarnings("ignore", message=".*Consolidated metadata.*Zarr format 3.*")


def load_dataset_with_indexer(dataset_name):
    ds_path = copernicus_data_directory / f"{dataset_name}_subset.zarr"
    index_path = ds_path.parent / (ds_path.name + "_latlon_index.npz")

    try:
        ds = xr.open_zarr(ds_path, consolidated=True)
    except (KeyError, FileNotFoundError, ValueError):
        print("⚠️  Consolidated metadata missing or corrupted, rebuilding...")
        # Open un-consolidated, add missing _ARRAY_DIMENSIONS if needed
        store = zarr.open(str(ds_path), mode="a")
        ref_var = "VHM0_WW" if "VHM0_WW" in store else list(store.array_keys())[0]
        ref_dims = store[ref_var].attrs.get('_ARRAY_DIMENSIONS', None)

        if not ref_dims:
            raise RuntimeError("Cannot rebuild: no _ARRAY_DIMENSIONS found on any variable.")

        for var_name in store.array_keys():
            if '_ARRAY_DIMENSIONS' not in store[var_name].attrs:
                print(f"🔧 Setting _ARRAY_DIMENSIONS for {var_name}")
                store[var_name].attrs['_ARRAY_DIMENSIONS'] = list(ref_dims)

        # Rebuild consolidated metadata
        xr.open_zarr(ds_path, consolidated=False).to_zarr(ds_path, mode="a", consolidated=True)
        ds = xr.open_zarr(ds_path, consolidated=True)

    indexer = LatLonIndexer.load(index_path)
    return ds, indexer


def fetch_and_index_marine_dataset(dataset_name, bbox):
    # copernicusmarine.login()
    dataset_info = datasets.get(dataset_name)
    if dataset_info is None:
        raise ValueError(f"Dataset '{dataset_name}' is not defined in the datasets configuration.")

    output_file = copernicus_data_directory / f"{dataset_name.lower().replace(' ', '_')}_subset.zarr"

    output_file = Path(output_file)
    if output_file.is_dir():
        print(f"✔ {dataset_name} already exists at {output_file}, skipping download.")
    else:
        dataset_id = dataset_info["dataset_id"]
        variable_list = list(dataset_info["variables"].values())
        date_range = dataset_info.get("date_range", None)

        print(f"⬇ Processing {dataset_name}...")
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variable_list,
                minimum_longitude=bbox[0],
                maximum_longitude=bbox[2],
                minimum_latitude=bbox[1],
                maximum_latitude=bbox[3],
                start_datetime=date_range[0] if date_range else None,
                end_datetime=date_range[1] if date_range else None,
                output_filename=str(output_file),
                force_download=True,
            )
            print(f"✅ Successfully downloaded {dataset_name} to {output_file}")

        except Exception as e:
            print(f"❌ Failed to process {dataset_name}: {e}", file=sys.stderr)

    index_path = output_file.parent / (output_file.name + "_latlon_index.npz")
    if index_path.exists():
        print(f"✔ LatLon index already exists at {index_path}, skipping indexing.")
        return

    ds_downloaded = xr.open_zarr(output_file, consolidated=True)
    lats = ds_downloaded["latitude"].values
    lons = ds_downloaded["longitude"].values

    indexer = LatLonIndexer(lats, lons)
    indexer.save(index_path)
    print(f"🗺️ LatLon index saved to {index_path}")


class LatLonIndexer:
    def __init__(self, lats, lons):
        self.lats = np.asarray(lats)
        self.lons = np.asarray(lons)
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        self.shape = lat_grid.shape

        self.points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        self.kdtree = cKDTree(self.points)

    def query(self, lat, lon):
        _, idx = self.kdtree.query([lat, lon])
        return np.unravel_index(idx, self.shape)

    def query_batch(self, lat_array, lon_array):
        # stack lat/lon into Nx2 array
        points = np.column_stack((lat_array, lon_array))
        _, idx = self.kdtree.query(points)
        return np.unravel_index(idx, self.shape)

    def save(self, path):
        np.savez_compressed(path, lats=self.lats, lons=self.lons)

    @classmethod
    def load(cls, path):
        npz = np.load(path)
        return cls(lats=npz["lats"], lons=npz["lons"])


def compute_angular_components_DEPRECATED(ds, input_vars, output_vars, phases=None):
    def components(u, v):
        mag = np.sqrt(u ** 2 + v ** 2)
        dir_rad = np.arctan2(-u, -v)
        dir_deg = np.rad2deg(dir_rad) % 360
        return mag, dir_deg

    # --- Start of Proposed Modification ---

    # Identify if 'depth' dimension exists and is present in input variables
    # Assuming 'depth' is always the second dimension if present.
    # We'll apply .isel(depth=0) to the input DataArrays before calculation.
    # This ensures that 'mag' and 'deg' will also be 3D (time, lat, lon).

    ds_processed = ds # Start with the original dataset

    # Check for 'depth' dimension in the first input variable (assuming consistent dimensions)
    # Check if 'depth' exists and its position (expected to be the second dim)
    if 'depth' in ds[input_vars[0]].dims and ds[input_vars[0]].dims.index('depth') == 1:
        # If 'depth' is present as the second dimension, select the first level (index 0)
        # for all relevant variables.
        # This will return a new Dataset or DataArray view with 'depth' squeezed out
        # or reduced to a single point.
        ds_processed = ds.isel(depth=0)
        # print(f"DEBUG: Squeezed depth dimension. New ds_processed dims: {ds_processed.dims}")
        # For verification, check a variable's shape after squeezing:
        # print(f"DEBUG: ds_processed['{input_vars[0]}'].shape after squeezing: {ds_processed[input_vars[0]].shape}")

    # --- End of Proposed Modification ---


    # Now, use ds_processed for calculations
    # If depth was squeezed, these will be 3D DataArrays (time, lat, lon)
    mag, deg = components(ds_processed[input_vars[0]], ds_processed[input_vars[1]])

    result = {
        output_vars[0]: mag,
        output_vars[1]: deg,
    }

    if phases and "zos" in ds_processed: # Use ds_processed here too!
        zos = ds_processed["zos"]

        if zos.sizes["time"] < 2:
            raise ValueError("At least two time points are required for phase detection.")

        delta_zos = zos.diff(dim="time")

        # Pad with NaNs to align with original time
        pad = xr.full_like(zos.isel(time=0), np.nan)
        pad = pad.expand_dims(time=[zos.time.values[0]])
        delta_zos_full = xr.concat([pad, delta_zos], dim="time")

        # Ensure time coordinate matches original
        delta_zos_full["time"] = zos.time

        phase_arr = (delta_zos_full > 0).astype(np.int8)  # 1=flood, 0=ebb
        result["phase"] = phase_arr

    return result


def compute_angular_components(ds, input_vars, output_vars, phases=None):
    def components(u, v):
        mag = np.sqrt(u ** 2 + v ** 2)
        dir_rad = np.arctan2(-u, -v)
        dir_deg = np.rad2deg(dir_rad) % 360
        return mag, dir_deg

    mag, deg = components(ds[input_vars[0]], ds[input_vars[1]])

    result = {
        output_vars[0]: mag,
        output_vars[1]: deg,
    }

    if phases and "zos" in ds:
        zos = ds["zos"]

        if zos.sizes["time"] < 2:
            raise ValueError("At least two time points are required for phase detection.")

        delta_zos = zos.diff(dim="time")  # shape: (T-1, lat, lon)

        # Pad with NaNs to align with original time
        pad = xr.full_like(zos.isel(time=0), np.nan)
        pad = pad.expand_dims(time=[zos.time.values[0]])
        delta_zos_full = xr.concat([pad, delta_zos], dim="time")

        # Ensure time coordinate matches original
        delta_zos_full["time"] = zos.time

        phase_arr = (delta_zos_full > 0).astype(np.int8)  # 1=flood, 0=ebb
        result["phase"] = phase_arr

    return result


def process_batch_wind(start, end, latlon_batch, derived_vars_path, times_np, bin_edges, input_vars, output_vars,
                       n_months):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from collections import Counter

    ds = xr.open_zarr(derived_vars_path)
    derived_vars = compute_angular_components(ds, input_vars, output_vars)  # Must be re-run inside each process

    batch_size_actual = end - start

    binned_vars = get_binned_data_for_components_dask(derived_vars, latlon_batch, bin_edges)

    binned_stack = np.stack([binned_vars[var] for var in output_vars], axis=-1).transpose(1, 0, 2)
    flat = binned_stack.reshape(-1, len(output_vars))
    time_col = np.tile(times_np, batch_size_actual)
    node_indices = np.repeat(np.arange(batch_size_actual), len(times_np))

    df = pd.DataFrame(flat, columns=output_vars)
    df["time"] = pd.to_datetime(time_col)
    df["month"] = df["time"].dt.month
    df["node_index"] = node_indices
    df = df[~(df[output_vars] < 0).any(axis=1)]

    out_vars_arr = {var: np.full((batch_size_actual, n_months), -1, dtype=np.int8) for var in output_vars}
    confidence_arr = np.full((batch_size_actual, n_months), np.nan, dtype=np.float32)

    if not df.empty:
        for (node_idx, month), group in df.groupby(["node_index", "month"]):
            tuples = list(map(tuple, group[output_vars].values))
            counter = Counter(tuples)
            mode_tuple, mode_count = counter.most_common(1)[0]
            total = len(tuples)

            for j, var in enumerate(output_vars):
                out_vars_arr[var][node_idx, month - 1] = mode_tuple[j]
            confidence_arr[node_idx, month - 1] = mode_count / total

    return start, end, out_vars_arr, confidence_arr


def create_wind_modal_zarr(
        bins,
        batch_size=200,
        output_filename="wind_modal_monthly.zarr",
):
    output_path = copernicus_data_directory / "zarr" / output_filename
    if output_path.exists():
        print(f"✔ Output file {output_path} already exists, skipping creation.")
        return

    print(f"⬇ Creating modal Zarr dataset at {output_path}...")

    centroid_df = get_all_unique_h3_centroids_df()
    all_locations = centroid_df[["latitude", "longitude"]].values
    n_nodes = len(centroid_df["h3_id"])
    n_months = 12

    dataset_name = "wind_hourly"
    input_vars = list(datasets.get(dataset_name, {}).get("variables", {}).values())
    output_vars = ["angle_deg", "magnitude"]

    ds_path = copernicus_data_directory / f"{dataset_name}_subset.zarr"
    ds, indexer = load_dataset_with_indexer(dataset_name)
    lat_idx_arr, lon_idx_arr = indexer.query_batch(all_locations[:, 0], all_locations[:, 1])
    latlon_indices = np.column_stack((lat_idx_arr, lon_idx_arr))

    times = ds.time.values
    bin_edges = {var: np.array(bins[dataset_name][var]["bin_edges"]) for var in output_vars}

    empty = xr.Dataset(
        {
            **{
                var: (("h3_id", "month"), np.full((n_nodes, n_months), -1, dtype=np.int8))
                for var in output_vars
            },
            "confidence": (("h3_id", "month"), np.full((n_nodes, n_months), np.nan, dtype=np.float32))
        },
        coords={
            "h3_id": ("h3_id", centroid_df["h3_id"].values),
            "month": ("month", np.arange(1, n_months + 1))
        }
    )
    empty.to_zarr(output_path, mode="w")
    zarr_store = zarr.open(str(output_path), mode='a')

    tasks = []
    with ProcessPoolExecutor() as executor:
        for start in range(0, n_nodes, batch_size):
            end = min(start + batch_size, n_nodes)
            latlon_batch = latlon_indices[start:end]  # small NumPy array, safe to pass

            future = executor.submit(
                process_batch_wind,
                start, end, latlon_batch,
                ds_path,
                times,
                bin_edges,
                input_vars,
                output_vars,
                n_months
            )
            tasks.append(future)

        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="✅ Writing", unit="batch"):
            start, end, out_vars_arr, confidence_arr = fut.result()
            for var in output_vars:
                zarr_store[var][start:end, :] = out_vars_arr[var]
            zarr_store["confidence"][start:end, :] = confidence_arr

    print(f"✅ Modal Zarr dataset written to {output_path}")

_cached_ds = None
def process_batch_current(start, end, latlon_batch, derived_vars_path, times_np, bin_edges, input_vars, output_vars,
                       n_months, phases):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from collections import Counter
    import os # Import os for pid

    global _cached_ds

    # Check if the dataset is already loaded in this worker process
    if _cached_ds is None:
        print(f"Worker PID {os.getpid()}: Opening Zarr store {derived_vars_path} for the first time...")
        _cached_ds = xr.open_zarr(derived_vars_path)
        print(f"Worker PID {os.getpid()}: Zarr store opened. Dims: {_cached_ds.dims}")
    else:
        # print(f"Worker PID {os.getpid()}: Using cached Zarr store.")
        pass # Using cached ds

    ds = _cached_ds # Use the cached dataset

    # # Debugging output
    # print(f"Inside process_batch_current. ds dims: {ds.dims}")
    # print(f"Inside process_batch_current. ds coordinates: {ds.coords}")
    # print(f"Inside process_batch_current. derived_vars_path: {derived_vars_path}")
    # # Add checks for specific variables' dimensions
    # for var in input_vars + output_vars:
    #     if var in ds:
    #         print(f"  ds['{var}'] dims: {ds[var].dims}, shape: {ds[var].shape}")

    derived_vars = compute_angular_components(ds, input_vars, output_vars, phases)

    # # Debugging output
    # print(f"Inside process_batch_current. derived_vars after compute_angular_components:")
    # for var, da in derived_vars.items():
    #     print(f"  derived_vars['{var}'] dims: {da.dims}, shape: {da.shape}")
    #
    # # Sanity check
    # ny, nx = ds.dims["latitude"], ds.dims["longitude"]
    # print(f"ds dimensions: ny={ny}, nx={nx}")
    # for i, (lat_idx, lon_idx) in enumerate(latlon_batch):
    #     if lat_idx >= ny or lon_idx >= nx:
    #         raise IndexError(f"Index out of bounds at {i}: lat_idx={lat_idx}, lon_idx={lon_idx} for shape=({ny}, {nx})")

    batch_size_actual = end - start

    # --- OPTIMIZED DATA EXTRACTION ---

    # 1. Prepare indices for Xarray's .isel
    lats_to_select = [idx[0] for idx in latlon_batch]
    lons_to_select = [idx[1] for idx in latlon_batch]

    # Create xr.DataArray for advanced indexing (this is key for efficiency)
    lat_indexer = xr.DataArray(lats_to_select, dims='batch_points')
    lon_indexer = xr.DataArray(lons_to_select, dims='batch_points')

    # Now, extract ALL relevant time series for the batch in one go for all variables
    # This creates a Dask graph for the batch extraction.
    # The .data will trigger the computation of these batch-wise slices.
    # The result will be (time, batch_points) numpy array.

    # Extract all required raw data for output_vars and phase
    # This operation will trigger Dask computation to load the data for the entire batch
    # across all relevant variables (magnitude, angle, phase)
    # The .compute() is explicitly added to force the Dask graph to run here
    # and load the data as numpy arrays. This can be memory intensive if batch_size_actual * times_np is huge.
    # If memory becomes an issue, you might need to leave it as a dask array and pass that to delayed,
    # but for most typical scenarios, loading it here is more efficient for the downstream pandas/numpy.

    extracted_data = {}
    for var in output_vars + ['phase']:
        # derived_vars[var] is now 3D (time, lat, lon) after depth removal
        da = derived_vars[var]
        extracted_data[var] = da.isel(
            latitude=lat_indexer,
            longitude=lon_indexer
        ).data.compute() # .data gets the underlying Dask array, .compute() materializes it to NumPy

    # Bin angle and magnitude using the extracted numpy arrays
    binned_vars = {}
    for var in output_vars:
        # extracted_data[var] is now (time, batch_points) NumPy array
        # Assign bin index to each time series within the batch
        binned = [
            delayed(assign_bin_index)(extracted_data[var][:, i], bin_edges[var])
            for i in range(batch_size_actual)
        ]
        binned_compute = compute(*binned) # Execute the delayed binning tasks
        binned_vars[var] = np.stack(binned_compute, axis=0).T.astype(np.int16) # (time, batch_points)

    # Now, the phase array is also readily available from extracted_data
    # extracted_data['phase'] is (time, batch_points) NumPy array
    phase_array = extracted_data['phase'].T.flatten() # Flattens to (time * batch_points)

    # --- END OPTIMIZED DATA EXTRACTION ---

    # # Bin angle and magnitude
    # binned_vars = get_binned_data_for_components_dask(derived_vars, latlon_batch, bin_edges)
    #
    # # Fetch phase values (0=ebb, 1=flood) from derived_vars["phase"]
    # # shape: (time, batch_size_actual)
    # phase_array = np.stack([
    #     derived_vars["phase"][:, lat, lon].values
    #     for lat, lon in latlon_batch
    # ], axis=1)  # shape: (time, batch_size_actual)
    # phase_array = phase_array.T.flatten()  # shape: time * batch_size_actual

    # Build flat DataFrame
    binned_stack = np.stack([binned_vars[var] for var in output_vars], axis=-1).transpose(1, 0, 2)
    flat = binned_stack.reshape(-1, len(output_vars))

    time_col = np.tile(times_np, batch_size_actual)
    node_indices = np.repeat(np.arange(batch_size_actual), len(times_np))

    df = pd.DataFrame(flat, columns=output_vars)
    df["time"] = pd.to_datetime(time_col)
    df["month"] = df["time"].dt.month
    df["node_index"] = node_indices
    df["phase"] = phase_array
    df = df[~(df[output_vars] < 0).any(axis=1)]

    # Preallocate output arrays
    out_vars_arr = {var: np.full((batch_size_actual, n_months, 2), -1, dtype=np.int8) for var in output_vars}
    confidence_arr = np.full((batch_size_actual, n_months, 2), np.nan, dtype=np.float32)

    if not df.empty:
        for (node_idx, month, phase), group in df.groupby(["node_index", "month", "phase"]):
            tuples = list(map(tuple, group[output_vars].values))
            counter = Counter(tuples)
            mode_tuple, mode_count = counter.most_common(1)[0]
            total = len(tuples)
            phase_idx = phases.index("flood") if phase == 1 else phases.index("ebb")

            for j, var in enumerate(output_vars):
                out_vars_arr[var][node_idx, month - 1, phase_idx] = mode_tuple[j]
            confidence_arr[node_idx, month - 1, phase_idx] = mode_count / total

    return start, end, out_vars_arr, confidence_arr


def create_current_modal_zarr(
        bins,
        batch_size=50,
        output_filename="current_modal_monthly.zarr",
):
    output_path = copernicus_data_directory / "zarr" / output_filename
    if output_path.exists():
        print(f"✔ Output file {output_path} already exists, skipping creation.")
        return

    print(f"⬇ Creating modal Zarr dataset at {output_path}...")

    centroid_df = get_all_unique_h3_centroids_df()
    all_locations = centroid_df[["latitude", "longitude"]].values
    n_nodes = len(centroid_df["h3_id"])
    n_months = 12
    phases = ["ebb", "flood"]

    dataset_name = "current_hourly"
    input_vars = list(datasets.get(dataset_name, {}).get("variables", {}).values())
    output_vars = ["angle_deg", "magnitude"]

    ds_path = copernicus_data_directory / f"{dataset_name}_subset.zarr"
    ds, indexer = load_dataset_with_indexer(dataset_name)
    lat_idx_arr, lon_idx_arr = indexer.query_batch(all_locations[:, 0], all_locations[:, 1])
    latlon_indices = np.column_stack((lat_idx_arr, lon_idx_arr))

    times = ds.time.values
    bin_edges = {var: np.array(bins[dataset_name][var]["bin_edges"]) for var in output_vars}

    # Create empty Zarr dataset
    empty = xr.Dataset(
        {
            **{
                var: (("h3_id", "month", "phase"), np.full((n_nodes, n_months, 2), -1, dtype=np.int8))
                for var in output_vars
            },
            "confidence": (("h3_id", "month", "phase"), np.full((n_nodes, n_months, 2), np.nan, dtype=np.float32))
        },
        coords={
            "h3_id": ("h3_id", centroid_df["h3_id"].values),
            "month": ("month", np.arange(1, n_months + 1)),
            "phase": ("phase", phases)
        }
    )
    empty.to_zarr(output_path, mode="w")
    zarr_store = zarr.open(str(output_path), mode='a')

    # Parallel batch processing
    tasks = []
    with ProcessPoolExecutor() as executor:
        for start in range(0, n_nodes, batch_size):
            end = min(start + batch_size, n_nodes)
            latlon_batch = latlon_indices[start:end]

            future = executor.submit(
                process_batch_current,
                start, end, latlon_batch,
                ds_path,
                times,
                bin_edges,
                input_vars,
                output_vars,
                n_months,
                phases
            )
            tasks.append(future)

        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="✅ Writing", unit="batch"):
            start, end, out_vars_arr, confidence_arr = fut.result()
            for var in output_vars:
                zarr_store[var][start:end, :, :] = out_vars_arr[var]
            zarr_store["confidence"][start:end, :, :] = confidence_arr

    print(f"✅ Modal Zarr dataset written to {output_path}")


def main():
    AOI_index = 0  # Index of the AOI to use, can be changed or passed via CLI

    bbox = list(AOIS[AOI_index]["bounds"])  # Use Europe bounding box
    print(f"Using AOI: {AOIS[AOI_index]['name']} with bounds {bbox}")

    hourly_datasets = ["wind_hourly", "current_hourly"]

    for dataset_name in hourly_datasets:
        fetch_and_index_marine_dataset(dataset_name, bbox)

    bins = compute_all_bins_to_json()

    create_wind_modal_zarr(bins)
    create_current_modal_zarr(bins)

    analyse_confidence()

if __name__ == "__main__":
    main()
