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
from scipy.spatial import cKDTree
from tqdm import tqdm

from process.bin_utils import compute_all_bins_to_json, get_binned_data_for_components_dask
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
        # Determine ebb (0) vs flood (1)
        # zos.shape == (time, lat, lon)
        delta_zos = zos.diff(dim="time", label="upper")  # shape: (time-1, lat, lon)

        # Shift result back to align with original time index
        delta_zos_full = xr.concat([delta_zos.isel(time=0)*np.nan, delta_zos], dim="time")

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


def process_batch_current(start, end, latlon_batch, derived_vars_path, times_np, bin_edges, input_vars, output_vars,
                       n_months, phases):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from collections import Counter

    ds = xr.open_zarr(derived_vars_path)
    derived_vars = compute_angular_components(ds, input_vars, output_vars, phases)

    batch_size_actual = end - start

    # Bin angle and magnitude
    binned_vars = get_binned_data_for_components_dask(derived_vars, latlon_batch, bin_edges)

    # Fetch phase values (0=ebb, 1=flood) from derived_vars["phase"]
    # shape: (time, batch_size_actual)
    phase_array = np.stack([
        derived_vars["phase"][:, lat, lon].values
        for lat, lon in latlon_batch
    ], axis=1)  # shape: (time, batch_size_actual)
    phase_array = phase_array.T.flatten()  # shape: time * batch_size_actual

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
        batch_size=200,
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
