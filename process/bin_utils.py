import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
from dask import delayed, compute
from tqdm import tqdm

from process.config import datasets, copernicus_data_directory


def process_node(idx, latlon_batch, derived_vars_var, bin_edges, n_times):
    lat_idx, lon_idx = latlon_batch[idx]
    values = derived_vars_var[:, lat_idx, lon_idx]
    if np.all(np.isnan(values)):
        return idx, np.full(n_times, -1, dtype=np.int16)
    binned = assign_bin_index(values, bin_edges)
    return idx, binned


def get_binned_data_for_components_dask(derived_vars, latlon_batch, bins_dict):

    binned_vars = {}
    for var, da in derived_vars.items():
        bin_edges = bins_dict[var]
        delayed_bins = [
            delayed(assign_bin_index)(da[:, lat_idx, lon_idx].data, bin_edges)
            for lat_idx, lon_idx in latlon_batch
        ]
        binned = compute(*delayed_bins)
        stacked = np.stack(binned, axis=0).T.astype(np.int16)
        binned_vars[var] = stacked

    return binned_vars


def get_binned_data_for_derived_vars(
        derived_vars: dict[str, np.ndarray],
        times: np.ndarray,
        latlon_batch: np.ndarray,
        bins_dict,
        max_workers=8,
):
    n_times = len(times)
    n_nodes = latlon_batch.shape[0]

    binned_vars = {
        var: np.full((n_times, n_nodes), -1, dtype=np.int16)
        for var in derived_vars.keys()
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for var in derived_vars:
            futures = {
                executor.submit(process_node, idx, latlon_batch, derived_vars[var], bins_dict[var], n_times): idx
                for idx in range(n_nodes)
            }
            for future in as_completed(futures):
                idx, binned = future.result()
                binned_vars[var][:, idx] = binned

    return binned_vars, times


def get_binned_data_for_nodes(ds, indexer, centroid_df, var_names, bins_dict, batch_size=500):
    n_nodes = len(centroid_df)
    times = ds.time.to_index()
    binned_vars = {
        var: np.full((len(times), n_nodes), -1, dtype=np.int16)
        for var in var_names
    }

    locations = centroid_df[["lat", "lon"]].values
    latlons = indexer.query(locations[:, 0], locations[:, 1])  # list of (lat_idx, lon_idx)

    for batch_start in tqdm(range(0, n_nodes, batch_size), desc="Binning variables"):
        batch_end = min(batch_start + batch_size, n_nodes)
        batch_indices = range(batch_start, batch_end)

        for var in var_names:
            bin_edges = bins_dict[var]

            for idx in batch_indices:
                lat_idx, lon_idx = latlons[idx]
                values = ds[var][:, lat_idx, lon_idx].values
                if not np.all(np.isnan(values)):
                    binned = assign_bin_index(values, bin_edges)
                    binned_vars[var][:, idx] = binned

    return binned_vars, times


def compute_variable_bins_sampled(
    ds_path: Path,
    max_magnitude_bins: int = 8,
    samples: int = 10_000,
    random_seed: int = 42,
):
    import xarray as xr
    import numpy as np

    rng = np.random.default_rng(random_seed)
    ds = xr.open_zarr(ds_path, consolidated=True)

    # Identify variable names
    if "uo" in ds and "vo" in ds:
        u_var, v_var = "uo", "vo"
    elif "eastward_wind" in ds and "northward_wind" in ds:
        u_var, v_var = "eastward_wind", "northward_wind"
    else:
        raise ValueError("No recognised u/v vector variables found.")

    dims = ds[u_var].dims
    sizes = ds[u_var].sizes

    # Sample random indices
    indices = {
        dim: xr.DataArray(rng.integers(0, sizes[dim], size=samples), dims="sample")
        for dim in dims
    }

    print(f"🔍 Sampling {samples} vector pairs for {ds_path}...")

    u = ds[u_var].isel(indices).values
    v = ds[v_var].isel(indices).values

    # Compute magnitude
    magnitude = np.sqrt(u**2 + v**2)

    # Fixed directional bins (8 sectors)
    direction_bin_edges = np.linspace(0, 360, 9)  # 8 bins
    direction_bin_centres = (direction_bin_edges[:-1] + direction_bin_edges[1:]) / 2

    # Compute adaptive magnitude bins
    def freedman_diaconis_bins(data, max_bins=7):
        data = data[np.isfinite(data)]
        if len(data) < 10:
            return None
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1 / 3))
        if bin_width == 0:
            return None
        bins = int(np.ceil((data.max() - data.min()) / bin_width))
        bins = min(bins, max_bins)
        return np.histogram_bin_edges(data, bins=bins)

    magnitude_bins = freedman_diaconis_bins(magnitude, max_bins=max_magnitude_bins)
    if magnitude_bins is None or len(magnitude_bins) < 2:
        print("⚠️ Failed to compute valid magnitude bins.")
        return None

    magnitude_bin_centres = 0.5 * (magnitude_bins[:-1] + magnitude_bins[1:])

    return {
        "angle_deg": {
            "bin_edges": direction_bin_edges.tolist(),
            "midpoints": direction_bin_centres.tolist(),
            "bin_count": 8
        },
        "magnitude": {
            "bin_edges": magnitude_bins.tolist(),
            "midpoints": magnitude_bin_centres.tolist(),
            "bin_count": len(magnitude_bins) - 1
        }
    }


def compute_all_bins_to_json(output_filename="copernicus_variable_bins.json"):
    # Load from output_path if it exists
    output_path = copernicus_data_directory / output_filename
    if output_path.exists():
        print(f"✔ Output file {output_path} already exists, loading previous results.")
        with open(output_path, "r") as f:
            return json.load(f)

    all_results = {}

    ds_names = ["current_hourly", "wind_hourly"]

    for ds_name in ds_names:
        dataset_info = datasets.get(ds_name)
        if dataset_info is None:
            raise ValueError(f"Dataset '{ds_name}' is not defined in the datasets configuration.")

        ds_file = copernicus_data_directory / f"{ds_name}_subset.zarr"

        result = compute_variable_bins_sampled(ds_file)
        all_results[ds_name] = result

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Bin definitions saved to {output_path}")

    return all_results


def assign_bin_index(values, bin_edges, nan_sentinel=-1):
    values = np.asarray(values)
    bin_indices = np.full(values.shape, nan_sentinel, dtype=int)

    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]

    # Compute bin indices
    indices = np.digitize(valid_values, bin_edges) - 1  # zero-based

    # Clamp values outside the bin range (values might exceed those in the sampled subset used to create the bins)
    indices = np.clip(indices, 0, len(bin_edges) - 2)

    bin_indices[valid_mask] = indices
    return bin_indices
