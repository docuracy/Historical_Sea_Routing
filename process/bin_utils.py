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


def freedman_diaconis_bins(data, max_bins=15):
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([0.0, 1.0])

    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return np.linspace(np.min(data), np.max(data), num=3)

    bin_width = 2 * iqr * data.size ** (-1 / 3)
    if bin_width == 0:
        return np.linspace(np.min(data), np.max(data), num=3)

    bins = np.arange(np.min(data), np.max(data) + bin_width, bin_width)
    if len(bins) > max_bins:
        bins = np.linspace(np.min(data), np.max(data), max_bins + 1)
    return bins


def compute_variable_bins_sampled(
        ds_path: Path,
        wave_types: list[str] = ["WW", "SW1"],
        max_bins: int = 7,
        samples_per_variable: int = 10_000,
        random_seed: int = 42,
):
    rng = np.random.default_rng(random_seed)
    ds = xr.open_zarr(ds_path, consolidated=True)
    results = {}

    dims = ds[f"VHM0_{wave_types[0]}"].dims
    dim_sizes = {dim: ds[f"VHM0_{wave_types[0]}"].sizes[dim] for dim in dims}
    idx_choices = {
        dim: rng.integers(0, dim_sizes[dim], size=samples_per_variable)
        for dim in dims
    }

    for wt in wave_types:
        mag = ds[f"VHM0_{wt}"].isel({dim: xr.DataArray(idx_choices[dim], dims="sample") for dim in dims}).values
        dir_deg = ds[f"VMDR_{wt}"].isel({dim: xr.DataArray(idx_choices[dim], dims="sample") for dim in dims}).values
        dir_rad = np.deg2rad(dir_deg)

        u_comp = -mag * np.sin(dir_rad)
        v_comp = -mag * np.cos(dir_rad)

        for comp_name, comp_data in [(f"{wt.lower()}_u", u_comp), (f"{wt.lower()}_v", v_comp)]:
            comp_clean = comp_data[np.isfinite(comp_data)]
            if comp_clean.size < 10:
                print(f"⚠️ Too few valid values for {comp_name}, skipping.")
                continue

            bins = freedman_diaconis_bins(comp_clean, max_bins=max_bins)
            if bins.size < 2:
                print(f"⚠️ Failed to compute bins for {comp_name}, skipping.")
                continue

            midpoints = 0.5 * (bins[:-1] + bins[1:])
            results[comp_name] = {
                "bin_count": bins.size,
                "bins": bins.tolist(),
                "midpoints": midpoints.tolist(),
            }

    return results


def compute_all_bins_to_json(output_filename="copernicus_variable_bins.json"):
    # Load from output_path if it exists
    output_path = copernicus_data_directory / output_filename
    if output_path.exists():
        print(f"✔ Output file {output_path} already exists, loading previous results.")
        with open(output_path, "r") as f:
            return json.load(f)

    all_results = {}

    ds_name = "waves_hourly"
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
    # Create an output array initialized to nan_sentinel
    bin_indices = np.full(values.shape, nan_sentinel, dtype=int)

    # Mask for valid (non-NaN) values
    valid_mask = ~np.isnan(values)

    # Only digitize valid values
    bin_indices[valid_mask] = np.digitize(values[valid_mask], bin_edges) - 1  # zero-based

    return bin_indices
