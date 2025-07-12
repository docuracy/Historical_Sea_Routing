# fetch_environment_data.py

'''

COPERNICUS MARINE

1. Fetch monthly data from https://doi.org/10.48670/moi-00181 ("cmems_obs-wind_glo_phy_my_l4_P1M") to determine the **maximum standard deviation of differences** for both eastward and northward wind components. Use only the `eastward_wind_sdd` and `northward_wind_sdd` variables, for the period 2023–2024 inclusive.

2. Use these maximum variability estimates to determine the **minimum required sample size** to estimate mean wind vectors with 95% confidence, assuming a normal distribution and a specified margin of error.

3. Compute a well-distributed set of sampling time points (hours throughout the day) for the first 27 days of each month in 2023–2024, using a modulo–prime number scheme to avoid temporal clustering.

4. For each time point, fetch the corresponding **hourly data** from https://doi.org/10.48670/moi-00185 ("cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H") in Zarr format. Use only the `eastward_wind` and `northward_wind` variables.

5. Combine all hourly samples from both waves_hourly and wind_hourly into a single Zarr file indexed on latitude and longitude for downstream processing.

'''
import json
import logging
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from copernicusmarine import subset
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from process.bin_utils import compute_all_bins_to_json, get_binned_data_for_components_dask
from process.config import datasets, AOIS, copernicus_data_directory
from process.sea_graph import get_all_unique_h3_centroids_df

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


def compute_monthly_joint_modes(binned_vars, times, centroid_df):
    n_months = 12
    n_nodes = len(centroid_df)
    months = times.month.values
    var_names = list(binned_vars.keys())

    modal_values = {
        f"{var}_bin_mode": np.full((n_nodes, n_months), -1, dtype=np.int8)
        for var in var_names
    }

    for idx in range(n_nodes):
        node_bins = np.stack([binned_vars[var][:, idx] for var in var_names], axis=1)
        valid_mask = ~(node_bins == -1).any(axis=1)
        node_bins = node_bins[valid_mask]
        node_months = months[valid_mask]

        if node_bins.shape[0] == 0:
            continue

        bin_tuples = [tuple(row) for row in node_bins]

        for m in range(1, n_months + 1):
            month_bins = [bt for bt, mon in zip(bin_tuples, node_months) if mon == m]
            if month_bins:
                joint_mode, _ = Counter(month_bins).most_common(1)[0]
                for vi, var in enumerate(var_names):
                    modal_values[f"{var}_bin_mode"][idx, m - 1] = joint_mode[vi]

    return modal_values


def compute_monthly_pairwise_correlations(binned_vars, times, centroid_df, method="spearman"):
    from scipy.stats import spearmanr, pearsonr

    months = times.month.values
    var_names = list(binned_vars.keys())
    results = []

    for m in range(1, 13):
        month_mask = (months == m)

        if not month_mask.any():
            continue

        binned_month = {
            var: binned_vars[var][month_mask, :] for var in var_names
        }

        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                if i >= j:
                    continue

                vals1 = binned_month[var1]
                vals2 = binned_month[var2]

                # Flatten and filter invalid values
                flat1 = vals1.ravel()
                flat2 = vals2.ravel()
                valid = (flat1 != -1) & (flat2 != -1)

                if valid.sum() < 30:
                    continue

                x, y = flat1[valid], flat2[valid]
                corr_func = spearmanr if method == "spearman" else pearsonr
                corr, _ = corr_func(x, y)
                results.append((m, var1, var2, corr))

    return results


def fetch_and_index_marine_dataset(dataset_name, bbox):
    dataset_info = datasets.get(dataset_name)
    if dataset_info is None:
        raise ValueError(f"Dataset '{dataset_name}' is not defined in the datasets configuration.")

    output_file = copernicus_data_directory / f"{dataset_name.lower().replace(' ', '_')}_subset.zarr"

    output_file = Path(output_file)
    if output_file.is_dir():
        print(f"✔ {dataset_name} already exists at {output_file}, skipping download.")
        return output_file

    dataset_id = dataset_info["dataset_id"]
    variable_list = list(dataset_info["variables"].values())
    date_range = dataset_info.get("date_range", None)

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

        ds_downloaded = xr.open_zarr(output_file, consolidated=True)
        lats = ds_downloaded["latitude"].values
        lons = ds_downloaded["longitude"].values

        indexer = LatLonIndexer(lats, lons)
        index_path = output_file.parent / (output_file.name + "_latlon_index.npz")
        indexer.save(index_path)
        print(f"🗺️ LatLon index saved to {index_path}")

        print(f"✅ Final Zarr for {dataset_name} written to {output_file}")

    except Exception as e:
        print(f"❌ Failed to process {dataset_name}: {e}", file=sys.stderr)


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


def batched(df, batch_size):
    """Yield successive batches from DataFrame"""
    for i in range(0, len(df), batch_size):
        yield i, df.iloc[i:i + batch_size]


def compute_wave_components(ds):
    def components(mag, dir_deg):
        dir_rad = np.deg2rad(dir_deg)
        u = -mag * np.sin(dir_rad)
        v = -mag * np.cos(dir_rad)
        return u, v

    ww_u, ww_v = components(ds["VHM0_WW"], ds["VMDR_WW"])
    sw1_u, sw1_v = components(ds["VHM0_SW1"], ds["VMDR_SW1"])

    return {
        "ww_u": ww_u,
        "ww_v": ww_v,
        "sw1_u": sw1_u,
        "sw1_v": sw1_v,
    }


def process_batch(start, end, latlon_batch, derived_vars_path, times_np, bin_edges, output_vars, n_months):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from collections import Counter

    ds = xr.open_zarr(derived_vars_path)
    derived_vars = compute_wave_components(ds)  # Must be re-run inside each process

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


def create_modal_zarr_from_waves(
        bins,
        batch_size=200,
        output_filename="environment_modal_monthly.zarr",
):
    output_path = copernicus_data_directory / "zarr" / output_filename
    if output_path.exists():
        print(f"✔ Output file {output_path} already exists, skipping creation.")
        return

    print(f"⬇ Creating modal Zarr dataset at {output_path}...")

    centroid_df = get_all_unique_h3_centroids_df()
    n_nodes = len(centroid_df["h3_id"])
    n_months = 12

    dataset_name = "waves_hourly"
    output_vars = ["ww_u", "ww_v", "sw1_u", "sw1_v"]

    ds_path = copernicus_data_directory / f"{dataset_name}_subset.zarr"
    if not ds_path.exists():
        print(f"❌ Dataset not found at {ds_path}, aborting.")
        return

    ds, indexer = load_dataset_with_indexer(dataset_name)
    all_locations = centroid_df[["latitude", "longitude"]].values
    lat_idx_arr, lon_idx_arr = indexer.query_batch(all_locations[:, 0], all_locations[:, 1])
    latlon_indices = np.column_stack((lat_idx_arr, lon_idx_arr))

    times = ds.time.values
    bin_edges = {var: np.array(bins[dataset_name][var]["bins"]) for var in output_vars}

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
                process_batch,
                start, end, latlon_batch,
                ds_path,
                times,
                bin_edges, output_vars, n_months
            )
            tasks.append(future)

        for fut in tqdm(as_completed(tasks), total=len(tasks), desc="✅ Writing", unit="batch"):
            start, end, out_vars_arr, confidence_arr = fut.result()
            for var in output_vars:
                zarr_store[var][start:end, :] = out_vars_arr[var]
            zarr_store["confidence"][start:end, :] = confidence_arr

    print(f"✅ Modal Zarr dataset written to {output_path}")


def plot_correlation_heatmap(corr_matrix, title="Variable Correlation Clustermap"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white")

    g = sns.clustermap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        figsize=(14, 12),
        annot_kws={"size": 8},
    )

    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    plt.title(title, y=1.05)

    output_path = copernicus_data_directory / "figures" / (title.replace(" ", "_").lower() + ".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved correlation heatmap to: {output_path}")

    plt.show()


def compute_wind_proxy_coefficients(sampled_ds):
    """
    Given the sampled dataset with wave and wind variables, compute
    linear regression coefficients to approximate wind components
    from wave variables waves_hourly_VSDX and waves_hourly_VSDY.

    Returns:
      dict with keys 'u' and 'v' each mapping to (coef, intercept),
      or None if required variables are missing.
    """
    output_path = copernicus_data_directory / "wind_proxy_coefficients.json"
    if output_path.exists():
        print(f"✔ Wind proxy coefficients already exist at {output_path}, loading previous results.")
        with open(output_path, 'r') as f:
            return json.load(f)

    required_vars = ['wind_hourly_eastward_wind', 'wind_hourly_northward_wind',
                     'waves_hourly_VSDX', 'waves_hourly_VSDY']

    if not all(var in sampled_ds for var in required_vars):
        return None

    # Flatten all time and h3_id dimensions
    wind_u = sampled_ds['wind_hourly_eastward_wind'].values.ravel()
    wind_v = sampled_ds['wind_hourly_northward_wind'].values.ravel()
    vsdx = sampled_ds['waves_hourly_VSDX'].values.ravel()
    vsdy = sampled_ds['waves_hourly_VSDY'].values.ravel()

    # Mask out NaNs
    valid_mask = (~np.isnan(wind_u) & ~np.isnan(wind_v) &
                  ~np.isnan(vsdx) & ~np.isnan(vsdy))

    if np.sum(valid_mask) == 0:
        return None

    reg_u = LinearRegression().fit(vsdx[valid_mask].reshape(-1, 1), wind_u[valid_mask])
    reg_v = LinearRegression().fit(vsdy[valid_mask].reshape(-1, 1), wind_v[valid_mask])

    wind_proxy_coefficients = {
        'u': (reg_u.coef_[0], reg_u.intercept_),
        'v': (reg_v.coef_[0], reg_v.intercept_)
    }

    print(f"Computed wind proxy coefficients: {wind_proxy_coefficients}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(wind_proxy_coefficients, f, indent=2)
        print(f"Saved wind proxy coefficients to {output_path}")

    return wind_proxy_coefficients


def sample_ds_at_centroids_with_indexer(ds, indexer, centroid_df, batch_size=1000):
    """
    Sample dataset values at centroid locations using a precomputed LatLonIndexer.
    Assumes the indexer knows how to map lat/lon to ds indices.
    """
    centroid_points = centroid_df[['latitude', 'longitude']].values
    n = len(centroid_df)

    sampled_vars = {var: [] for var in ds.data_vars}

    # Flags for vector components
    compute_ww_uv = 'VHM0_WW' in ds and 'VMDR_WW' in ds
    compute_sw_uv = 'VHM0_SW1' in ds and 'VMDR_SW1' in ds
    if compute_ww_uv:
        sampled_vars['ww_u'] = []
        sampled_vars['ww_v'] = []
    if compute_sw_uv:
        sampled_vars['sw_u'] = []
        sampled_vars['sw_v'] = []

    for start in tqdm(range(0, n, batch_size), desc="Sampling centroids batches"):
        end = min(start + batch_size, n)
        batch_points = centroid_points[start:end]

        # Use the indexer to get indices (lat_idx, lon_idx) for all batch_points
        lat_idxs, lon_idxs = indexer.query_batch(batch_points[:, 0], batch_points[:, 1])

        batch_data = {}
        for var in ds.data_vars:
            var_da = ds[var]
            batch_samples = np.stack([
                var_da.isel(latitude=lat_idx, longitude=lon_idx).values
                for lat_idx, lon_idx in zip(lat_idxs, lon_idxs)
            ]).T  # shape: (time, batch_size)
            batch_data[var] = batch_samples
            sampled_vars[var].append(batch_samples)

        # Compute WW U/V components
        if compute_ww_uv:
            h = batch_data['VHM0_WW']
            d_rad = np.deg2rad(batch_data['VMDR_WW'])
            ww_u = -h * np.sin(d_rad)
            ww_v = -h * np.cos(d_rad)
            sampled_vars['ww_u'].append(ww_u)
            sampled_vars['ww_v'].append(ww_v)

        # Compute SW U/V components
        if compute_sw_uv:
            h = batch_data['VHM0_SW1']
            d_rad = np.deg2rad(batch_data['VMDR_SW1'])
            sw_u = -h * np.sin(d_rad)
            sw_v = -h * np.cos(d_rad)
            sampled_vars['sw_u'].append(sw_u)
            sampled_vars['sw_v'].append(sw_v)

    # Drop original vars used for derived components if we've computed them
    if compute_ww_uv:
        for var in ['VHM0_WW', 'VMDR_WW']:
            sampled_vars.pop(var, None)
    if compute_sw_uv:
        for var in ['VHM0_SW1', 'VMDR_SW1']:
            sampled_vars.pop(var, None)

    final_vars = {}
    for var, batches in sampled_vars.items():
        combined = np.concatenate(batches, axis=1)
        final_vars[var] = (("time", "h3_id"), combined)

    sampled_ds = xr.Dataset(
        final_vars,
        coords={
            "time": ds['time'].values,
            "h3_id": centroid_df["h3_id"].values
        }
    )
    return sampled_ds


def flatten(ds):
    """
    Flatten Dataset with dims (time, h3_id) and variables into a pandas DataFrame
    with index (h3_id, time) and columns for variables.
    """
    df = ds.to_array(dim="variable").to_dataframe(name="value").reset_index()
    pivot_df = df.pivot(index=["h3_id", "time"], columns="variable", values="value")
    return pivot_df


def run_correlation_analysis(hourly_datasets, drop_na=True, sample_size=1000):
    output_path = copernicus_data_directory / "correlation_matrix.json"
    if output_path.exists():
        print(f"✔ Correlation matrix already exists at {output_path}, loading previous results.")
        return pd.read_json(output_path, orient="split")

    centroid_df = get_all_unique_h3_centroids_df()
    centroid_df_sample = centroid_df.sample(sample_size, random_state=42).reset_index(drop=True)

    datasets = []
    for dataset_name in tqdm(hourly_datasets, desc="Loading datasets"):
        ds, indexer = load_dataset_with_indexer(dataset_name)
        sampled_ds = sample_ds_at_centroids_with_indexer(ds, indexer, centroid_df_sample)
        df = flatten(sampled_ds)
        df = df.add_prefix(f"{dataset_name}_")
        datasets.append(df)

    combined = pd.concat(datasets, axis=1)

    if drop_na:
        combined = combined.dropna()

    corr_matrix = combined.corr()

    output_path = copernicus_data_directory / "correlation_matrix.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_json(output_path, orient="split")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(corr_matrix.round(2))

    plot_correlation_heatmap(corr_matrix)

    return corr_matrix


def main():
    AOI_index = 0  # Index of the AOI to use, can be changed or passed via CLI

    bbox = list(AOIS[AOI_index]["bounds"])  # Use Europe bounding box
    print(f"Using AOI: {AOIS[AOI_index]['name']} with bounds {bbox}")

    hourly_datasets = ["wind_hourly", "waves_hourly"]

    for dataset_name in hourly_datasets:
        fetch_and_index_marine_dataset(dataset_name, bbox)

    sampled_data = run_correlation_analysis(hourly_datasets, drop_na=False)

    wind_proxy_coefficients = compute_wind_proxy_coefficients(sampled_data)

    bins = compute_all_bins_to_json()

    create_modal_zarr_from_waves(bins)


if __name__ == "__main__":
    main()
