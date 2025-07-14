import xarray as xr

# Original dataset path (with unwanted 'depth' dimension)
source_path = "/home/stephen/PycharmProjects/Historical_Sea_Routing/process/data/copernicus/current_hourly_subset.zarr"
cleaned_path = source_path.replace(".zarr", "_surface.zarr")

# Open lazily
ds = xr.open_zarr(source_path, consolidated=True)

# Use ds.sizes instead of ds.dims to avoid the FutureWarning
if "depth" in ds.sizes and ds.sizes["depth"] == 1:
    ds_surface = ds.isel(depth=0).drop_vars("depth")

    # ✅ Force Zarr v2 output to avoid codec issues
    ds_surface.to_zarr(cleaned_path, mode="w", consolidated=True, encoding={}, zarr_version=2)

    print(f"✅ Surface-only dataset written to: {cleaned_path}")
else:
    raise ValueError("Dataset has no 'depth' dimension of size 1 — not safe to proceed.")