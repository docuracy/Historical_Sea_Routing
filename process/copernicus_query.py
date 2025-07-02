import logging

import xarray as xr
from matplotlib import pyplot as plt

from config import datasets, copernicus_data_directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.getLogger('h5py._conv').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

daylight_zarr_path = copernicus_data_directory / "zarr" / "daylight_ratios.zarr"


def get_daylight_ratio(latlon: tuple, month: int = 0) -> float:
    try:
        ds = xr.open_zarr(daylight_zarr_path, consolidated=True)

        # Ensure input coordinates are within valid range
        lat, lon = latlon
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude {lat} out of range")

        # Interpolate (Zarr-backed arrays are lazy, so we need to compute)
        interpolated = ds.daylight_ratio.interp(latitude=lat, month=month)
        value = interpolated.compute().item()  # Forces actual computation

        return float(value)
    except Exception as e:
        logger.error(f"Error retrieving daylight ratio for {latlon} in month {month}: {e}")
        return 0.5


def estimate_visibility(latlon: tuple,
                        temp_depression: xr.DataArray,
                        precip_rate: xr.DataArray,
                        low_cloud_cover: xr.DataArray,
                        month: int = 0,
                        max_visibility_m: float = 50000.0,  # Max visibility 50 km
                        min_visibility_m: float = 10.0,  # Min visibility 10 m, avoid 0
                        td_sensitivity_factor: float = 10000.0,  # Metres * degrees C for temp depression
                        td_epsilon: float = 0.5,  # Small constant to prevent div by zero for temp depression
                        precip_sensitivity_factor: float = 5000.0,  # Metres * (kg/m^2) for precipitation
                        precip_epsilon: float = 0.01,  # Small constant to prevent div by zero for precip
                        lcc_threshold: float = 0.8,
                        lcc_sensitivity_factor: float = 5000.0,
                        lcc_epsilon: float = 0.01
                        ) -> xr.DataArray:
    """
    Estimates atmospheric visibility in metres based on temperature depression,
    precipitation rate, cloud base height, and low cloud cover.
    """

    # Ensure all inputs are positive for calculations where appropriate
    temp_depression = temp_depression.where(temp_depression > 0, 0)  # Non-negative temp depression
    precip_rate = precip_rate.where(precip_rate > 0, 0)  # Non-negative precip rate

    # 1. Visibility from Humidity (Temperature Depression)
    # Lower temp_depression (higher humidity/fog) -> lower visibility
    # Add td_epsilon to prevent division by zero or extremely large values when Td is near zero.
    vis_from_humidity = td_sensitivity_factor / (temp_depression + td_epsilon)

    # 2. Visibility from Precipitation
    # Higher precip_rate -> lower visibility
    # Add precip_epsilon to prevent division by zero or extremely large values when precip is near zero.
    vis_from_precip = precip_sensitivity_factor / (precip_rate + precip_epsilon)

    # 3. Visibility from Low Cloud Cover (LCC)
    # Higher LCC (closer to 1) -> lower visibility
    lcc_impact_inverse = 1 / (low_cloud_cover + lcc_epsilon)
    vis_from_lcc = lcc_impact_inverse * lcc_sensitivity_factor

    # 4. Combine all factors by taking the minimum
    combined_visibility = xr.ufuncs.minimum(
        xr.ufuncs.minimum(vis_from_humidity, vis_from_precip),
        vis_from_lcc
    )

    # 5. Clip the final result to the defined min and max visibility
    final_visibility = combined_visibility.clip(min=min_visibility_m, max=max_visibility_m)

    return final_visibility.rename("visibility_m")


def weather(latlon: tuple, month: int):
    """Return visibility estimates (m) at a point, optionally filtered by month and aggregated."""
    zarr_file = copernicus_data_directory / "zarr" / "weather.zarr"

    ds = xr.open_zarr(zarr_file, consolidated=True)

    try:
        # Subset at nearest point
        subset = ds[["tp", "t2m", "d2m", "lcc"]].sel(
            latitude=latlon[0], longitude=latlon[1], method="nearest"
        )

        # Rename time coordinate if needed
        if "valid_time" in subset.coords:
            subset = subset.rename({"valid_time": "time"})

        subset = subset.sel(time=subset["time.month"] == month)

        # Temperature depression
        t2m_c = subset["t2m"] - 273.15
        d2m_c = subset["d2m"] - 273.15
        temp_depression = (t2m_c - d2m_c).rename("temp_depression")

        precip_rate = subset["tp"].rename("precip_rate")
        low_cloud_cover = subset["lcc"].rename("low_cloud_cover")

        # Estimate visibility
        visibility = estimate_visibility(
            latlon=latlon,
            temp_depression=temp_depression,
            precip_rate=precip_rate,
            low_cloud_cover=low_cloud_cover,
            month=month,
        )
        visibility.name = "visibility_m"

        return int(visibility.mean().compute().item())

    except Exception as e:
        logger.error(f"Error during visibility computation: {e}", exc_info=True)
        return None


def query_all_datasets(latlon: tuple, month: int = 0) -> dict:
    """
    Query all datasets at a given lat/lon point.
    If month=0, use annual average dataset if available, else time=0 from original.
    If month in 1..12, use monthly average dataset if available, else fallback as above.
    """
    results = {}

    for dataset_name, ds_info in datasets.items():
        try:
            if dataset_name == "Weather":
                results["visibility_m"] = weather(latlon, month)
            else:
                base_name = dataset_name.lower().replace(" ", "_")

                zarr_file = copernicus_data_directory / "zarr" / f"{base_name}.zarr"

                vars_nc = list(ds_info["variables"].values())

                with xr.open_zarr(zarr_file, consolidated=True) as ds:
                    if "month" in ds.dims and month > 0:
                        # Monthly averages dataset: select month along with lat/lon
                        point_data = ds[vars_nc].sel(latitude=latlon[0], longitude=latlon[1], month=month,
                                                     method="nearest")
                    elif "time" in ds.dims:
                        # Annual average or base dataset with time dim: select first time step plus lat/lon
                        point_data = ds[vars_nc].sel(latitude=latlon[0], longitude=latlon[1], time=ds.time[0],
                                                     method="nearest")
                    else:
                        # Dataset without time dimension: just spatial selection
                        point_data = ds[vars_nc].sel(latitude=latlon[0], longitude=latlon[1], method="nearest")

                    # Extract scalar values
                    for nc_var in vars_nc:
                        results[nc_var] = float(point_data[nc_var].values)

        except Exception as e:
            logger.error(f"Error querying dataset '{dataset_name}': {e}")

    results["daylight_ratio"] = get_daylight_ratio(latlon, month)

    return results


def query_all_months(latlon: tuple) -> list[dict]:
    results = []
    for m in range(1, 13):
        try:
            res = query_all_datasets(latlon, month=m)
            results.append(res)
        except Exception as e:
            logger.warning(f"⚠️ Skipping month {m} due to error: {e}")
            results.append(None)
    return results


def plot_monthly_visibility(results: list[dict], title="Monthly Visibility Estimate", save_path=None):
    months = range(1, 13)
    visibilities = [
        r["visibility_m"] if r is not None else None
        for r in results
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(months, visibilities, marker="o", linestyle="-", color="dodgerblue", label="Visibility (m)")
    plt.xticks(months)
    plt.xlabel("Month")
    plt.ylabel("Estimated Visibility (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    latlon = (55.155916883303675, -5.883977412991199) # POINT (-5.883977412991199 55.155916883303675)
    month = 0  # Use 0 for annual average, 1-12 for specific month, or False for graphing all months

    # TODO: Compute land visibility

    try:
        # if month == 0:
        #     results = query_all_datasets(latlon, month=0)
        #     logger.info(f"✅ Annual results:\n{results}")
        #     exit(0)
        # elif not month == False:
        #     logger.info(f"Querying monthly visibility for {latlon} in month {month}...")
        #     results = query_all_datasets(latlon, month=month)
        #     logger.info(f"✅ Monthly results for month {month}:\n{results}")
        #     exit(0)
        results = query_all_months(latlon)
        logger.info(f"✅ Monthly results:\n{results}")
        plot_monthly_visibility(results)
    except Exception as e:
        logger.error(f"Unexpected error in monthly analysis: {e}")
