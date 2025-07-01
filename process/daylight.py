from datetime import date, datetime
import numpy as np
import xarray as xr
from astral import LocationInfo
from astral.sun import sun, elevation
from tqdm import tqdm
import concurrent.futures

from process.config import copernicus_data_directory
from process.copernicus_fetch_data import compute_chunk_sizes


def monthly_avg_daylight_ratio(lat, year, month):
    loc = LocationInfo(latitude=lat, longitude=0)

    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    days_in_month = (next_month - date(year, month, 1)).days

    daylight_seconds_sum = 0

    for day_num in range(1, days_in_month + 1):
        current_date = date(year, month, day_num)
        try:
            s = sun(loc.observer, date=current_date)
            sunrise = s['sunrise']
            sunset = s['sunset']
            daylight_seconds = (sunset - sunrise).total_seconds()
        except Exception:
            noon = datetime(year, month, day_num, 12, 0, 0)
            elev = elevation(loc.observer, noon)
            daylight_seconds = 24 * 3600 if elev > 0 else 0

        daylight_seconds_sum += daylight_seconds

    avg_daylight_seconds = daylight_seconds_sum / days_in_month
    return avg_daylight_seconds / (24 * 3600)


def monthly_avg_daylight_ratio_wrapper(args):
    lat, year, month = args
    return monthly_avg_daylight_ratio(lat, year, month)


def build_monthly_latitude_chart_parallel(year=1500, lat_resolution=1):
    latitudes = np.arange(-90, 91, lat_resolution)
    months = np.arange(1, 13)
    ratios = np.zeros((len(latitudes), len(months)))

    args_list = [(lat, year, month) for lat in latitudes for month in months]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(monthly_avg_daylight_ratio_wrapper, args_list),
                            total=len(args_list),
                            desc="Computing daylight ratios"))

    for idx, val in enumerate(results):
        i = idx // len(months)
        j = idx % len(months)
        ratios[i, j] = val

    return latitudes, months, ratios


def compute_daylight_chunks(lat_min=-90, lat_max=90, lat_res=1, month_len=12, max_chunk_size=100):
    chunking = compute_chunk_sizes(
        aoi_bounds=(0, lat_min, 0, lat_max),  # dummy lon values, only lat matters
        resolution_degrees=lat_res,
        time_len=month_len,
        max_chunk_size=max_chunk_size
    )
    # Rename 'lat' → 'latitude', 'time' → 'month', discard 'lon'
    return {
        "latitude": chunking["lat"],
        "month": chunking["time"]
    }


def save_daylight_to_zarr(latitudes, months, ratios, filename="daylight_ratios.zarr"):
    ds = xr.Dataset(
        {
            "daylight_ratio": (("latitude", "month"), ratios)
        },
        coords={
            "latitude": latitudes,
            "month": months
        },
        attrs={
            "description": "Monthly average daylight ratio (daylight hours / 24h) by latitude",
            "year": 1500,
        }
    )

    zarr_path = copernicus_data_directory / "zarr" / filename

    # Ensure parent directory exists
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_sizes = compute_daylight_chunks(
        lat_min=latitudes.min(),
        lat_max=latitudes.max(),
        lat_res=np.diff(latitudes).mean()
    )

    ds.chunk(chunk_sizes).to_zarr(zarr_path, mode="w", consolidated=True)
    print(f"Saved Zarr dataset to {zarr_path}")


if __name__ == "__main__":
    year = 1500
    lat_resolution = 1

    latitudes, months, ratios = build_monthly_latitude_chart_parallel(year=year, lat_resolution=lat_resolution)
    save_daylight_to_zarr(latitudes, months, ratios)