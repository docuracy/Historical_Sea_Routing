from datetime import date
from pathlib import Path

# Define output directory
head_directory = Path(__file__).resolve().parent.parent.parent
copernicus_data_directory = head_directory / "process" / "geo" / "data" / "copernicus"
copernicus_data_directory.mkdir(parents=True, exist_ok=True)

# Define areas of interest (AOIs) with their geographical bounds
AOIS = [
    {
        "name": "Europe",
        # lon_min (east coast of Greenland), lat_min (Canarias), lon_max (eastern Mediterranean), lat_max (north coast of Norway)
        "bounds": (-45.00, 25.00, 37.00, 72.00),  # lon_min, lat_min, lon_max, lat_max
    },
    {
        "name": "UK-Eire",
        "bounds": (-11.0, 49.5, 2.0, 61.0),  # lon_min, lat_min, lon_max, lat_max
    },
]

# Define H3 resolution for land-adjacent zones
COASTAL_SEA_RESOLUTION = 7

# Define datasets
datasets = {
    "Bathymetry": {
        "doi": "https://doi.org/10.48670/moi-00017",
        "resolution_degrees": 0.083,
        "dataset_id": "cmems_mod_wav_anfc_0.083deg_static",
        "variables": {
            "sea floor depth": "deptho"
        }
    },
    "Wind": {
        "doi": "https://doi.org/10.48670/moi-00181",
        "resolution_degrees": 0.25,
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_P1M",
        "date_range": ("1995-01-01", "2005-01-01"),
        "variables": {
            "eastward wind": "eastward_wind",
            "northward wind": "northward_wind"
        }
    },
    "Waves": {
        "doi": "https://doi.org/10.48670/moi-00017",
        "resolution_degrees": 0.083,  # This is significantly higher resolution than the ERA5 dataset (0.5 degrees)
        "dataset_id": "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        "date_range": ("2023-01-01", "2025-01-01"),  # Only two complete years currently available (June 2025)
        "variables": {
            "stokes drift x velocity": "VSDX",
            "stokes drift y velocity": "VSDY",
            "wind wave from direction": "VMDR_WW",
            "wind wave mean period": "VTM01_WW",
            "wind wave significant height": "VHM0_WW",
            "primary swell wave from direction": "VMDR_SW1",
            "primary swell wave mean period": "VTM01_SW1",
            "primary swell wave significant height": "VHM0_SW1",
        }
    },
    "Weather": {
        "doi": "https://doi.org/10.24381/cds.f17050d7",
        "resolution_degrees": 0.25,
        "dataset_id": "reanalysis-era5-single-levels",
        "product_type": "reanalysis",
        "years": [str(year) for year in range(1940, date.today().year, 10)],  # Sample of complete years since 1940
        "months": [f"{m:02d}" for m in range(1, 13)],  # All months
        "days": [f"{d:02d}" for d in [1, 15]],  # Sample of days (1st and 15th)
        "hours": [f"{h:02d}:00" for h in range(0, 24, 6)],  # Sample of hours (every 6 hours)
        "variables": {
            "2m temperature": "2m_temperature",
            "2m dewpoint temperature": "2m_dewpoint_temperature",
            "total precipitation": "total_precipitation",
            "low cloud cover": "low_cloud_cover",
            "clear_sky_ssrd": "surface_solar_radiation_downwards_clear_sky",
            "total_ssrd": "surface_solar_radiation_downwards",
        }
    },
    "DEM": {
        "description": "Digital Elevation Model (DEM) tiles in Terrarium format from Mapzen AWS S3 bucket.",
        "zoom_level": 12,
        "source_url_template": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
    }
}


# Add label and variable name lists
def enrich_datasets(dsets):
    for name, ds_info in dsets.items():
        if name == "DEM":
            continue  # Skip enrichment for DEM
        ds_info["all_labels"] = list(ds_info.get("variables", {}).keys())
        ds_info["nc_variables"] = list(ds_info.get("variables", {}).values())
    return dsets


datasets = enrich_datasets(datasets)
