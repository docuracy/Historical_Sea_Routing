import json

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from process.config import copernicus_data_directory, screenshots_directory

zarr_path = copernicus_data_directory / "zarr" / "environment_modal_monthly.zarr"
output_path = copernicus_data_directory / "copernicus_modal_monthly_confidence_summary.json"
ds = xr.open_zarr(zarr_path, consolidated=False)

conf = ds["confidence"].values.flatten()
conf = conf[~np.isnan(conf)]  # remove NaNs

print(f"Confidence summary statistics:")
print(f"  Count: {len(conf)}")
print(f"  Min: {conf.min():.4f}")
print(f"  Max: {conf.max():.4f}")
print(f"  Mean: {conf.mean():.4f}")
print(f"  Median: {np.median(conf):.4f}")
print(f"  25th percentile: {np.percentile(conf, 25):.4f}")
print(f"  75th percentile: {np.percentile(conf, 75):.4f}")

plt.figure(figsize=(8, 4))
plt.hist(conf, bins=50, color='steelblue', alpha=0.8)
plt.title("Histogram of confidence values")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(screenshots_directory / "modal_confidence_histogram.png", dpi=150)
plt.show()

confidence = ds["confidence"]

# Convert to DataFrame for convenient grouping
df_conf = confidence.to_dataframe().reset_index()

# Group by month and calculate summary stats
monthly_summary = df_conf.groupby("month")["confidence"].agg(
    count='count',
    min='min',
    max='max',
    mean='mean',
    median='median',
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75)
)

months = monthly_summary.index
mean = monthly_summary['mean']
median = monthly_summary['median']
q25 = monthly_summary['q25']
q75 = monthly_summary['q75']

plt.figure(figsize=(10, 6))
plt.plot(months, mean, label='Mean confidence', marker='o')
plt.plot(months, median, label='Median confidence', marker='s')

# Fill between 25th and 75th percentiles
plt.fill_between(months, q25, q75, color='steelblue', alpha=0.3, label='25th-75th percentile')

plt.xticks(months)
plt.xlabel("Month")
plt.ylabel("Confidence")
plt.title("Monthly Confidence Statistics")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(screenshots_directory / "modal_confidence_monthly_summary.png", dpi=150)
plt.show()

print(monthly_summary)

# Convert DataFrame to dictionary for JSON serialization
summary_dict = monthly_summary.round(4).to_dict(orient='index')

with open(output_path, "w") as f:
    json.dump(summary_dict, f, indent=2)

print(f"Monthly confidence summary saved to {output_path}")
