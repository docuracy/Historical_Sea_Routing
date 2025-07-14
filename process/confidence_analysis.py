import json

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from process.config import copernicus_data_directory, screenshots_directory


def compute_confidence_summary(arr: np.ndarray) -> dict:
    arr = arr.flatten()
    arr = arr[~np.isnan(arr)]
    return {
        "count": len(arr),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "values": arr
    }


def plot_histogram(conf_values, ds_name, phase=None):
    plt.figure(figsize=(8, 4))
    plt.hist(conf_values, bins=50, color='steelblue', alpha=0.8)
    title = f"{ds_name.replace("_", " ").title()}: Histogram of confidence values{' (' + phase + ')' if phase else ''}"
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    suffix = f"_{phase}" if phase else ""
    plt.savefig(screenshots_directory / f"{ds_name}_modal_confidence_histogram{suffix}.png", dpi=150)
    plt.show()


def plot_monthly_summary(monthly_summary, ds_name, phase=None):
    months = monthly_summary.index
    mean = monthly_summary['mean']
    median = monthly_summary['median']
    q25 = monthly_summary['q25']
    q75 = monthly_summary['q75']

    plt.figure(figsize=(10, 6))
    plt.plot(months, mean, label='Mean confidence', marker='o')
    plt.plot(months, median, label='Median confidence', marker='s')
    plt.fill_between(months, q25, q75, color='steelblue', alpha=0.3, label='25th-75th percentile')

    plt.xticks(months)
    plt.xlabel("Month")
    plt.ylabel("Confidence")
    title = f"{ds_name.replace("_", " ").title()}: Monthly Confidence Statistics{' (' + phase + ')' if phase else ''}"
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    suffix = f"_{phase}" if phase else ""
    plt.savefig(screenshots_directory / f"{ds_name}_modal_confidence_monthly_summary{suffix}.png", dpi=150)
    plt.show()


def analyse_confidence(ds_name: str, phases: list[str] = None):
    zarr_path = copernicus_data_directory / "zarr" / f"{ds_name}_modal_monthly.zarr"
    output_path = copernicus_data_directory / f"copernicus_{ds_name}_modal_monthly_confidence_summary.json"
    ds = xr.open_zarr(zarr_path, consolidated=False)

    summaries = {}

    if phases:
        for i, phase in enumerate(phases):
            conf = ds["confidence"].isel(phase=i).values
            summary = compute_confidence_summary(conf)
            print(f"Phase: {phase}")
            for k, v in summary.items():
                if k != "values":
                    print(f"  {k.capitalize()}: {v}")
            plot_histogram(summary["values"], ds_name, phase)

            # Monthly summary
            df_conf = ds["confidence"].isel(phase=i).to_dataframe().reset_index()
            monthly = df_conf.groupby("month")["confidence"].agg(
                count='count',
                min='min',
                max='max',
                mean='mean',
                median='median',
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75)
            )
            plot_monthly_summary(monthly, ds_name, phase)
            summaries[phase] = monthly.round(4).to_dict(orient='index')
    else:
        conf = ds["confidence"].values
        summary = compute_confidence_summary(conf)
        print("No phase separation")
        for k, v in summary.items():
            if k != "values":
                print(f"  {k.capitalize()}: {v}")
        plot_histogram(summary["values"], ds_name)

        df_conf = ds["confidence"].to_dataframe().reset_index()
        monthly = df_conf.groupby("month")["confidence"].agg(
            count='count',
            min='min',
            max='max',
            mean='mean',
            median='median',
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75)
        )
        plot_monthly_summary(monthly, ds_name)
        summaries = monthly.round(4).to_dict(orient='index')

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved confidence summary to {output_path}")


def main():
    datasets = {
        "wind": None,
        "current": ["ebb", "flood"],
    }

    for ds_name, phases in datasets.items():
        print(f"\n=== Processing {ds_name} ===")
        try:
            analyse_confidence(ds_name, phases=phases)
        except Exception as e:
            print(f"Error processing {ds_name}: {e}")


if __name__ == "__main__":
    main()
