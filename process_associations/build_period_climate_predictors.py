"""Create period-mean and Theil-Sen climate predictors for process models."""

from pathlib import Path

import sys

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from path_config import (
    CODE_DIR,
    DATA_DIR,
    FIGURE_OUTPUT_DIR,
    FIGURE_SOURCE_DIR,
    PROCESS_DATA_DIR,
    RAW_DATA_DIR,
    ROBUSTNESS_DATA_DIR,
    TEMPORAL_DATA_DIR,
)

import numpy as np
import pandas as pd
import xarray as xr


ROOT = PROCESS_DATA_DIR
CLIMATE = ROOT / "climate_annual_native_0p1deg_1990_2022.nc"
RESPONSES = ROOT / "process_response_model_table_0p1deg.csv"
PERIODS = {
    "early_1990_1999": (1990, 1999),
    "late_2000_2022": (2000, 2022),
    "late_equal_2000_2009": (2000, 2009),
    "late_recent_2013_2022": (2013, 2022),
    "full_1990_2022": (1990, 2022),
}


def slopes(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Vectorized Theil-Sen estimator for complete annual climate series."""
    row, col = np.triu_indices(values.shape[0], k=1)
    pair_slopes = (values[col] - values[row]) / (years[col] - years[row])[:, None]
    return np.median(pair_slopes, axis=0).astype(np.float32)


def main() -> None:
    climate = xr.open_dataset(CLIMATE)
    years = climate.year.values.astype(int)
    variables = list(climate.data_vars)
    lon_grid, lat_grid = np.meshgrid(climate.lon.values, climate.lat.values)
    lookup = pd.DataFrame({"lon": lon_grid.ravel(), "lat": lat_grid.ravel()})
    climate_flat = {
        name: climate[name].values.reshape(years.size, -1) for name in variables
    }
    climate.close()

    response = pd.read_csv(RESPONSES)
    response["lon"] = response["lon"].round(6)
    response["lat"] = response["lat"].round(6)
    outputs = []
    for period, (start, end) in PERIODS.items():
        time_idx = np.where((years >= start) & (years <= end))[0]
        period_years = years[time_idx]
        frame = lookup.copy()
        frame["period"] = period
        for name, array in climate_flat.items():
            values = array[time_idx]
            frame[f"{name}_mean"] = np.nanmean(values, axis=0)
            frame[f"{name}_slope"] = slopes(values, period_years)
        outputs.append(frame)

    climate_table = pd.concat(outputs, ignore_index=True)
    climate_table["lon"] = climate_table["lon"].round(6)
    climate_table["lat"] = climate_table["lat"].round(6)
    climate_table.to_csv(ROOT / "climate_period_predictors_0p1deg.csv", index=False)

    merged = response.merge(climate_table, on=["period", "lon", "lat"], how="left", validate="one_to_one")
    climate_columns = [
        f"{name}_{metric}" for name in variables for metric in ("mean", "slope")
    ]
    if merged[climate_columns].isna().any().any():
        raise ValueError("Climate predictors missing after response-table merge")
    merged.to_csv(ROOT / "process_response_climate_model_table_0p1deg.csv", index=False)

    audit = (
        merged.groupby("period")[[column for column in merged.columns if column.endswith("_mean") or column.endswith("_slope")]]
        .agg(["min", "median", "max"])
    )
    audit.to_csv(ROOT / "climate_period_predictor_audit.csv")
    print(f"Rows in merged response-climate table: {len(merged)}")
    print(merged.groupby("period").size().to_string())


if __name__ == "__main__":
    main()
