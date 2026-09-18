"""Evaluate sensitivity of persistent-forest AGBD trends to mask definitions.

This script does not overwrite the primary 10% forest / 90% persistence
response dataset.  It recomputes persistent-forest AGBD Theil-Sen slopes for
alternative forest-presence and temporal-persistence thresholds and compares
them with the primary definition at the native 0.1 degree climate lattice.
"""

from __future__ import annotations

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
from pyproj import Transformer
from scipy.stats import spearmanr


RESULT_ROOT = DATA_DIR
INPUT_NC = TEMPORAL_DATA_DIR / "agb_1km_annual_1990_2023.nc"
CLIMATE_REFERENCE = RAW_DATA_DIR / "climate" / "HCD01_10km" / "t2m" / "1990.nc"
OUTPUT_DIR = PROCESS_DATA_DIR

FOREST_THRESHOLDS = (0.05, 0.10, 0.20)
PERSISTENCE_THRESHOLDS = (0.80, 0.90, 1.00)
MIN_PERSISTENT_1KM_CELLS = 10
REFERENCE = (0.10, 0.90)
PERIODS = {
    "early_1990_1999": (1990, 1999),
    "late_2000_2022": (2000, 2022),
    "full_1990_2022": (1990, 2022),
}


def nanmean_by_group(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    finite = np.isfinite(values)
    sums = np.bincount(groups[finite], weights=values[finite], minlength=n_groups)
    counts = np.bincount(groups[finite], minlength=n_groups)
    result = np.full(n_groups, np.nan, dtype=np.float32)
    result[counts > 0] = (sums[counts > 0] / counts[counts > 0]).astype(np.float32)
    return result


def theil_sen_pairwise(series: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Exact Theil-Sen slope from all pairwise slopes, vectorised by cell."""
    pair_slopes = []
    for first in range(len(years) - 1):
        for second in range(first + 1, len(years)):
            pair_slopes.append((series[second] - series[first]) / (years[second] - years[first]))
    slopes = np.nanmedian(np.stack(pair_slopes), axis=0)
    slopes[np.isfinite(series).sum(axis=0) < 3] = np.nan
    return slopes.astype(np.float32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    climate = xr.open_dataset(CLIMATE_REFERENCE, decode_times=False)
    climate_lon = climate["lon"].values.astype(float)
    climate_lat = climate["lat"].values.astype(float)
    climate.close()

    dataset = xr.open_dataset(INPUT_NC)
    all_years = dataset["year"].values.astype(int)
    use_time = np.where((all_years >= 1990) & (all_years <= 2022))[0]
    years = all_years[use_time]
    valid_grid = dataset["ecological_zone_id"].values > 0
    y_index, x_index = np.where(valid_grid)
    x = dataset["x"].values[x_index]
    y = dataset["y"].values[y_index]
    forest_fraction = dataset["forest_fraction"].isel(time=use_time).values[:, valid_grid]
    forest_agbd = dataset["forest_mean_agbd"].isel(time=use_time).values[:, valid_grid]
    dataset.close()

    transformer = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    lon_index = np.abs(lon[:, None] - climate_lon[None, :]).argmin(axis=1)
    lat_index = np.abs(lat[:, None] - climate_lat[None, :]).argmin(axis=1)
    climate_cell = lat_index.astype(np.int32) * climate_lon.size + lon_index.astype(np.int32)
    unique_cells, inverse = np.unique(climate_cell, return_inverse=True)
    n_cells = unique_cells.size
    cell_lon = climate_lon[unique_cells % climate_lon.size]
    cell_lat = climate_lat[unique_cells // climate_lon.size]

    slope_store: dict[tuple[str, float, float], np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []

    for period, (start, end) in PERIODS.items():
        loc = np.where((years >= start) & (years <= end))[0]
        period_years = years[loc]
        fractions = forest_fraction[loc]
        agbd = forest_agbd[loc]

        for forest_threshold in FOREST_THRESHOLDS:
            presence = np.isfinite(fractions) & (fractions >= forest_threshold)
            for persistence_threshold in PERSISTENCE_THRESHOLDS:
                persistent = (
                    presence[0]
                    & presence[-1]
                    & (presence.mean(axis=0) >= persistence_threshold)
                )
                count = np.bincount(inverse[persistent], minlength=n_cells).astype(np.int32)
                annual = np.stack(
                    [
                        nanmean_by_group(
                            np.where(persistent, agbd[t], np.nan), inverse, n_cells
                        )
                        for t in range(period_years.size)
                    ]
                )
                slope = theil_sen_pairwise(annual, period_years)
                baseline = np.nanmean(annual[:3], axis=0)
                eligible = (
                    (count >= MIN_PERSISTENT_1KM_CELLS)
                    & np.isfinite(slope)
                    & np.isfinite(baseline)
                )
                slope[~eligible] = np.nan
                slope_store[(period, forest_threshold, persistence_threshold)] = slope
                summary_rows.append(
                    {
                        "period": period,
                        "forest_presence_threshold": forest_threshold,
                        "temporal_persistence_threshold": persistence_threshold,
                        "min_persistent_1km_cells": MIN_PERSISTENT_1KM_CELLS,
                        "eligible_0p1deg_cells": int(eligible.sum()),
                        "median_persistent_forest_agbd_slope": float(np.nanmedian(slope)),
                        "mean_persistent_forest_agbd_slope": float(np.nanmean(slope)),
                        "median_persistent_forest_agbd_baseline": float(np.nanmedian(baseline[eligible])),
                    }
                )

    comparison_rows: list[dict[str, object]] = []
    spatial_rows: list[pd.DataFrame] = []
    for period in PERIODS:
        reference = slope_store[(period, *REFERENCE)]
        for forest_threshold in FOREST_THRESHOLDS:
            for persistence_threshold in PERSISTENCE_THRESHOLDS:
                slope = slope_store[(period, forest_threshold, persistence_threshold)]
                common = np.isfinite(reference) & np.isfinite(slope)
                if common.sum() >= 3:
                    rho, p_value = spearmanr(reference[common], slope[common])
                    sign_agreement = np.mean(np.sign(reference[common]) == np.sign(slope[common]))
                else:
                    rho, p_value, sign_agreement = np.nan, np.nan, np.nan
                comparison_rows.append(
                    {
                        "period": period,
                        "forest_presence_threshold": forest_threshold,
                        "temporal_persistence_threshold": persistence_threshold,
                        "reference_forest_threshold": REFERENCE[0],
                        "reference_persistence_threshold": REFERENCE[1],
                        "common_eligible_cells": int(common.sum()),
                        "spearman_rho_vs_reference": rho,
                        "spearman_p_vs_reference": p_value,
                        "slope_sign_agreement_vs_reference": sign_agreement,
                    }
                )
                spatial_rows.append(
                    pd.DataFrame(
                        {
                            "period": period,
                            "forest_presence_threshold": forest_threshold,
                            "temporal_persistence_threshold": persistence_threshold,
                            "lon": cell_lon,
                            "lat": cell_lat,
                            "persistent_forest_agbd_slope": slope,
                        }
                    )
                )

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "persistent_forest_threshold_sensitivity_summary.csv", index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        OUTPUT_DIR / "persistent_forest_threshold_sensitivity_reference_comparison.csv",
        index=False,
    )
    pd.concat(spatial_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "persistent_forest_threshold_sensitivity_slopes.csv", index=False
    )


if __name__ == "__main__":
    main()
