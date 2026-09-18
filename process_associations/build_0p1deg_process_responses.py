"""Build process-separated 0.1 degree AGB response variables for the Loess Plateau.

The master lattice is the native 0.1 degree HCD01 climate grid. Annual 1 km
forest fraction and forest mean AGBD are aggregated first; Theil-Sen slopes are
then calculated from the aggregated annual series. This avoids averaging
fine-resolution slope estimates across the coarser climate cells.
"""

from __future__ import annotations

import json
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
from scipy.stats import theilslopes


RESULT_ROOT = DATA_DIR
INPUT_NC = TEMPORAL_DATA_DIR / "agb_1km_annual_1990_2023.nc"
CLIMATE_REFERENCE = RAW_DATA_DIR / "climate" / "HCD01_10km" / "t2m" / "1990.nc"
OUTPUT_DIR = PROCESS_DATA_DIR

FOREST_THRESHOLD = 0.10
PERSISTENCE_THRESHOLD = 0.90
SATURATION_THRESHOLD = 0.90
MIN_PERSISTENT_1KM_CELLS = 10

PERIODS = {
    "early_1990_1999": (1990, 1999),
    "late_2000_2022": (2000, 2022),
    "late_equal_2000_2009": (2000, 2009),
    "late_recent_2013_2022": (2013, 2022),
    "full_1990_2022": (1990, 2022),
}


def nanmean_by_group(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Compute group means while respecting NaNs."""
    finite = np.isfinite(values)
    sums = np.bincount(groups[finite], weights=values[finite], minlength=n_groups)
    counts = np.bincount(groups[finite], minlength=n_groups)
    out = np.full(n_groups, np.nan, dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return out


def slope_by_cell(series: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Theil-Sen slopes for a (time, cell) array."""
    output = np.full(series.shape[1], np.nan, dtype=np.float32)
    for idx in range(series.shape[1]):
        values = series[:, idx]
        valid = np.isfinite(values)
        if valid.sum() >= 3:
            output[idx] = theilslopes(values[valid], years[valid])[0]
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    climate = xr.open_dataset(CLIMATE_REFERENCE, decode_times=False)
    climate_lon = climate["lon"].values.astype(float)
    climate_lat = climate["lat"].values.astype(float)
    climate.close()

    ds = xr.open_dataset(INPUT_NC)
    years_all = ds["year"].values.astype(int)
    use_time = np.where((years_all >= 1990) & (years_all <= 2022))[0]
    years = years_all[use_time]

    zone_grid = ds["ecological_zone_id"].values
    valid_grid = zone_grid > 0
    y_idx, x_idx = np.where(valid_grid)
    zone = zone_grid[valid_grid].astype(np.int16)

    x = ds["x"].values[x_idx]
    y = ds["y"].values[y_idx]
    transformer = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)

    lon_index = np.abs(lon[:, None] - climate_lon[None, :]).argmin(axis=1)
    lat_index = np.abs(lat[:, None] - climate_lat[None, :]).argmin(axis=1)
    climate_cell = lat_index.astype(np.int32) * climate_lon.size + lon_index.astype(np.int32)
    unique_cells, inverse = np.unique(climate_cell, return_inverse=True)
    n_cells = unique_cells.size
    cell_count = np.bincount(inverse, minlength=n_cells).astype(np.int32)
    zone_by_cell = np.zeros(n_cells, dtype=np.int16)
    zone_max_count = np.zeros(n_cells, dtype=np.int32)
    for zone_id in range(1, 7):
        zone_count = np.bincount(inverse[zone == zone_id], minlength=n_cells).astype(np.int32)
        replace = zone_count > zone_max_count
        zone_by_cell[replace] = zone_id
        zone_max_count[replace] = zone_count[replace]

    cell_lon_index = unique_cells % climate_lon.size
    cell_lat_index = unique_cells // climate_lon.size
    cell_lon = climate_lon[cell_lon_index]
    cell_lat = climate_lat[cell_lat_index]

    # The study data are in an equal-area 1 km grid, so unweighted means are area means.
    forest_fraction = ds["forest_fraction"].isel(time=use_time).values[:, valid_grid]
    forest_agbd = ds["forest_mean_agbd"].isel(time=use_time).values[:, valid_grid]
    ds.close()

    annual_forest_fraction = np.stack(
        [nanmean_by_group(forest_fraction[t], inverse, n_cells) for t in range(years.size)]
    )

    period_records: list[dict[str, object]] = []
    persistent_agbd = np.full((len(PERIODS), years.size, n_cells), np.nan, dtype=np.float32)
    persistence_fraction = np.full((len(PERIODS), n_cells), np.nan, dtype=np.float32)
    persistent_count = np.zeros((len(PERIODS), n_cells), dtype=np.int32)
    forest_slope = np.full((len(PERIODS), n_cells), np.nan, dtype=np.float32)
    forest_baseline = np.full((len(PERIODS), n_cells), np.nan, dtype=np.float32)
    agbd_slope = np.full((len(PERIODS), n_cells), np.nan, dtype=np.float32)
    agbd_baseline = np.full((len(PERIODS), n_cells), np.nan, dtype=np.float32)

    for p_idx, (period_name, (start, end)) in enumerate(PERIODS.items()):
        loc = np.where((years >= start) & (years <= end))[0]
        period_years = years[loc]
        period_fraction = forest_fraction[loc]
        forest_presence = np.isfinite(period_fraction) & (period_fraction >= FOREST_THRESHOLD)
        persistent = (
            forest_presence[0]
            & forest_presence[-1]
            & (forest_presence.mean(axis=0) >= PERSISTENCE_THRESHOLD)
        )

        count = np.bincount(inverse[persistent], minlength=n_cells).astype(np.int32)
        persistent_count[p_idx] = count
        persistence_fraction[p_idx] = count / cell_count

        for local_t, global_t in enumerate(loc):
            values = forest_agbd[global_t].copy()
            values[~persistent] = np.nan
            persistent_agbd[p_idx, global_t] = nanmean_by_group(values, inverse, n_cells)

        forest_slope[p_idx] = slope_by_cell(annual_forest_fraction[loc], period_years)
        agbd_slope[p_idx] = slope_by_cell(persistent_agbd[p_idx, loc], period_years)
        forest_baseline[p_idx] = np.nanmean(annual_forest_fraction[loc[:3]], axis=0)
        agbd_baseline[p_idx] = np.nanmean(persistent_agbd[p_idx, loc[:3]], axis=0)

        expansion_state_eligible = (forest_baseline[p_idx] < SATURATION_THRESHOLD)
        biomass_sample_eligible = (
            (persistent_count[p_idx] >= MIN_PERSISTENT_1KM_CELLS)
            & np.isfinite(agbd_slope[p_idx])
            & np.isfinite(agbd_baseline[p_idx])
        )

        period_records.append(
            {
                "period": period_name,
                "start_year": start,
                "end_year": end,
                "n_master_cells": int(n_cells),
                "n_forest_change_state_eligible": int(expansion_state_eligible.sum()),
                "n_persistent_forest_eligible": int(biomass_sample_eligible.sum()),
                "forest_threshold": FOREST_THRESHOLD,
                "persistence_threshold": PERSISTENCE_THRESHOLD,
                "min_persistent_1km_cells": MIN_PERSISTENT_1KM_CELLS,
            }
        )

    # Build a compact rectangular subset of the native climate grid.
    lat_unique = np.unique(cell_lat_index)
    lon_unique = np.unique(cell_lon_index)
    lat_lookup = {value: pos for pos, value in enumerate(lat_unique)}
    lon_lookup = {value: pos for pos, value in enumerate(lon_unique)}
    row = np.array([lat_lookup[value] for value in cell_lat_index])
    col = np.array([lon_lookup[value] for value in cell_lon_index])
    shape = (lat_unique.size, lon_unique.size)

    def gridify(values: np.ndarray, fill=np.nan, dtype=np.float32) -> np.ndarray:
        arr = np.full(shape, fill, dtype=dtype)
        arr[row, col] = values
        return arr

    def gridify_period(values: np.ndarray, fill=np.nan, dtype=np.float32) -> np.ndarray:
        arr = np.full((values.shape[0], *shape), fill, dtype=dtype)
        for p_idx in range(values.shape[0]):
            arr[p_idx, row, col] = values[p_idx]
        return arr

    def gridify_period_time(values: np.ndarray) -> np.ndarray:
        arr = np.full((values.shape[0], values.shape[1], *shape), np.nan, dtype=np.float32)
        for p_idx in range(values.shape[0]):
            for t_idx in range(values.shape[1]):
                arr[p_idx, t_idx, row, col] = values[p_idx, t_idx]
        return arr

    period_names = list(PERIODS)
    out = xr.Dataset(
        data_vars={
            "forest_fraction": (("year", "lat", "lon"), np.stack([gridify(v) for v in annual_forest_fraction])),
            "persistent_forest_agbd": (("period", "year", "lat", "lon"), gridify_period_time(persistent_agbd)),
            "forest_fraction_slope": (("period", "lat", "lon"), gridify_period(forest_slope)),
            "persistent_forest_agbd_slope": (("period", "lat", "lon"), gridify_period(agbd_slope)),
            "forest_fraction_baseline": (("period", "lat", "lon"), gridify_period(forest_baseline)),
            "persistent_forest_agbd_baseline": (("period", "lat", "lon"), gridify_period(agbd_baseline)),
            "persistent_forest_fraction": (("period", "lat", "lon"), gridify_period(persistence_fraction)),
            "persistent_forest_1km_count": (("period", "lat", "lon"), gridify_period(persistent_count, fill=-1, dtype=np.int32)),
            "valid_1km_count": (("lat", "lon"), gridify(cell_count, fill=-1, dtype=np.int32)),
            "ecological_zone_id": (("lat", "lon"), gridify(zone_by_cell, fill=0, dtype=np.int16)),
        },
        coords={
            "period": period_names,
            "year": years,
            "lat": climate_lat[lat_unique],
            "lon": climate_lon[lon_unique],
        },
        attrs={
            "title": "Process-separated annual and trend response variables on the native 0.1 degree climate lattice",
            "source_1km_data": str(INPUT_NC),
            "climate_master_lattice": str(CLIMATE_REFERENCE),
            "forest_fraction_threshold": FOREST_THRESHOLD,
            "temporal_persistence_threshold": PERSISTENCE_THRESHOLD,
            "note": "Annual 1 km values were aggregated before trend estimation. Areas were equal at the source EPSG:6933 grid.",
        },
    )

    output_nc = OUTPUT_DIR / "process_responses_native_0p1deg_1990_2022.nc"
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in out.data_vars
        if out[name].dtype.kind == "f"
    }
    out.to_netcdf(output_nc, encoding=encoding)

    rows = []
    for p_idx, period_name in enumerate(period_names):
        for c_idx in range(n_cells):
            rows.append(
                {
                    "period": period_name,
                    "lon": cell_lon[c_idx],
                    "lat": cell_lat[c_idx],
                    "ecological_zone_id": int(zone_by_cell[c_idx]),
                    "valid_1km_count": int(cell_count[c_idx]),
                    "forest_fraction_slope": forest_slope[p_idx, c_idx],
                    "forest_fraction_baseline": forest_baseline[p_idx, c_idx],
                    "forest_change_state_eligible": bool(forest_baseline[p_idx, c_idx] < SATURATION_THRESHOLD),
                    "persistent_forest_agbd_slope": agbd_slope[p_idx, c_idx],
                    "persistent_forest_agbd_baseline": agbd_baseline[p_idx, c_idx],
                    "persistent_forest_fraction": persistence_fraction[p_idx, c_idx],
                    "persistent_forest_1km_count": int(persistent_count[p_idx, c_idx]),
                    "persistent_forest_eligible": bool(
                        persistent_count[p_idx, c_idx] >= MIN_PERSISTENT_1KM_CELLS
                        and np.isfinite(agbd_slope[p_idx, c_idx])
                        and np.isfinite(agbd_baseline[p_idx, c_idx])
                    ),
                }
            )
    model_table = pd.DataFrame(rows)
    model_table.to_csv(OUTPUT_DIR / "process_response_model_table_0p1deg.csv", index=False)
    pd.DataFrame(period_records).to_csv(OUTPUT_DIR / "process_response_construction_audit.csv", index=False)

    with (OUTPUT_DIR / "process_response_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "input": str(INPUT_NC),
                "master_climate_grid": str(CLIMATE_REFERENCE),
                "periods": PERIODS,
                "forest_threshold": FOREST_THRESHOLD,
                "persistence_threshold": PERSISTENCE_THRESHOLD,
                "saturation_threshold": SATURATION_THRESHOLD,
                "minimum_persistent_1km_cells": MIN_PERSISTENT_1KM_CELLS,
                "aggregation_rule": "annual equal-area 1 km values aggregated first, then Theil-Sen slopes estimated at 0.1 degree",
            },
            stream,
            indent=2,
        )

    print(f"Wrote {output_nc}")
    print(pd.DataFrame(period_records).to_string(index=False))


if __name__ == "__main__":
    main()
