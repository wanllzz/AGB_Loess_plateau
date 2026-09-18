"""Trend-free pre-whitening Mann–Kendall sensitivity for the main mapped trends.

This script retains Theil–Sen slopes from the primary analysis and recalculates
Mann–Kendall significance after trend-free pre-whitening (Yue and Wang, 2002).
For incomplete annual series, the internal TFPW detrending uses the original
year coordinates, so missing observations do not compress elapsed time.
It summarizes the change in mapped trend-category proportions; it does not
replace the primary trend maps.
"""
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
import json

import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, prange

ROOT = DATA_DIR
OUT = ROBUSTNESS_DATA_DIR
NC = TEMPORAL_DATA_DIR / "agb_1km_annual_1990_2023.nc"
TEMP = TEMPORAL_DATA_DIR
ALPHA = 0.05

METRICS = {
    "landscape_agbd": "landscape_mean_agbd",
    "forest_fraction": "forest_fraction",
    "persistent_forest_agbd": "forest_mean_agbd",
}
PERIODS = {
    "1990_1999": (1990, 1999),
    "2000_2023": (2000, 2023),
    "1990_2023": (1990, 2023),
}


@njit(cache=True)
def norm_cdf(x):
    # Abramowitz and Stegun approximation, sufficiently accurate for MK p values.
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989423 * np.exp(-x * x / 2.0)
    prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1.0 - prob if x >= 0 else prob


@njit(cache=True)
def median_sorted(v, n):
    arr = np.sort(v[:n])
    if n % 2 == 1:
        return arr[n // 2]
    return 0.5 * (arr[n // 2 - 1] + arr[n // 2])


@njit(cache=True)
def tfpw_mk_p(values, times):
    """TFPW-MK p value, retaining original time coordinates after missing values."""
    n0 = values.shape[0]
    x = np.empty(n0, dtype=np.float64)
    t = np.empty(n0, dtype=np.float64)
    n = 0
    for i in range(n0):
        if np.isfinite(values[i]):
            x[n] = values[i]
            t[n] = times[i]
            n += 1
    if n < 4:
        return np.nan

    pairs = np.empty(n * (n - 1) // 2, dtype=np.float64)
    q = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            pairs[q] = (x[j] - x[i]) / (t[j] - t[i])
            q += 1
    slope = median_sorted(pairs, q)

    detrended = np.empty(n, dtype=np.float64)
    mean = 0.0
    for i in range(n):
        detrended[i] = x[i] - t[i] * slope
        mean += detrended[i]
    mean /= n
    denom = 0.0
    numer = 0.0
    for i in range(n):
        d = detrended[i] - mean
        denom += d * d
        if i > 0:
            numer += d * (detrended[i - 1] - mean)
    rho = numer / denom if denom > 0.0 else 0.0

    y = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        # Reintroduce the trend at the original year of the retained observation.
        y[i] = detrended[i + 1] - detrended[i] * rho + t[i + 1] * slope
    m = n - 1

    s = 0.0
    for i in range(m - 1):
        for j in range(i + 1, m):
            if y[j] > y[i]:
                s += 1.0
            elif y[j] < y[i]:
                s -= 1.0

    sorted_y = np.sort(y)
    tie_term = 0.0
    k = 0
    while k < m:
        count = 1
        while k + count < m and sorted_y[k + count] == sorted_y[k]:
            count += 1
        if count > 1:
            tie_term += count * (count - 1) * (2 * count + 5)
        k += count
    var_s = (m * (m - 1) * (2 * m + 5) - tie_term) / 18.0
    if var_s <= 0.0:
        return 1.0
    if s > 0:
        z = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0
    return 2.0 * (1.0 - norm_cdf(abs(z)))


@njit(parallel=True, cache=True)
def tfpw_matrix(arr, times):
    # arr is time x cell.
    result = np.empty(arr.shape[1], dtype=np.float64)
    for col in prange(arr.shape[1]):
        result[col] = tfpw_mk_p(arr[:, col], times)
    return result


def load_original(metric, period):
    import rasterio
    p = TEMP / f"{metric}_mann_kendall_p_{period}.tif"
    s = TEMP / f"{metric}_theil_sen_slope_{period}.tif"
    with rasterio.open(p) as ds:
        pv = ds.read(1).astype(float)
        if ds.nodata is not None:
            pv[pv == ds.nodata] = np.nan
    with rasterio.open(s) as ds:
        slope = ds.read(1).astype(float)
        if ds.nodata is not None:
            slope[slope == ds.nodata] = np.nan
    return pv, slope


def classify(slope, p):
    out = np.full(slope.shape, np.nan, dtype=float)
    valid = np.isfinite(slope) & np.isfinite(p)
    out[valid & (slope == 0)] = 0
    out[valid & (slope > 0) & (p >= ALPHA)] = 1
    out[valid & (slope > 0) & (p < ALPHA)] = 2
    out[valid & (slope < 0) & (p >= ALPHA)] = -1
    out[valid & (slope < 0) & (p < ALPHA)] = -2
    return out


def summary_row(metric, period, original_p, slope, tfpw_p):
    original = classify(slope, original_p)
    tfpw = classify(slope, tfpw_p)
    valid = np.isfinite(original) & np.isfinite(tfpw)
    return {
        "metric": metric,
        "period": period.replace("_", "–"),
        "valid_cells": int(valid.sum()),
        "standard_sig_increase_percent": float(100 * np.mean(original[valid] == 2)),
        "tfpw_sig_increase_percent": float(100 * np.mean(tfpw[valid] == 2)),
        "standard_sig_decrease_percent": float(100 * np.mean(original[valid] == -2)),
        "tfpw_sig_decrease_percent": float(100 * np.mean(tfpw[valid] == -2)),
        "category_agreement_percent": float(100 * np.mean(original[valid] == tfpw[valid])),
        "median_abs_p_difference": float(np.nanmedian(np.abs(original_p[valid] - tfpw_p[valid]))),
    }


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    ds = xr.open_dataset(NC)
    years = ds["year"].values.astype(int)
    rows = []
    for metric, var in METRICS.items():
        arr = ds[var].values.astype(np.float64)  # time, x, y
        for period, (start, end) in PERIODS.items():
            mask = (years >= start) & (years <= end)
            sub = arr[mask].reshape(mask.sum(), -1)
            print(f"TFPW-MK {metric} {period}: {sub.shape}", flush=True)
            p_tfpw = tfpw_matrix(sub, years[mask].astype(np.float64)).reshape(arr.shape[1:])
            original_p, slope = load_original(metric, period)
            rows.append(summary_row(metric, period, original_p, slope, p_tfpw))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "Table_S19_tfpw_mk_trend_significance_sensitivity.csv", index=False)
    (OUT / "Table_S19_tfpw_mk_methods.json").write_text(json.dumps({
        "method": "trend-free pre-whitening Mann-Kendall sensitivity",
        "reference": "Yue and Wang (2002)",
        "alpha": ALPHA,
        "primary_slope": "Theil-Sen slope retained from the primary analysis",
        "time_index_handling": "Original year coordinates retained for the internal Sen slope, detrending, and trend restoration after missing observations",
        "implementation": "Matches pymannkendall.trend_free_pre_whitening_modification_test for complete annual series; extends it to retain true year intervals for incomplete series",
    }, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
