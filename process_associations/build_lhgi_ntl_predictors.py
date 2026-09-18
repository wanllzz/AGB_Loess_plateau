"""Build period-specific LHGI and corrected nighttime-light predictors on the 0.1° climate grid.

The script deliberately keeps only final period summaries.  LHGI is an annual 0.1°
grazing-pressure product.  Corrected annual nighttime-light values are first
area-averaged from the published 1-km product to the native climate lattice, then
log1p-transformed.  Population and LULC are handled separately because their source
time steps and measurement definitions require different aggregation rules.
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
from typing import Tuple

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
import xarray as xr


ROOT = PROCESS_DATA_DIR
GRID = ROOT / "process_responses_native_0p1deg_1990_2022.nc"
BASE_TABLE = ROOT / "process_response_climate_terrain_model_table_0p1deg.csv"
LHGI_DIR = RAW_DATA_DIR / "grazing" / "LHGI"
NTL_DIR = RAW_DATA_DIR / "nighttime_lights" / "EVAL_NTL"

PERIODS = {
    "early_1990_1999": (1990, 1999),
    "late_2000_2022": (2000, 2022),
    "late_equal_2000_2009": (2000, 2009),
    "late_recent_2013_2022": (2013, 2022),
    "full_1990_2022": (1990, 2022),
}


def theil_sen(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Pairwise-median slope for columns of shape (year, gridcell)."""
    slopes = []
    for i in range(len(years) - 1):
        for j in range(i + 1, len(years)):
            slopes.append((values[j] - values[i]) / float(years[j] - years[i]))
    return np.nanmedian(np.stack(slopes, axis=0), axis=0)


def target_geometry():
    with xr.open_dataset(GRID) as ds:
        lat = ds.lat.values.astype(float)
        lon = ds.lon.values.astype(float)
    dx = abs(float(np.median(np.diff(lon))))
    dy = abs(float(np.median(np.diff(lat))))
    # The master grid is ordered north-to-south, matching Rasterio's north-up
    # array convention. Do not reverse the reprojected result.
    transform = Affine(dx, 0.0, lon.min() - dx / 2, 0.0, -dy, lat.max() + dy / 2)
    return lat, lon, transform


def resample_to_target(path: Path, transform: Affine, shape: Tuple[int, int]) -> np.ndarray:
    """Resample one source to the fixed geographic 0.1° grid using area averages."""
    dest = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.average,
            num_threads=2,
        )
    return dest


def build_annual_series(prefix: str, directory: Path, years: np.ndarray, transform, shape) -> np.ndarray:
    annual = []
    for year in years:
        path = directory / f"{prefix}{year}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Processing {path.name}", flush=True)
        annual.append(resample_to_target(path, transform, shape))
    return np.stack(annual, axis=0)


def available_years(prefix: str, directory: Path) -> list:
    available = []
    for path in directory.glob(f"{prefix}*.tif"):
        try:
            available.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(available)


def build_ntl_series(years: np.ndarray, transform, shape) -> np.ndarray:
    """Create a non-overlapping EVAL-to-VAL NTL series.

    EVAL is the reconstructed VIIRS-like product through 2013; VAL is the
    processed official annual NPP-VIIRS product from 2012 onward. The 2012-2013
    overlap is resolved by retaining EVAL through 2013 and VAL from 2014.
    """
    annual = []
    for year in years:
        prefix = "EVAL_China_" if year <= 2013 else "VAL_China_"
        path = NTL_DIR / f"{prefix}{year}.tif"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Processing {path.name}", flush=True)
        annual.append(resample_to_target(path, transform, shape))
    return np.stack(annual, axis=0)


def period_summary(values: np.ndarray, years: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
    mask = (years >= start) & (years <= end)
    sub_values, sub_years = values[mask], years[mask]
    return np.nanmean(sub_values, axis=0), theil_sen(sub_values, sub_years)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lat, lon, transform = target_geometry()
    shape = (len(lat), len(lon))
    years = np.arange(1990, 2023)

    lhgi = build_annual_series("LHGI_", LHGI_DIR, years, transform, shape)
    ntl_raw = build_ntl_series(years, transform, shape)
    # Published NTL can contain small negative calibration residuals; log1p is
    # evaluated only for non-negative physical radiance/intensity values.
    ntl_log = np.log1p(np.clip(ntl_raw, 0, None))

    table = pd.read_csv(BASE_TABLE)
    rows = []
    for period, (start, end) in PERIODS.items():
        lhgi_mean, lhgi_slope = period_summary(lhgi, years, start, end)
        fields = {
            "period": period,
            "lat": np.repeat(lat, len(lon)),
            "lon": np.tile(lon, len(lat)),
            "LHGI_mean": lhgi_mean.ravel(),
            "LHGI_sen_slope": lhgi_slope.ravel(),
        }
        ntl_mean, ntl_slope = period_summary(ntl_log, years, start, end)
        fields["NTL_log1p_mean"] = ntl_mean.ravel()
        fields["NTL_log1p_sen_slope"] = ntl_slope.ravel()
        rows.append(pd.DataFrame(fields))
    human = pd.concat(rows, ignore_index=True)
    # CSV serialisation can introduce tiny coordinate differences; standardise the
    # scientific grid keys rather than silently losing otherwise matched samples.
    for frame in (table, human):
        frame["lat"] = frame["lat"].round(6)
        frame["lon"] = frame["lon"].round(6)
    out = table.merge(human, on=["period", "lat", "lon"], how="left", validate="one_to_one")
    out.to_csv(ROOT / "process_response_climate_terrain_lhgi_ntl_model_table_0p1deg.csv", index=False)

    audit = []
    for name, values in [("LHGI", lhgi)]:
        finite = values[np.isfinite(values)]
        audit.append({
            "variable": name,
            "years": f"{years.min()}-{years.max()}",
            "finite_fraction": float(np.isfinite(values).mean()),
            "minimum": float(np.nanmin(finite)),
            "median": float(np.nanmedian(finite)),
            "maximum": float(np.nanmax(finite)),
        })
    for name, values in [("NTL_raw", ntl_raw), ("NTL_log1p", ntl_log)]:
        finite = values[np.isfinite(values)]
        audit.append({
            "variable": name,
            "years": f"{years.min()}-{years.max()}",
            "finite_fraction": float(np.isfinite(values).mean()),
            "minimum": float(np.nanmin(finite)),
            "median": float(np.nanmedian(finite)),
            "maximum": float(np.nanmax(finite)),
            "note": "EVAL 1990-2013; processed official VAL 2014-2022.",
        })
    pd.DataFrame(audit).to_csv(ROOT / "lhgi_ntl_predictor_audit.csv", index=False)


if __name__ == "__main__":
    main()
