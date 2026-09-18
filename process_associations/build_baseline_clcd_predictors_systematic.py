"""Low-memory systematic aggregation of CLCD baseline cover fractions.

Each 0.1° analysis cell is subdivided into a 20 x 20 regular 0.005° sampling
lattice. CLCD is sampled by nearest-neighbour reprojection to this nested lattice,
then class frequencies are calculated within each parent cell. The procedure avoids
holding China-scale 30-m rasters in memory; it produces only final 0.1° tables.
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
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_lhgi_ntl_predictors as grid_utils  # noqa: E402


ROOT = PROCESS_DATA_DIR
BASE_TABLE = ROOT / "process_response_climate_terrain_lhgi_ntl_model_table_0p1deg.csv"
CLCD_DIR = RAW_DATA_DIR / "landcover" / "CLCD"
SAMPLING_FACTOR = 20

# CLCD v01: 1 cropland; 2 forest; 3 shrubland; 4 grassland; 5 water;
# 6 snow/ice; 7 barren; 8 impervious. Forest is not included as a predictor
# because the forest-fraction trend is a response variable.
GROUPS = {
    "cropland_fraction": 1,
    "shrubland_fraction": 3,
    "grassland_fraction": 4,
}
BASELINE_YEARS = {1990: (1990, 1991, 1992), 2000: (2000, 2001, 2002)}
PERIOD_BASELINE_YEAR = {
    "early_1990_1999": 1990,
    "late_2000_2022": 2000,
    "late_equal_2000_2009": 2000,
    "late_recent_2013_2022": 2000,
    "full_1990_2022": 1990,
}


def sample_year(year: int, lat: np.ndarray, lon: np.ndarray) -> dict:
    path = CLCD_DIR / f"CLCD_v01_{year}_albert.tif"
    if not path.exists():
        raise FileNotFoundError(path)
    dx = abs(float(np.median(np.diff(lon))))
    dy = abs(float(np.median(np.diff(lat))))
    width = len(lon) * SAMPLING_FACTOR
    height = len(lat) * SAMPLING_FACTOR
    transform = Affine(
        dx / SAMPLING_FACTOR, 0.0, lon.min() - dx / 2,
        0.0, -dy / SAMPLING_FACTOR, lat.max() + dy / 2,
    )
    sampled = np.zeros((height, width), dtype="uint8")
    print(f"Sampling {path.name} to {height} x {width} nested grid", flush=True)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=sampled,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=0,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=0,
            resampling=Resampling.nearest,
            num_threads=2,
        )
    cells = sampled.reshape(len(lat), SAMPLING_FACTOR, len(lon), SAMPLING_FACTOR)
    valid = (cells != 0).mean(axis=(1, 3))
    result = {}
    for name, code in GROUPS.items():
        raw_fraction = (cells == code).mean(axis=(1, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            result[name] = np.where(valid > 0, raw_fraction / valid, np.nan).astype("float32")
    return result


def mean_baseline(start_year: int, lat: np.ndarray, lon: np.ndarray) -> dict:
    stacks = {name: [] for name in GROUPS}
    for year in BASELINE_YEARS[start_year]:
        annual = sample_year(year, lat, lon)
        for name, values in annual.items():
            stacks[name].append(values)
    return {name: np.nanmean(np.stack(values, axis=0), axis=0) for name, values in stacks.items()}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lat, lon, _ = grid_utils.target_geometry()
    baselines = {year: mean_baseline(year, lat, lon) for year in BASELINE_YEARS}

    rows = []
    for period, start_year in PERIOD_BASELINE_YEAR.items():
        fields = {
            "period": period,
            "lat": np.repeat(lat, len(lon)),
            "lon": np.tile(lon, len(lat)),
            "LULC_baseline_start_year": start_year,
            "LULC_baseline_years": f"{start_year}-{start_year + 2}",
        }
        for name, values in baselines[start_year].items():
            fields[name] = values.ravel()
        rows.append(pd.DataFrame(fields))
    lcl = pd.concat(rows, ignore_index=True)
    model = pd.read_csv(BASE_TABLE)
    for frame in (model, lcl):
        frame["lat"] = frame["lat"].round(6)
        frame["lon"] = frame["lon"].round(6)
    merged = model.merge(lcl, on=["period", "lat", "lon"], how="left", validate="one_to_one")
    merged.to_csv(ROOT / "process_response_main_predictor_table_0p1deg.csv", index=False)

    audit = []
    for start_year, values_by_name in baselines.items():
        for name, values in values_by_name.items():
            finite = values[np.isfinite(values)]
            audit.append({
                "baseline_years": f"{start_year}-{start_year + 2}",
                "variable": name,
                "finite_fraction": float(np.isfinite(values).mean()),
                "minimum": float(np.nanmin(finite)),
                "median": float(np.nanmedian(finite)),
                "maximum": float(np.nanmax(finite)),
                "CLCD_class": GROUPS[name],
                "sampling_lattice": f"{SAMPLING_FACTOR}x{SAMPLING_FACTOR} per 0.1 degree cell",
            })
    pd.DataFrame(audit).to_csv(ROOT / "clcd_baseline_predictor_audit.csv", index=False)


if __name__ == "__main__":
    main()
