"""Build population-density baseline and observed interval-change predictors.

GHSL population rasters report population counts in an equal-area Mollweide grid.
Counts are summed onto the 0.1° analysis lattice and divided by geodesic cell
area to obtain density.  No annual interpolation is performed: observed 1990-1995,
2000-2020, and 1990-2020 changes are expressed only as annualised log differences.
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
from pyproj import Geod
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_lhgi_ntl_predictors as grid_utils  # noqa: E402


ROOT = PROCESS_DATA_DIR
BASE_TABLE = ROOT / "process_response_main_predictor_table_0p1deg.csv"
POP_DIR = RAW_DATA_DIR / "population" / "GHSL"
EPOCHS = {1990: 100, 1995: 100, 2000: 1000, 2020: 1000}
PERIOD_INTERVAL = {
    "early_1990_1999": (1990, 1995),
    "late_2000_2022": (2000, 2020),
    "late_equal_2000_2009": (2000, 2020),
    "late_recent_2013_2022": (2000, 2020),
    "full_1990_2022": (1990, 2020),
}


def population_vsi(year: int) -> str:
    resolution = EPOCHS[year]
    stem = f"GHS_POP_E{year}_GLOBE_R2023A_54009_{resolution}_V1_0"
    archive = POP_DIR / f"{stem}.zip"
    if not archive.exists():
        raise FileNotFoundError(archive)
    return f"/vsizip/{archive.as_posix()}/{stem}.tif"


def target_transform(lat, lon):
    dx = abs(float(np.median(np.diff(lon))))
    dy = abs(float(np.median(np.diff(lat))))
    return Affine(dx, 0.0, lon.min() - dx / 2, 0.0, -dy, lat.max() + dy / 2)


def geodesic_area_km2(lat, lon) -> np.ndarray:
    geod = Geod(ellps="WGS84")
    dx = abs(float(np.median(np.diff(lon))))
    dy = abs(float(np.median(np.diff(lat))))
    result = np.empty((len(lat), len(lon)), dtype="float64")
    for i, centre_lat in enumerate(lat):
        for j, centre_lon in enumerate(lon):
            west, east = centre_lon - dx / 2, centre_lon + dx / 2
            south, north = centre_lat - dy / 2, centre_lat + dy / 2
            area, _ = geod.polygon_area_perimeter([west, east, east, west], [south, south, north, north])
            result[i, j] = abs(area) / 1_000_000
    return result


def density(year, transform, shape, area_km2) -> np.ndarray:
    source_path = population_vsi(year)
    print(f"Aggregating GHSL population {year}", flush=True)
    counts = np.full(shape, np.nan, dtype="float64")
    with rasterio.open(source_path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=counts,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=-200,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.sum,
            num_threads=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return (counts / area_km2).astype("float32")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lat, lon, _ = grid_utils.target_geometry()
    transform = target_transform(lat, lon)
    areas = geodesic_area_km2(lat, lon)
    densities = {year: density(year, transform, areas.shape, areas) for year in EPOCHS}

    rows = []
    for period, (start, end) in PERIOD_INTERVAL.items():
        baseline = densities[start]
        change = (np.log1p(densities[end]) - np.log1p(densities[start])) / (end - start)
        rows.append(pd.DataFrame({
            "period": period,
            "lat": np.repeat(lat, len(lon)),
            "lon": np.tile(lon, len(lat)),
            "POP_baseline_year": start,
            "POP_observed_endpoint_year": end,
            "POP_density_baseline_person_km2": baseline.ravel(),
            "POP_log1p_density_change_yr": change.ravel(),
        }))
    pop = pd.concat(rows, ignore_index=True)
    model = pd.read_csv(BASE_TABLE)
    for frame in (model, pop):
        frame["lat"] = frame["lat"].round(6)
        frame["lon"] = frame["lon"].round(6)
    out = model.merge(pop, on=["period", "lat", "lon"], how="left", validate="one_to_one")
    out.to_csv(ROOT / "process_response_complete_predictor_table_0p1deg.csv", index=False)

    audit = []
    for year, values in densities.items():
        finite = values[np.isfinite(values)]
        audit.append({
            "epoch": year,
            "variable": "population_density_person_km2",
            "finite_fraction": float(np.isfinite(values).mean()),
            "minimum": float(np.nanmin(finite)),
            "median": float(np.nanmedian(finite)),
            "maximum": float(np.nanmax(finite)),
            "native_resolution_m": EPOCHS[year],
        })
    pd.DataFrame(audit).to_csv(ROOT / "ghsl_population_predictor_audit.csv", index=False)


if __name__ == "__main__":
    main()
