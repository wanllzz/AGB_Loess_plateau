"""Priority robustness analyses for the Loess Plateau forest AGB study.

This script produces three safeguards requested during pre-submission review:
1. CFATD-derived forest fraction versus independently produced CLCD forest cover.
2. Antecedent (non-overlapping) forest-state baselines for spatial-XGBoost sensitivity.
3. Equal-length late-period AGB-stock decomposition sensitivity.

All outputs are concise tables. No intermediate rasters are retained.
"""

from __future__ import annotations

import importlib.util
import json
import sys
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
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from scipy.stats import pearsonr, spearmanr


ROOT = DATA_DIR
PROCESS_ROOT = PROCESS_DATA_DIR
OUT = ROBUSTNESS_DATA_DIR
OUT.mkdir(parents=True, exist_ok=True)

AGB_NC = TEMPORAL_DATA_DIR / "agb_1km_annual_1990_2023.nc"
PROCESS_NC = PROCESS_ROOT / "process_responses_native_0p1deg_1990_2022.nc"
MODEL_TABLE = PROCESS_ROOT / "process_response_complete_predictor_table_0p1deg.csv"
CLCD_DIR = RAW_DATA_DIR / "landcover" / "CLCD"
CFATD_DIR = RAW_DATA_DIR / "forest" / "CFATD"
MODEL_SCRIPT = CODE_DIR / "process_associations" / "fit_nested_spatial_xgboost_process_models.py"

FOREST_THRESHOLD = 0.10
SAMPLING_FACTOR = 20
BLOCK_SIZE_M = 50_000
N_BOOTSTRAP = 1000
RNG_SEED = 20260830
ZONE_NAMES = {0: "Loess Plateau", 1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C", 6: "D"}


def flat_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The response NetCDF retains the complete rectangular HCD01 lattice,
    # including cells outside the study area. This is the correct grid for
    # resampling external products; ecological_zone_id then supplies the mask.
    with xr.open_dataset(PROCESS_NC) as ds:
        return ds.lat.values.copy(), ds.lon.values.copy(), ds.ecological_zone_id.values.ravel().copy()


def sample_clcd_forest(year: int, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Estimate forest fraction from the annual CLCD class map at 0.1 degree."""
    path = CLCD_DIR / f"CLCD_v01_{year}_albert.tif"
    dx = float(np.median(np.diff(lon)))
    dy = abs(float(np.median(np.diff(lat))))
    transform = Affine(
        dx / SAMPLING_FACTOR, 0.0, lon.min() - dx / 2,
        0.0, -dy / SAMPLING_FACTOR, lat.max() + dy / 2,
    )
    sampled = np.zeros((len(lat) * SAMPLING_FACTOR, len(lon) * SAMPLING_FACTOR), dtype=np.uint8)
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
    forest = (cells == 2).mean(axis=(1, 3))
    return np.where(valid > 0, forest / valid, np.nan).astype(np.float32)


def cfatd_annual_fraction(years: list[int], lat: np.ndarray, lon: np.ndarray) -> dict[int, np.ndarray]:
    with xr.open_dataset(PROCESS_NC) as ds:
        source_lat = ds.lat.values
        source_lon = ds.lon.values
        if not (np.allclose(source_lat, lat) and np.allclose(source_lon, lon)):
            raise ValueError("The process-response and model lattices differ.")
        available = ds.year.values.astype(int)
        values = ds.forest_fraction.values
        return {year: values[np.where(available == year)[0][0]] for year in years}


def weighted_mean(values: np.ndarray, lat: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat))[:, None]
    good = np.isfinite(values)
    return float(np.nansum(values * weights) / np.nansum(np.where(good, weights, 0)))


def compare_forest_cover() -> None:
    lat, lon, zones = flat_grid()
    years = [1990, 1991, 1992, 2000, 2001, 2002, 2020, 2021, 2022]
    cfatd = cfatd_annual_fraction(years, lat, lon)
    clcd = {year: sample_clcd_forest(year, lat, lon) for year in years}

    records = []
    cells = []
    for year in years:
        a = cfatd[year].ravel()
        b = clcd[year].ravel()
        valid = np.isfinite(a) & np.isfinite(b) & (zones > 0)
        pearson = pearsonr(a[valid], b[valid])
        spearman = spearmanr(a[valid], b[valid])
        records.append({
            "comparison": "annual", "year_or_window": str(year), "n_cells": int(valid.sum()),
            "spatial_Pearson_r": float(pearson.statistic), "Pearson_p": float(pearson.pvalue),
            "spatial_Spearman_rho": float(spearman.statistic), "Spearman_p": float(spearman.pvalue),
            "CFATD_area_weighted_forest_fraction": weighted_mean(cfatd[year], lat),
            "CLCD_area_weighted_forest_fraction": weighted_mean(clcd[year], lat),
        })
        cells.append(pd.DataFrame({
            "year": year, "lat": np.repeat(lat, len(lon)), "lon": np.tile(lon, len(lat)),
            "ecological_zone_id": zones, "CFATD_forest_fraction": a, "CLCD_forest_fraction": b,
        }))

    windows = {"1990-1992": [1990, 1991, 1992], "2000-2002": [2000, 2001, 2002], "2020-2022": [2020, 2021, 2022]}
    means = {}
    for label, years_in_window in windows.items():
        a = np.nanmean(np.stack([cfatd[y] for y in years_in_window]), axis=0)
        b = np.nanmean(np.stack([clcd[y] for y in years_in_window]), axis=0)
        means[label] = (a, b)
        valid = np.isfinite(a) & np.isfinite(b) & (zones.reshape(a.shape) > 0)
        pearson = pearsonr(a[valid], b[valid])
        spearman = spearmanr(a[valid], b[valid])
        records.append({
            "comparison": "three_year_mean", "year_or_window": label, "n_cells": int(valid.sum()),
            "spatial_Pearson_r": float(pearson.statistic), "Pearson_p": float(pearson.pvalue),
            "spatial_Spearman_rho": float(spearman.statistic), "Spearman_p": float(spearman.pvalue),
            "CFATD_area_weighted_forest_fraction": weighted_mean(a, lat),
            "CLCD_area_weighted_forest_fraction": weighted_mean(b, lat),
        })

    first_a, first_b = means["1990-1992"]
    last_a, last_b = means["2020-2022"]
    delta_a, delta_b = last_a - first_a, last_b - first_b
    valid = np.isfinite(delta_a) & np.isfinite(delta_b) & (zones.reshape(delta_a.shape) > 0)
    pearson = pearsonr(delta_a[valid], delta_b[valid])
    spearman = spearmanr(delta_a[valid], delta_b[valid])
    records.append({
        "comparison": "change_1990_1992_to_2020_2022", "year_or_window": "change", "n_cells": int(valid.sum()),
        "spatial_Pearson_r": float(pearson.statistic), "Pearson_p": float(pearson.pvalue),
        "spatial_Spearman_rho": float(spearman.statistic), "Spearman_p": float(spearman.pvalue),
        "CFATD_area_weighted_forest_fraction": weighted_mean(delta_a, lat),
        "CLCD_area_weighted_forest_fraction": weighted_mean(delta_b, lat),
    })

    def change_class(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        out = np.full(start.shape, -1, dtype=np.int8)
        valid = np.isfinite(start) & np.isfinite(end)
        out[valid & (start >= FOREST_THRESHOLD) & (end >= FOREST_THRESHOLD)] = 0
        out[valid & (start < FOREST_THRESHOLD) & (end >= FOREST_THRESHOLD)] = 1
        out[valid & (start >= FOREST_THRESHOLD) & (end < FOREST_THRESHOLD)] = 2
        out[valid & (start < FOREST_THRESHOLD) & (end < FOREST_THRESHOLD)] = 3
        return out

    class_a, class_b = change_class(first_a, last_a), change_class(first_b, last_b)
    both = (class_a >= 0) & (class_b >= 0) & (zones.reshape(class_a.shape) > 0)
    agreement = float((class_a[both] == class_b[both]).mean())
    pa = np.bincount(class_a[both], minlength=4) / both.sum()
    pb = np.bincount(class_b[both], minlength=4) / both.sum()
    expected = float(np.dot(pa, pb))
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else np.nan
    pd.DataFrame(records).to_csv(OUT / "Table_S16_independent_forest_cover_consistency.csv", index=False)
    pd.concat(cells, ignore_index=True).to_csv(OUT / "forest_cover_consistency_cell_data.csv", index=False)
    pd.DataFrame([{
        "comparison_window": "1990-1992 versus 2020-2022", "threshold": FOREST_THRESHOLD,
        "n_common_cells": int(both.sum()), "four_class_agreement_percent": agreement * 100,
        "Cohen_kappa": kappa,
    }]).to_csv(OUT / "Table_S16_forest_change_class_agreement.csv", index=False)


def antecedent_baselines() -> Path:
    """Create antecedent 0.1 degree baselines without response-window overlap."""
    lat, lon, _ = flat_grid()
    dx, dy = abs(float(np.median(np.diff(lon)))), abs(float(np.median(np.diff(lat))))
    # A nested 0.005 degree lattice preserves the CFATD zero/non-zero forest
    # mask. Direct average resampling with zero as nodata would otherwise return
    # forest-only means only for fully forested parent cells.
    transform = Affine(
        dx / SAMPLING_FACTOR, 0.0, lon.min() - dx / 2,
        0.0, -dy / SAMPLING_FACTOR, lat.max() + dy / 2,
    )

    def annual_aggregates(year: int) -> tuple[np.ndarray, np.ndarray]:
        path = CFATD_DIR / f"AGB_{year}.tif"
        sampled = np.zeros((len(lat) * SAMPLING_FACTOR, len(lon) * SAMPLING_FACTOR), dtype=np.uint16)
        with rasterio.open(path) as source:
            reproject(
                source=rasterio.band(source, 1), destination=sampled,
                src_transform=source.transform, src_crs=source.crs, src_nodata=0,
                dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=0,
                resampling=Resampling.nearest, num_threads=2,
            )
        cells = sampled.reshape(len(lat), SAMPLING_FACTOR, len(lon), SAMPLING_FACTOR).astype(np.float32)
        forest = cells > 0
        fraction = forest.mean(axis=(1, 3)).astype(np.float32)
        total = np.where(forest, cells, 0).sum(axis=(1, 3))
        count = forest.sum(axis=(1, 3))
        forest_mean = np.divide(total, count, out=np.full(total.shape, np.nan, dtype=np.float32), where=count > 0)
        return fraction, forest_mean

    window_years = {"antecedent_1987_1989": [1987, 1988, 1989], "antecedent_1997_1999": [1997, 1998, 1999]}
    summaries = {}
    for label, years in window_years.items():
        values = [annual_aggregates(year) for year in years]
        summaries[label] = (np.nanmean(np.stack([v[0] for v in values]), axis=0), np.nanmean(np.stack([v[1] for v in values]), axis=0))
        print(f"Completed {label}", flush=True)

    model = pd.read_csv(MODEL_TABLE)
    model["lat"] = model["lat"].round(6)
    model["lon"] = model["lon"].round(6)
    records = []
    reference = {"early_1990_1999": "antecedent_1987_1989", "late_2000_2022": "antecedent_1997_1999", "full_1990_2022": "antecedent_1987_1989"}
    for period, source in reference.items():
        fraction, agbd = summaries[source]
        frame = pd.DataFrame({
            "period": period, "lat": np.repeat(lat, len(lon)), "lon": np.tile(lon, len(lat)),
            "antecedent_window": source.replace("antecedent_", "").replace("_", "-"),
            "antecedent_forest_fraction_baseline": fraction.ravel(),
            "antecedent_forest_agbd_baseline": agbd.ravel(),
        })
        frame["lat"] = frame["lat"].round(6)
        frame["lon"] = frame["lon"].round(6)
        records.append(frame)
    baseline = pd.concat(records, ignore_index=True)
    sensitivity_table = model.merge(baseline, on=["period", "lat", "lon"], how="left", validate="many_to_one")
    path = OUT / "antecedent_baseline_model_table.csv"
    sensitivity_table.to_csv(path, index=False)

    summary = []
    for period in reference:
        subset = sensitivity_table[sensitivity_table.period == period]
        summary.append({
            "period": period, "antecedent_window": reference[period].replace("antecedent_", "").replace("_", "-"),
            "forest_fraction_baseline_finite_fraction": float(subset.antecedent_forest_fraction_baseline.notna().mean()),
            "forest_agbd_baseline_finite_fraction": float(subset.antecedent_forest_agbd_baseline.notna().mean()),
            "correlation_with_main_forest_fraction_baseline": float(spearmanr(subset.forest_fraction_baseline, subset.antecedent_forest_fraction_baseline, nan_policy="omit").statistic),
            "correlation_with_main_agbd_baseline": float(spearmanr(subset.persistent_forest_agbd_baseline, subset.antecedent_forest_agbd_baseline, nan_policy="omit").statistic),
        })
    pd.DataFrame(summary).to_csv(OUT / "Table_S17_antecedent_baseline_construction.csv", index=False)
    return path


def fit_antecedent_sensitivity(table: Path) -> None:
    spec = importlib.util.spec_from_file_location("main_models", MODEL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["main_models"] = module
    spec.loader.exec_module(module)
    module.ROOT = OUT
    module.TABLE = table
    module.PROCESS_CONFIG = {
        "forest_fraction_change": {"response": "forest_fraction_slope", "eligible": "forest_change_state_eligible", "baseline": "antecedent_forest_fraction_baseline"},
        "persistent_forest_agbd_change": {"response": "persistent_forest_agbd_slope", "eligible": "persistent_forest_eligible", "baseline": "antecedent_forest_agbd_baseline"},
    }
    module.main()
    for path in OUT.glob("nested_spatial_xgboost_*.csv"):
        path.rename(path.with_name(path.name.replace("nested_spatial_xgboost", "antecedent_baseline_xgboost")))
    source = OUT / "crossvalidated_treeshap_importance.csv"
    if source.exists():
        source.rename(OUT / "antecedent_baseline_treeshap_importance.csv")


def decompose(a0: float, b0: float, a1: float, b1: float) -> dict[str, float]:
    area = 1e-7 * ((b0 + b1) / 2) * (a1 - a0)
    density = 1e-7 * ((a0 + a1) / 2) * (b1 - b0)
    total = area + density
    return {"area_contribution_Pg": area, "density_contribution_Pg": density, "stock_change_Pg": total,
            "area_contribution_percent": 100 * area / total if total else np.nan,
            "density_contribution_percent": 100 * density / total if total else np.nan}


def equal_window_decomposition() -> None:
    periods = {"1990-1999": (1990, 1999), "2000-2009": (2000, 2009), "2014-2023": (2014, 2023)}
    rng = np.random.default_rng(RNG_SEED)
    with xr.open_dataset(AGB_NC) as ds:
        years = ds.year.values.astype(int)
        fraction = ds.forest_fraction.values
        landscape = ds.landscape_mean_agbd.values
        zone_grid = ds.ecological_zone_id.values
        x, y = ds.x.values, ds.y.values
    block_x = ((x - x.min()) // BLOCK_SIZE_M).astype(int)
    block_y = ((y - y.min()) // BLOCK_SIZE_M).astype(int)
    blocks = block_y[:, None] * 1000 + block_x[None, :]
    index = {year: int(np.where(years == year)[0][0]) for year in years}
    result, bootstrap = [], []
    for zone_id, zone_name in ZONE_NAMES.items():
        mask = zone_grid > 0 if zone_id == 0 else zone_grid == zone_id
        for label, (start, end) in periods.items():
            f0, f1 = fraction[index[start]][mask], fraction[index[end]][mask]
            l0, l1 = landscape[index[start]][mask], landscape[index[end]][mask]
            valid = np.isfinite(f0) & np.isfinite(f1) & np.isfinite(l0) & np.isfinite(l1)
            area0, area1 = f0[valid], f1[valid]
            stock0, stock1 = l0[valid] * 1e-7, l1[valid] * 1e-7
            a0, a1, s0, s1 = area0.sum(), area1.sum(), stock0.sum(), stock1.sum()
            b0, b1 = s0 / (1e-7 * a0), s1 / (1e-7 * a1)
            row = {"zone": zone_name, "period": label, "start_year": start, "end_year": end, "n_cells": int(valid.sum()),
                   "forest_area_start_km2": a0, "forest_area_end_km2": a1, "mean_agbd_start_Mg_ha": b0, "mean_agbd_end_Mg_ha": b1}
            row.update(decompose(a0, b0, a1, b1)); result.append(row)
            cell_blocks = blocks[mask][valid]
            ids = np.unique(cell_blocks)
            ba0 = np.array([area0[cell_blocks == i].sum() for i in ids]); ba1 = np.array([area1[cell_blocks == i].sum() for i in ids])
            bs0 = np.array([stock0[cell_blocks == i].sum() for i in ids]); bs1 = np.array([stock1[cell_blocks == i].sum() for i in ids])
            draws = rng.integers(0, len(ids), size=(N_BOOTSTRAP, len(ids)))
            aa0, aa1 = ba0[draws].sum(1), ba1[draws].sum(1)
            bb0, bb1 = bs0[draws].sum(1) / (1e-7 * aa0), bs1[draws].sum(1) / (1e-7 * aa1)
            samples = np.stack([1e-7 * ((bb0 + bb1) / 2) * (aa1 - aa0), 1e-7 * ((aa0 + aa1) / 2) * (bb1 - bb0)], axis=1)
            for j, component in enumerate(["area", "density"]):
                bootstrap.append({"zone": zone_name, "period": label, "component": component, "estimate_Pg": row[f"{component}_contribution_Pg"],
                                  "ci_lower_Pg": float(np.quantile(samples[:, j], .025)), "ci_upper_Pg": float(np.quantile(samples[:, j], .975)),
                                  "n_blocks": int(len(ids)), "n_bootstrap": N_BOOTSTRAP})
    pd.DataFrame(result).to_csv(OUT / "Table_S18_equal_length_stock_decomposition.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(OUT / "Table_S18_equal_length_stock_decomposition_bootstrap.csv", index=False)


def main() -> None:
    compare_forest_cover()
    table = antecedent_baselines()
    fit_antecedent_sensitivity(table)
    equal_window_decomposition()
    (OUT / "robustness_metadata.json").write_text(json.dumps({
        "forest_fraction_consistency": "CFATD forest fraction compared with CLCD forest class fraction",
        "antecedent_baselines": {"early_and_full": "1987-1989", "late": "1997-1999"},
        "equal_length_decomposition_periods": ["1990-1999", "2000-2009", "2014-2023"],
        "random_seed": RNG_SEED,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
