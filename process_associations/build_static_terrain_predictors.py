"""Aggregate static terrain predictors to the native 0.1 degree climate lattice."""

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
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import xarray as xr


ROOT = PROCESS_DATA_DIR
RESPONSE_NC = ROOT / "process_responses_native_0p1deg_1990_2022.nc"
RESPONSE_TABLE = ROOT / "process_response_climate_model_table_0p1deg.csv"
DEM_ROOT = RAW_DATA_DIR / "terrain" / "DEM"
SOURCES = {"DEM": DEM_ROOT / "dem.tif", "SLP": DEM_ROOT / "podu.tif", "TPI": DEM_ROOT / "TPI.tif"}


def main() -> None:
    master = xr.open_dataset(RESPONSE_NC)
    lat = master.lat.values
    lon = master.lon.values
    master.close()
    transform = from_origin(float(lon.min() - 0.05), float(lat.max() + 0.05), 0.1, 0.1)
    target_shape = (lat.size, lon.size)
    values = {}
    for name, source in SOURCES.items():
        destination = np.full(target_shape, np.nan, dtype=np.float32)
        with rasterio.open(source) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
        values[name] = destination

    grid_lon, grid_lat = np.meshgrid(lon, lat)
    terrain = pd.DataFrame({"lon": grid_lon.ravel(), "lat": grid_lat.ravel(), **{k: v.ravel() for k, v in values.items()}})
    terrain["lon"] = terrain["lon"].round(6)
    terrain["lat"] = terrain["lat"].round(6)
    terrain.to_csv(ROOT / "terrain_static_predictors_0p1deg.csv", index=False)

    model = pd.read_csv(RESPONSE_TABLE)
    model["lon"] = model["lon"].round(6)
    model["lat"] = model["lat"].round(6)
    merged = model.merge(terrain, on=["lon", "lat"], how="left", validate="many_to_one")
    missing = merged[list(SOURCES)].isna().any(axis=1)
    if missing.any():
        # Border climate cells can contain only a sliver of the source DEM.
        # Fill these sparse gaps from the nearest available master-grid cell.
        from scipy.spatial import cKDTree

        available = terrain.dropna(subset=list(SOURCES)).copy()
        tree = cKDTree(available[["lon", "lat"]].to_numpy())
        _, nearest = tree.query(merged.loc[missing, ["lon", "lat"]].to_numpy(), k=1)
        merged.loc[missing, list(SOURCES)] = available.iloc[nearest][list(SOURCES)].to_numpy()
        pd.DataFrame({
            "lon": merged.loc[missing, "lon"],
            "lat": merged.loc[missing, "lat"],
            "reason": "edge_no_data_nearest_master_cell_fill",
        }).to_csv(ROOT / "terrain_edge_fill_audit.csv", index=False)
    merged.to_csv(ROOT / "process_response_climate_terrain_model_table_0p1deg.csv", index=False)
    print(merged[list(SOURCES)].describe().to_string())


if __name__ == "__main__":
    main()
