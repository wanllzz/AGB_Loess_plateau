"""Aggregate HCD01/PSP climate variables to annual values on the native 0.1° grid.

Source files are read one year at a time. Zip members are extracted into a
temporary directory and discarded immediately, so no nationwide intermediate
climate rasters are retained in the project results directory.
"""

from __future__ import annotations

import tempfile
import zipfile
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
import argparse

import numpy as np
import pandas as pd
import xarray as xr


RESULT_ROOT = DATA_DIR
PROCESS_DIR = PROCESS_DATA_DIR
RESPONSE_NC = PROCESS_DIR / "process_responses_native_0p1deg_1990_2022.nc"
CLIMATE_ROOT = RAW_DATA_DIR / "climate" / "HCD01_10km"
PSP_ZIP = CLIMATE_ROOT / "PSPv1.1" / "PSP_CN_HCD01_1961-2024.zip"
DEFAULT_YEARS = np.arange(1990, 2023)


def nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray([np.abs(source - value).argmin() for value in target])


def annual_from_file(
    path: Path,
    variable: str,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    ds = xr.open_dataset(path, decode_times=False)
    # The Loess Plateau indices are contiguous. Slices avoid advanced indexing
    # that can force the NetCDF backend to read a nationwide daily array.
    lat_slice = slice(int(lat_indices.min()), int(lat_indices.max()) + 1)
    lon_slice = slice(int(lon_indices.min()), int(lon_indices.max()) + 1)
    values = ds[variable].isel(lat=lat_slice, lon=lon_slice).isel(
        lat=lat_indices - lat_indices.min(), lon=lon_indices - lon_indices.min()
    )
    if aggregation == "sum":
        annual = values.sum(dim="time", skipna=True)
    else:
        annual = values.mean(dim="time", skipna=True)
    array = annual.values.astype(np.float32)
    ds.close()
    return array


def annual_from_zip(
    archive: Path,
    member: str,
    variable: str,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    aggregation: str,
) -> np.ndarray:
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory() as temp_dir:
        extracted = Path(bundle.extract(member, temp_dir))
        return annual_from_file(
            extracted, variable, lat_indices, lon_indices, aggregation
        )


def main(start_year: int, end_year: int) -> None:
    years = np.arange(start_year, end_year + 1)
    response = xr.open_dataset(RESPONSE_NC)
    target_lat = response["lat"].values
    target_lon = response["lon"].values
    response.close()

    reference = xr.open_dataset(CLIMATE_ROOT / "t2m" / "1990.nc", decode_times=False)
    source_lat = reference["lat"].values
    source_lon = reference["lon"].values
    reference.close()
    hcd_lat_idx = nearest_indices(source_lat, target_lat)
    hcd_lon_idx = nearest_indices(source_lon, target_lon)

    # PET has centres offset by 0.05° from the HCD01 meteorological grid.
    with zipfile.ZipFile(PSP_ZIP) as bundle, tempfile.TemporaryDirectory() as temp_dir:
        ref_name = "PSP_CN_HCD01_1961-2024/yearly/PET/PET_1990_yearly.nc"
        ref_file = Path(bundle.extract(ref_name, temp_dir))
        pet_ref = xr.open_dataset(ref_file, decode_times=False)
        pet_lat_idx = nearest_indices(pet_ref["lat"].values, target_lat)
        pet_lon_idx = nearest_indices(pet_ref["lon"].values, target_lon)
        pet_ref.close()

    config = {
        "Tmean": {
            "kind": "file",
            "path": CLIMATE_ROOT / "t2m",
            "member": "{year}.nc",
            "variable": "t2m",
            "aggregation": "mean",
            "units": "degC",
        },
        "PRE": {
            "kind": "zip",
            "path": CLIMATE_ROOT / "tp.zip",
            "member": "tp/{year}.nc",
            "variable": "tp",
            "aggregation": "sum",
            "units": "mm yr-1",
        },
        "RH": {
            "kind": "zip",
            "path": CLIMATE_ROOT / "RH.zip",
            "member": "RH/{year}.nc",
            "variable": "RH",
            "aggregation": "mean",
            "units": "percent",
        },
        "SSRD": {
            "kind": "zip",
            "path": CLIMATE_ROOT / "ssrd.zip",
            "member": "ssrd/{year}.nc",
            "variable": "ssrd",
            "aggregation": "mean",
            "units": "W m-2",
        },
        "WS": {
            "kind": "zip",
            "path": CLIMATE_ROOT / "WS.zip",
            "member": "WS/{year}.nc",
            "variable": "WS",
            "aggregation": "mean",
            "units": "m s-1",
        },
    }

    annual: dict[str, np.ndarray] = {
        name: np.full((years.size, target_lat.size, target_lon.size), np.nan, dtype=np.float32)
        for name in [*config, "PET"]
    }

    for y_idx, year in enumerate(years):
        print(f"Processing climate year {year}")
        for name, spec in config.items():
            if spec["kind"] == "file":
                file_path = spec["path"] / spec["member"].format(year=year)
                values = annual_from_file(
                    file_path,
                    spec["variable"],
                    hcd_lat_idx,
                    hcd_lon_idx,
                    spec["aggregation"],
                )
            else:
                values = annual_from_zip(
                    spec["path"],
                    spec["member"].format(year=year),
                    spec["variable"],
                    hcd_lat_idx,
                    hcd_lon_idx,
                    spec["aggregation"],
                )
            annual[name][y_idx] = values

        pet_member = f"PSP_CN_HCD01_1961-2024/yearly/PET/PET_{year}_yearly.nc"
        annual["PET"][y_idx] = annual_from_zip(
            PSP_ZIP,
            pet_member,
            "PET",
            pet_lat_idx,
            pet_lon_idx,
            "mean",
        )

    output = xr.Dataset(
        data_vars={
            name: (("year", "lat", "lon"), values, {"units": config.get(name, {}).get("units", "mm yr-1")})
            for name, values in annual.items()
        },
        coords={"year": years, "lat": target_lat, "lon": target_lon},
        attrs={
            "title": "Annual climate predictors on the native HCD01 0.1 degree master lattice",
            "meteorological_source": "HCD01 daily climate fields",
            "PET_source": "PSPv1.1 HCD01-driven yearly PET",
            "aggregation": "PRE summed from daily values; Tmean, RH, SSRD and WS averaged from daily values; PET used from yearly product",
        },
    )
    suffix = f"{start_year}_{end_year}"
    output_path = PROCESS_DIR / f"climate_annual_native_0p1deg_{suffix}.nc"
    output.to_netcdf(
        output_path,
        encoding={name: {"zlib": True, "complevel": 4} for name in output.data_vars},
    )

    audit_rows = []
    for name, values in annual.items():
        audit_rows.append(
            {
                "variable": name,
                "year_start": int(years.min()),
                "year_end": int(years.max()),
                "minimum": float(np.nanmin(values)),
                "median": float(np.nanmedian(values)),
                "maximum": float(np.nanmax(values)),
                "missing_fraction": float(np.isnan(values).mean()),
                "units": config.get(name, {}).get("units", "mm yr-1"),
            }
        )
    pd.DataFrame(audit_rows).to_csv(PROCESS_DIR / f"climate_annual_audit_{suffix}.csv", index=False)
    print(f"Wrote {output_path}")
    print(pd.DataFrame(audit_rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=int(DEFAULT_YEARS.min()))
    parser.add_argument("--end", type=int, default=int(DEFAULT_YEARS.max()))
    args = parser.parse_args()
    main(args.start, args.end)
