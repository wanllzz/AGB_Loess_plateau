"""Merge validated annual climate chunks into the final 1990-2022 input file."""

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
import os

import pandas as pd
import xarray as xr


ROOT = PROCESS_DATA_DIR
CHUNKS = [
    ROOT / "climate_annual_native_0p1deg_1990_1995.nc",
    ROOT / "climate_annual_native_0p1deg_1996_2000.nc",
    ROOT / "climate_annual_native_0p1deg_2001_2005.nc",
    ROOT / "climate_annual_native_0p1deg_2006_2010.nc",
    ROOT / "climate_annual_native_0p1deg_2011_2015.nc",
    ROOT / "climate_annual_native_0p1deg_2016_2020.nc",
    ROOT / "climate_annual_native_0p1deg_2021_2022.nc",
]
OUTPUT = ROOT / "climate_annual_native_0p1deg_1990_2022.nc"


def main() -> None:
    datasets = [xr.open_dataset(path) for path in CHUNKS]
    merged = xr.concat(datasets, dim="year").sortby("year")
    expected = list(range(1990, 2023))
    actual = merged.year.values.astype(int).tolist()
    if actual != expected:
        raise ValueError(f"Unexpected annual coverage: {actual}")
    if any(merged[name].isnull().any() for name in merged.data_vars):
        raise ValueError("Merged climate data contain missing values")

    temp_output = OUTPUT.with_suffix(".tmp.nc")
    merged.to_netcdf(
        temp_output,
        encoding={name: {"zlib": True, "complevel": 4} for name in merged.data_vars},
    )
    for dataset in datasets:
        dataset.close()
    merged.close()
    os.replace(temp_output, OUTPUT)

    audit = []
    verified = xr.open_dataset(OUTPUT)
    for name in verified.data_vars:
        values = verified[name].values
        audit.append(
            {
                "variable": name,
                "year_start": 1990,
                "year_end": 2022,
                "minimum": float(values.min()),
                "median": float(pd.Series(values.ravel()).median()),
                "maximum": float(values.max()),
                "missing_fraction": float(pd.isna(values.ravel()).mean()),
                "units": verified[name].attrs.get("units", ""),
            }
        )
    verified.close()
    pd.DataFrame(audit).to_csv(ROOT / "climate_annual_audit.csv", index=False)
    print(pd.DataFrame(audit).to_string(index=False))


if __name__ == "__main__":
    main()
