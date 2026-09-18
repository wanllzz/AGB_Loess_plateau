"""Pre-model Spearman and VIF diagnostics for process-specific AGB analyses."""

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
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression


ROOT = PROCESS_DATA_DIR
TABLE = ROOT / "process_response_complete_predictor_table_0p1deg.csv"
MAIN_PERIODS = ["early_1990_1999", "late_2000_2022"]

# Pre-specified candidates; PET is held out of the primary screen because it is
# a modelled atmospheric-demand quantity constructed from overlapping climate
# information. It will be assessed separately as a sensitivity predictor.
CANDIDATES = [
    "Tmean_mean", "PRE_mean", "RH_mean", "SSRD_mean", "WS_mean",
    "Tmean_slope", "PRE_slope",
    "DEM", "SLP", "TPI",
    "LHGI_mean", "LHGI_sen_slope",
    "NTL_log1p_mean", "NTL_log1p_sen_slope",
    "POP_density_baseline_person_km2", "POP_log1p_density_change_yr",
    "cropland_fraction", "shrubland_fraction", "grassland_fraction",
]


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    result = []
    for index, name in enumerate(frame.columns):
        other = np.delete(values, index, axis=1)
        target = values[:, index]
        r2 = LinearRegression().fit(other, target).score(other, target)
        vif = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
        result.append({"predictor": name, "R2_from_other_predictors": r2, "VIF": vif})
    return pd.DataFrame(result)


def main() -> None:
    data = pd.read_csv(TABLE)
    data = data[data.period.isin(MAIN_PERIODS)].copy()
    outputs = []
    vif_outputs = []
    for process, eligibility in {
        "forest_fraction_change": data.forest_change_state_eligible,
        "persistent_forest_agbd_change": data.persistent_forest_eligible,
    }.items():
        subset = data.loc[eligibility, CANDIDATES].dropna().copy()
        correlation, pvalues = spearmanr(subset, axis=0)
        for i, first in enumerate(CANDIDATES):
            for j in range(i + 1, len(CANDIDATES)):
                outputs.append({
                    "process": process,
                    "n_complete_samples": len(subset),
                    "predictor_1": first,
                    "predictor_2": CANDIDATES[j],
                    "spearman_rho": correlation[i, j],
                    "p_value": pvalues[i, j],
                    "abs_rho_ge_0_70": abs(correlation[i, j]) >= 0.70,
                })
        vif = calculate_vif(subset)
        vif.insert(0, "process", process)
        vif.insert(1, "n_complete_samples", len(subset))
        vif_outputs.append(vif)
    pd.DataFrame(outputs).to_csv(ROOT / "predictor_spearman_screen_main_periods.csv", index=False)
    pd.concat(vif_outputs, ignore_index=True).to_csv(ROOT / "predictor_vif_screen_main_periods.csv", index=False)
    pd.DataFrame({"candidate_predictor": CANDIDATES}).to_csv(ROOT / "predictor_screen_candidate_definitions.csv", index=False)


if __name__ == "__main__":
    main()
