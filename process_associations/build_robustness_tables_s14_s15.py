"""Assemble Supplementary Tables S14-S15 source tables for Fig. 5 robustness."""

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

import pandas as pd


ROOT = PROCESS_DATA_DIR
EARLY = "early_1990_1999"
LATE = "late_2000_2022"


def main():
    performance = pd.read_csv(ROOT / "common_sample_persistent_agbd_performance.csv")
    comparison = pd.read_csv(ROOT / "common_sample_persistent_agbd_comparison.csv")
    shift = pd.read_csv(ROOT / "common_sample_persistent_agbd_treeshap_importance.csv")
    fold_shift = pd.read_csv(ROOT / "common_sample_persistent_agbd_foldwise_importance_shift_summary.csv")
    full = shift.pivot(index="predictor", columns="period", values="relative_importance_percent").reset_index()
    full["late_minus_early_pp"] = full[LATE] - full[EARLY]
    full = full.merge(fold_shift, on="predictor", how="left")
    performance = performance[performance.model == "full"].merge(comparison, on="period", how="left")
    performance.to_csv(ROOT / "Table_S14_common_sample_performance_and_concordance.csv", index=False)
    full.to_csv(ROOT / "Table_S14_common_sample_treeshap_shift.csv", index=False)

    ranking = pd.read_csv(ROOT / "temporal_window_shap_ranking_comparison.csv")
    detail = pd.read_csv(ROOT / "temporal_window_shap_structure_sensitivity.csv")
    domain = pd.read_csv(ROOT / "temporal_window_shap_domain_sensitivity.csv")
    ranking.to_csv(ROOT / "Table_S15_alternative_window_ranking_concordance.csv", index=False)
    detail.to_csv(ROOT / "Table_S15_alternative_window_predictor_structure.csv", index=False)
    domain.to_csv(ROOT / "Table_S15_alternative_window_domain_structure.csv", index=False)


if __name__ == "__main__":
    main()
