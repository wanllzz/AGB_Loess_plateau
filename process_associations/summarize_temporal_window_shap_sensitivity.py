"""Summarise early-to-late TreeSHAP structure across alternative late windows."""

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
from scipy.stats import spearmanr


ROOT = PROCESS_DATA_DIR
PRIMARY = ROOT / "crossvalidated_treeshap_importance.csv"
EARLY = "early_1990_1999"
LATE = "late_2000_2022"
ALTERNATIVES = ["late_equal_2000_2009", "late_recent_2013_2022"]
PROCESSES = ["forest_fraction_change", "persistent_forest_agbd_change"]
GROUPS = {
    "Initial state": ["forest_fraction_baseline", "persistent_forest_agbd_baseline"],
    "Climate background": ["Tmean_mean", "PRE_mean", "WS_mean"],
    "Climate change": ["Tmean_slope", "PRE_slope"],
    "Terrain": ["SLP"],
    "Land-cover context": ["cropland_fraction", "shrubland_fraction", "grassland_fraction"],
    "Human pressure": ["LHGI_mean", "LHGI_sen_slope", "NTL_log1p_mean", "POP_log1p_density_change_yr"],
}


def label_group(predictor):
    return next(group for group, members in GROUPS.items() if predictor in members)


def main():
    primary = pd.read_csv(PRIMARY)
    all_details, all_domains, all_comparisons = [], [], []
    for process in PROCESSES:
        early = primary[(primary.process == process) & (primary.period == EARLY)].set_index("predictor")
        main_late = primary[(primary.process == process) & (primary.period == LATE)].set_index("predictor")
        for period in ALTERNATIVES:
            file = ROOT / f"temporal_window_sensitivity_{process}_{period}_treeshap_importance.csv"
            alternative = pd.read_csv(file).set_index("predictor")
            shared = early.index.intersection(alternative.index)
            detail = pd.DataFrame({
                "process": process,
                "alternative_window": period,
                "predictor": shared,
                "predictor_domain": [label_group(item) for item in shared],
                "early_relative_importance_percent": early.loc[shared, "relative_importance_percent"].to_numpy(),
                "alternative_relative_importance_percent": alternative.loc[shared, "relative_importance_percent"].to_numpy(),
                "main_late_relative_importance_percent": main_late.loc[shared, "relative_importance_percent"].to_numpy(),
            })
            detail["alternative_minus_early_pp"] = detail.alternative_relative_importance_percent - detail.early_relative_importance_percent
            detail["alternative_minus_main_late_pp"] = detail.alternative_relative_importance_percent - detail.main_late_relative_importance_percent
            all_details.append(detail)
            domain = detail.groupby(["process", "alternative_window", "predictor_domain"], as_index=False)[[
                "early_relative_importance_percent", "alternative_relative_importance_percent", "main_late_relative_importance_percent"
            ]].sum()
            domain["alternative_minus_early_pp"] = domain.alternative_relative_importance_percent - domain.early_relative_importance_percent
            domain["alternative_minus_main_late_pp"] = domain.alternative_relative_importance_percent - domain.main_late_relative_importance_percent
            all_domains.append(domain)
            rho_late, p_late = spearmanr(alternative.loc[shared, "relative_importance_percent"], main_late.loc[shared, "relative_importance_percent"])
            rho_early, p_early = spearmanr(alternative.loc[shared, "relative_importance_percent"], early.loc[shared, "relative_importance_percent"])
            alternative_top = set(alternative.loc[shared, "relative_importance_percent"].sort_values(ascending=False).head(5).index)
            late_top = set(main_late.loc[shared, "relative_importance_percent"].sort_values(ascending=False).head(5).index)
            all_comparisons.append({
                "process": process, "alternative_window": period,
                "ranking_rho_vs_main_late": rho_late, "ranking_p_vs_main_late": p_late,
                "ranking_rho_vs_early": rho_early, "ranking_p_vs_early": p_early,
                "top5_overlap_vs_main_late": len(alternative_top & late_top),
                "alternative_top5": "; ".join(alternative.loc[shared, "relative_importance_percent"].sort_values(ascending=False).head(5).index),
                "main_late_top5": "; ".join(main_late.loc[shared, "relative_importance_percent"].sort_values(ascending=False).head(5).index),
            })
    pd.concat(all_details, ignore_index=True).to_csv(ROOT / "temporal_window_shap_structure_sensitivity.csv", index=False)
    pd.concat(all_domains, ignore_index=True).to_csv(ROOT / "temporal_window_shap_domain_sensitivity.csv", index=False)
    pd.DataFrame(all_comparisons).to_csv(ROOT / "temporal_window_shap_ranking_comparison.csv", index=False)


if __name__ == "__main__":
    main()
