"""Common-sample sensitivity for early-to-late persistent-forest AGBD associations."""

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
from sklearn.model_selection import GroupKFold

from fit_nested_spatial_xgboost_process_models import (
    FINAL_PREDICTORS,
    evaluate_model,
    spatial_blocks,
)


ROOT = PROCESS_DATA_DIR
TABLE = ROOT / "process_response_complete_predictor_table_0p1deg.csv"
PRIMARY_SHAP = ROOT / "crossvalidated_treeshap_importance.csv"
EARLY = "early_1990_1999"
LATE = "late_2000_2022"
PROCESS = "persistent_forest_agbd_change"
RESPONSE = "persistent_forest_agbd_slope"
BASELINE = "persistent_forest_agbd_baseline"


def summary_shap(raw):
    result = raw.groupby(["process", "period", "predictor"], as_index=False).agg(
        mean_abs_treeshap=("mean_abs_treeshap", "mean"),
        sd_across_outer_folds=("mean_abs_treeshap", "std"),
        mean_signed_treeshap=("mean_treeshap", "mean"),
    )
    result["relative_importance_percent"] = result.groupby(["process", "period"])["mean_abs_treeshap"].transform(
        lambda values: 100 * values / values.sum()
    )
    return result


def main():
    data = pd.read_csv(TABLE)
    feature_columns = [BASELINE, *FINAL_PREDICTORS]
    fields = ["lon", "lat", "ecological_zone_id", RESPONSE, *feature_columns, "persistent_forest_eligible"]
    early = data[data.period == EARLY].loc[:, fields].copy()
    late = data[data.period == LATE].loc[:, fields].copy()
    early["common_complete"] = early.persistent_forest_eligible & early[[RESPONSE, *feature_columns]].notna().all(axis=1)
    late["common_complete"] = late.persistent_forest_eligible & late[[RESPONSE, *feature_columns]].notna().all(axis=1)
    common_grid = early.loc[early.common_complete, ["lon", "lat"]].merge(
        late.loc[late.common_complete, ["lon", "lat"]], on=["lon", "lat"], how="inner"
    ).drop_duplicates().sort_values(["lat", "lon"]).reset_index(drop=True)
    common_grid["spatial_block"] = spatial_blocks(common_grid)
    groups = common_grid.spatial_block.to_numpy()
    fold_id = np.zeros(len(common_grid), dtype=int)
    for fold, (_, test_idx) in enumerate(GroupKFold(n_splits=5).split(common_grid, groups=groups), start=1):
        fold_id[test_idx] = fold
    common_grid["outer_fold"] = fold_id

    all_metrics, all_folds, all_params, all_shap, all_predictions = [], [], [], [], []
    for period, source in [(EARLY, early), (LATE, late)]:
        frame = common_grid.merge(source.drop(columns=["common_complete"]), on=["lon", "lat"], how="left")
        frame = frame.sort_values(["lat", "lon"]).reset_index(drop=True)
        frame = frame.rename(columns={RESPONSE: "response"})
        if not np.array_equal(frame.outer_fold.to_numpy(), fold_id):
            raise RuntimeError("Outer-fold assignment changed after stage merge.")
        for label, features in {
            "state_only": [BASELINE],
            "full": feature_columns,
        }.items():
            print(f"Fitting {PROCESS}, {period}, {label} on common sample (n={len(frame)})", flush=True)
            metrics, folds, parameters, shap, predictions = evaluate_model(frame, features, label, PROCESS, period)
            all_metrics.append(metrics)
            all_folds.extend(folds)
            all_params.extend(parameters)
            all_shap.extend(shap)
            all_predictions.append(predictions)

    performance = pd.DataFrame(all_metrics)
    folds = pd.DataFrame(all_folds)
    raw_shap = pd.DataFrame(all_shap)
    shap = summary_shap(raw_shap)
    fold_relative = raw_shap.copy()
    fold_relative["relative_importance_percent"] = fold_relative.groupby(["period", "outer_fold"])["mean_abs_treeshap"].transform(
        lambda values: 100 * values / values.sum()
    )
    fold_shift = fold_relative.pivot(index=["outer_fold", "predictor"], columns="period", values="relative_importance_percent").reset_index()
    fold_shift["late_minus_early_pp"] = fold_shift[LATE] - fold_shift[EARLY]
    fold_shift_summary = fold_shift.groupby("predictor", as_index=False).agg(
        mean_delta_pp=("late_minus_early_pp", "mean"),
        sd_delta_pp=("late_minus_early_pp", "std"),
        positive_fold_count=("late_minus_early_pp", lambda values: int((values > 0).sum())),
        negative_fold_count=("late_minus_early_pp", lambda values: int((values < 0).sum())),
    )
    primary = pd.read_csv(PRIMARY_SHAP)
    correlation_rows = []
    for period in [EARLY, LATE]:
        common_rank = shap[shap.period == period].set_index("predictor").relative_importance_percent
        primary_rank = primary[(primary.process == PROCESS) & (primary.period == period)].set_index("predictor").relative_importance_percent
        shared = common_rank.index.intersection(primary_rank.index)
        rho, p_value = spearmanr(common_rank.loc[shared], primary_rank.loc[shared])
        top_common = set(common_rank.sort_values(ascending=False).head(5).index)
        top_primary = set(primary_rank.sort_values(ascending=False).head(5).index)
        correlation_rows.append({
            "period": period, "common_n": len(common_grid), "ranking_spearman_rho_vs_primary": rho,
            "ranking_p_value": p_value, "top5_overlap_count": len(top_common & top_primary),
            "common_top5": "; ".join(common_rank.sort_values(ascending=False).head(5).index),
            "primary_top5": "; ".join(primary_rank.sort_values(ascending=False).head(5).index),
        })
    comparison = pd.DataFrame(correlation_rows)

    performance.to_csv(ROOT / "common_sample_persistent_agbd_performance.csv", index=False)
    folds.to_csv(ROOT / "common_sample_persistent_agbd_fold_metrics.csv", index=False)
    pd.DataFrame(all_params).to_csv(ROOT / "common_sample_persistent_agbd_selected_parameters.csv", index=False)
    raw_shap.to_csv(ROOT / "common_sample_persistent_agbd_fold_treeshap.csv", index=False)
    shap.to_csv(ROOT / "common_sample_persistent_agbd_treeshap_importance.csv", index=False)
    fold_shift.to_csv(ROOT / "common_sample_persistent_agbd_foldwise_importance_shift.csv", index=False)
    fold_shift_summary.to_csv(ROOT / "common_sample_persistent_agbd_foldwise_importance_shift_summary.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(ROOT / "common_sample_persistent_agbd_oof_predictions.csv", index=False)
    common_grid.to_csv(ROOT / "common_sample_persistent_agbd_grid_and_folds.csv", index=False)
    comparison.to_csv(ROOT / "common_sample_persistent_agbd_comparison.csv", index=False)


if __name__ == "__main__":
    main()
