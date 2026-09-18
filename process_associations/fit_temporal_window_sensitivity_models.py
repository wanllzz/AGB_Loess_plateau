"""Run non-overwriting temporal-window sensitivity models for the AGB study."""

import argparse
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
import traceback

import pandas as pd

from fit_nested_spatial_xgboost_process_models import (
    FINAL_PREDICTORS,
    PROCESS_CONFIG,
    evaluate_model,
    spatial_blocks,
)


ROOT = PROCESS_DATA_DIR
TABLE = ROOT / "process_response_complete_predictor_table_0p1deg.csv"
SENSITIVITY_PERIODS = ["late_equal_2000_2009", "late_recent_2013_2022"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--process",
        choices=tuple(PROCESS_CONFIG),
        required=True,
        help="One process per run; keeps each spatial nested-model task bounded.",
    )
    parser.add_argument("--period", choices=SENSITIVITY_PERIODS, required=True)
    args = parser.parse_args()
    data = pd.read_csv(TABLE)
    all_metrics, all_folds, all_parameters, all_shap, all_predictions = [], [], [], [], []
    process = args.process
    config = PROCESS_CONFIG[process]
    period = args.period
    fields = [
        "lon", "lat", "ecological_zone_id", config["response"], config["baseline"], *FINAL_PREDICTORS,
    ]
    subset = data[(data.period == period) & data[config["eligible"]]].loc[:, fields].dropna().copy()
    subset = subset.rename(columns={config["response"]: "response"})
    subset["spatial_block"] = spatial_blocks(subset)
    for label, features in {
        "state_only": [config["baseline"]],
        "full": [config["baseline"], *FINAL_PREDICTORS],
    }.items():
        print(f"Fitting {process}, {period}, {label} (n={len(subset)})", flush=True)
        metrics, folds, parameters, shap, predictions = evaluate_model(
            subset, features, label, process, period
        )
        all_metrics.append(metrics)
        all_folds.extend(folds)
        all_parameters.extend(parameters)
        all_shap.extend(shap)
        all_predictions.append(predictions)

    prefix = f"temporal_window_sensitivity_{process}_{period}"
    pd.DataFrame(all_metrics).to_csv(ROOT / f"{prefix}_performance.csv", index=False)
    pd.DataFrame(all_folds).to_csv(ROOT / f"{prefix}_fold_metrics.csv", index=False)
    pd.DataFrame(all_parameters).to_csv(
        ROOT / f"{prefix}_selected_parameters.csv", index=False
    )
    shap = pd.DataFrame(all_shap)
    summary = shap.groupby(["process", "period", "predictor"], as_index=False).agg(
        mean_abs_treeshap=("mean_abs_treeshap", "mean"),
        sd_across_outer_folds=("mean_abs_treeshap", "std"),
        mean_signed_treeshap=("mean_treeshap", "mean"),
    )
    summary["relative_importance_percent"] = summary.groupby(["process", "period"])[
        "mean_abs_treeshap"
    ].transform(lambda values: 100 * values / values.sum())
    summary.to_csv(ROOT / f"{prefix}_treeshap_importance.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        ROOT / f"{prefix}_oof_predictions.csv", index=False
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
