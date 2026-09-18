"""Nested spatial XGBoost models for separated forest-cover and AGBD processes.

The workflow compares state-only and full models under outer spatial GroupKFold
validation. Hyperparameters are selected solely within outer-training blocks.
TreeSHAP contributions are calculated with XGBoost's native pred_contribs output
on outer-test blocks, avoiding training-sample explanation bias.
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

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterSampler
from xgboost import DMatrix, XGBRegressor


ROOT = PROCESS_DATA_DIR
TABLE = ROOT / "process_response_complete_predictor_table_0p1deg.csv"
MAIN_PERIODS = ["early_1990_1999", "late_2000_2022", "full_1990_2022"]
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_BLOCK_LAT = 5
N_BLOCK_LON = 6
RANDOM_SEED = 20260829

FINAL_PREDICTORS = [
    "Tmean_mean", "PRE_mean", "WS_mean", "Tmean_slope", "PRE_slope", "SLP",
    "LHGI_mean", "LHGI_sen_slope", "NTL_log1p_mean", "POP_log1p_density_change_yr",
    "cropland_fraction", "shrubland_fraction", "grassland_fraction",
]
PROCESS_CONFIG = {
    "forest_fraction_change": {
        "response": "forest_fraction_slope",
        "eligible": "forest_change_state_eligible",
        "baseline": "forest_fraction_baseline",
    },
    "persistent_forest_agbd_change": {
        "response": "persistent_forest_agbd_slope",
        "eligible": "persistent_forest_eligible",
        "baseline": "persistent_forest_agbd_baseline",
    },
}


def spatial_blocks(frame: pd.DataFrame) -> np.ndarray:
    """Fixed 5x6 geographic bins; all cells in a bin form one contiguous block."""
    lat_edges = np.linspace(frame.lat.min(), frame.lat.max(), N_BLOCK_LAT + 1)
    lon_edges = np.linspace(frame.lon.min(), frame.lon.max(), N_BLOCK_LON + 1)
    lat_block = np.clip(np.digitize(frame.lat.to_numpy(), lat_edges[1:-1]), 0, N_BLOCK_LAT - 1)
    lon_block = np.clip(np.digitize(frame.lon.to_numpy(), lon_edges[1:-1]), 0, N_BLOCK_LON - 1)
    return lat_block * N_BLOCK_LON + lon_block


def estimator(params=None):
    kwargs = dict(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=2,
        tree_method="hist",
    )
    if params:
        kwargs.update(params)
    return XGBRegressor(**kwargs)


def parameter_candidates():
    distributions = {
        "n_estimators": randint(200, 501),
        "max_depth": randint(2, 7),
        "learning_rate": uniform(0.025, 0.105),
        "min_child_weight": randint(1, 11),
        "subsample": uniform(0.60, 0.35),
        "colsample_bytree": uniform(0.60, 0.35),
        "reg_alpha": uniform(0.0, 0.08),
        "reg_lambda": uniform(0.6, 1.4),
    }
    return list(ParameterSampler(distributions, n_iter=8, random_state=RANDOM_SEED))


def select_parameters(x, y, groups):
    candidates = parameter_candidates()
    splitter = GroupKFold(n_splits=N_INNER_FOLDS)
    best_params, best_rmse = None, np.inf
    for params in candidates:
        fold_errors = []
        for train_idx, validation_idx in splitter.split(x, y, groups):
            model = estimator(params)
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            prediction = model.predict(x.iloc[validation_idx])
            fold_errors.append(mean_squared_error(y.iloc[validation_idx], prediction, squared=False))
        score = float(np.mean(fold_errors))
        if score < best_rmse:
            best_params, best_rmse = params, score
    return best_params, best_rmse


def moran_i(values, coordinates, k=8):
    """Global k-nearest-neighbour Moran's I with row-standardised weights."""
    from sklearn.neighbors import NearestNeighbors

    values = np.asarray(values, dtype=float)
    centred = values - np.mean(values)
    neighbours = NearestNeighbors(n_neighbors=min(k + 1, len(values))).fit(coordinates).kneighbors(return_distance=False)[:, 1:]
    numerator = sum(np.sum(centred[i] * centred[neighbours[i]]) / len(neighbours[i]) for i in range(len(values)))
    denominator = np.sum(centred ** 2)
    return float(numerator / denominator) if denominator > 0 else np.nan


def evaluate_model(frame, feature_columns, model_label, process, period):
    x = frame[feature_columns]
    y = frame["response"]
    groups = frame["spatial_block"].to_numpy()
    outer = GroupKFold(n_splits=N_OUTER_FOLDS)
    predictions = np.full(len(frame), np.nan)
    shap_rows, parameter_rows, fold_rows = [], [], []

    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if model_label == "state_only":
            params, inner_rmse = {}, np.nan
        else:
            params, inner_rmse = select_parameters(x_train, y_train, groups[train_idx])
        model = estimator(params)
        model.fit(x_train, y_train)
        test_prediction = model.predict(x_test)
        predictions[test_idx] = test_prediction

        fold_rows.append({
            "process": process, "period": period, "model": model_label, "outer_fold": fold,
            "n_train": len(train_idx), "n_test": len(test_idx), "n_train_blocks": len(np.unique(groups[train_idx])),
            "n_test_blocks": len(np.unique(groups[test_idx])), "inner_cv_rmse": inner_rmse,
            "R2": r2_score(y_test, test_prediction),
            "RMSE": mean_squared_error(y_test, test_prediction, squared=False),
            "MAE": mean_absolute_error(y_test, test_prediction),
        })
        parameter_rows.append({"process": process, "period": period, "model": model_label, "outer_fold": fold, **params})

        if model_label == "full":
            contribution = model.get_booster().predict(DMatrix(x_test, feature_names=feature_columns), pred_contribs=True)[:, :-1]
            for feature_index, feature in enumerate(feature_columns):
                shap_rows.append({
                    "process": process, "period": period, "outer_fold": fold, "predictor": feature,
                    "mean_abs_treeshap": float(np.mean(np.abs(contribution[:, feature_index]))),
                    "mean_treeshap": float(np.mean(contribution[:, feature_index])),
                })

    metrics = {
        "process": process, "period": period, "model": model_label,
        "n_samples": len(frame), "n_blocks": len(np.unique(groups)),
        "R2": r2_score(y, predictions),
        "RMSE": mean_squared_error(y, predictions, squared=False),
        "MAE": mean_absolute_error(y, predictions),
        "response_moran_I": moran_i(y, frame[["lon", "lat"]].to_numpy()),
        "residual_moran_I": moran_i(y - predictions, frame[["lon", "lat"]].to_numpy()),
    }
    prediction_table = frame[["lon", "lat", "ecological_zone_id", "spatial_block", "response"]].copy()
    prediction_table["process"] = process
    prediction_table["period"] = period
    prediction_table["model"] = model_label
    prediction_table["oof_prediction"] = predictions
    prediction_table["oof_residual"] = y.to_numpy() - predictions
    return metrics, fold_rows, parameter_rows, shap_rows, prediction_table


def main() -> None:
    data = pd.read_csv(TABLE)
    all_metrics, all_folds, all_parameters, all_shap, all_predictions = [], [], [], [], []
    for process, config in PROCESS_CONFIG.items():
        for period in MAIN_PERIODS:
            fields = ["lon", "lat", "ecological_zone_id", config["response"], config["baseline"], *FINAL_PREDICTORS]
            subset = data[(data.period == period) & data[config["eligible"]]].loc[:, fields].dropna().copy()
            subset = subset.rename(columns={config["response"]: "response"})
            subset["spatial_block"] = spatial_blocks(subset)
            for label, features in {
                "state_only": [config["baseline"]],
                "full": [config["baseline"], *FINAL_PREDICTORS],
            }.items():
                print(f"Fitting {process}, {period}, {label} (n={len(subset)})", flush=True)
                metrics, folds, parameters, shap, predictions = evaluate_model(subset, features, label, process, period)
                all_metrics.append(metrics)
                all_folds.extend(folds)
                all_parameters.extend(parameters)
                all_shap.extend(shap)
                all_predictions.append(predictions)

    performance = pd.DataFrame(all_metrics)
    performance.to_csv(ROOT / "nested_spatial_xgboost_performance.csv", index=False)
    pd.DataFrame(all_folds).to_csv(ROOT / "nested_spatial_xgboost_fold_metrics.csv", index=False)
    pd.DataFrame(all_parameters).to_csv(ROOT / "nested_spatial_xgboost_selected_parameters.csv", index=False)
    shap = pd.DataFrame(all_shap)
    summary = shap.groupby(["process", "period", "predictor"], as_index=False).agg(
        mean_abs_treeshap=("mean_abs_treeshap", "mean"),
        sd_across_outer_folds=("mean_abs_treeshap", "std"),
        mean_signed_treeshap=("mean_treeshap", "mean"),
    )
    summary["relative_importance_percent"] = summary.groupby(["process", "period"])["mean_abs_treeshap"].transform(lambda x: 100 * x / x.sum())
    summary.to_csv(ROOT / "crossvalidated_treeshap_importance.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(ROOT / "nested_spatial_xgboost_oof_predictions.csv", index=False)


if __name__ == "__main__":
    main()
