"""Create Fig. 5: early-to-late reorganization of cross-validated predictor associations."""

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


ROOT = PROCESS_DATA_DIR
FIG_DIR = ROOT / "figures_process_associations"
TABLE = ROOT / "process_response_complete_predictor_table_0p1deg.csv"
SHAP = ROOT / "crossvalidated_treeshap_importance.csv"
PARAMETERS = ROOT / "nested_spatial_xgboost_selected_parameters.csv"
EARLY, LATE = "early_1990_1999", "late_2000_2022"
PERIOD_LABELS = {EARLY: "1990–1999", LATE: "2000–2022"}
PROCESS_CONFIG = {
    "forest_fraction_change": {
        "response": "forest_fraction_slope", "eligible": "forest_change_state_eligible",
        "baseline": "forest_fraction_baseline", "ale_predictor": "PRE_mean",
    },
    "persistent_forest_agbd_change": {
        "response": "persistent_forest_agbd_slope", "eligible": "persistent_forest_eligible",
        "baseline": "persistent_forest_agbd_baseline", "ale_predictor": "Tmean_mean",
    },
}
FINAL_PREDICTORS = [
    "Tmean_mean", "PRE_mean", "WS_mean", "Tmean_slope", "PRE_slope", "SLP",
    "LHGI_mean", "LHGI_sen_slope", "NTL_log1p_mean", "POP_log1p_density_change_yr",
    "cropland_fraction", "shrubland_fraction", "grassland_fraction",
]
DISPLAY = {
    "forest_fraction_baseline": "Initial forest fraction", "persistent_forest_agbd_baseline": "Initial forest AGBD",
    "Tmean_mean": "Mean temperature", "PRE_mean": "Precipitation", "WS_mean": "Wind speed",
    "Tmean_slope": "Temperature trend", "PRE_slope": "Precipitation trend", "SLP": "Slope",
    "LHGI_mean": "Grazing intensity", "LHGI_sen_slope": "Grazing-intensity trend",
    "NTL_log1p_mean": "Nighttime light", "POP_log1p_density_change_yr": "Population-density change",
    "cropland_fraction": "Cropland fraction", "shrubland_fraction": "Shrubland fraction",
    "grassland_fraction": "Grassland fraction",
}
GROUPS = {
    "Initial state": ["forest_fraction_baseline", "persistent_forest_agbd_baseline"],
    "Climate background": ["Tmean_mean", "PRE_mean", "WS_mean"],
    "Climate change": ["Tmean_slope", "PRE_slope"],
    "Terrain": ["SLP"],
    "Land-cover context": ["cropland_fraction", "shrubland_fraction", "grassland_fraction"],
    "Human pressure": ["LHGI_mean", "LHGI_sen_slope", "NTL_log1p_mean", "POP_log1p_density_change_yr"],
}

mpl.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8.0, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def panel_label(ax, label):
    ax.text(0.015, 0.985, f"({label})", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=10.5,
            zorder=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.2})


def spatial_blocks(frame):
    lat_edges = np.linspace(frame.lat.min(), frame.lat.max(), 6)
    lon_edges = np.linspace(frame.lon.min(), frame.lon.max(), 7)
    lat_block = np.clip(np.digitize(frame.lat.to_numpy(), lat_edges[1:-1]), 0, 4)
    lon_block = np.clip(np.digitize(frame.lon.to_numpy(), lon_edges[1:-1]), 0, 5)
    return lat_block * 6 + lon_block


def prepare_importance(shap, process):
    data = shap[(shap.process == process) & (shap.period.isin([EARLY, LATE]))].copy()
    data["group"] = data.predictor.map(lambda value: next((group for group, members in GROUPS.items() if value in members), "Other"))
    group = data.groupby(["group", "period"], as_index=False).relative_importance_percent.sum()
    delta = data.pivot(index="predictor", columns="period", values="relative_importance_percent").fillna(0)
    delta["delta_percent_point"] = delta[LATE] - delta[EARLY]
    delta = delta.reset_index().sort_values("delta_percent_point")
    return group, delta


def plot_group_shift(ax, group, process, label):
    order = list(GROUPS)
    matrix = group.pivot(index="group", columns="period", values="relative_importance_percent").reindex(order).fillna(0)
    y = np.arange(len(order)); width = 0.34
    ax.barh(y - width / 2, matrix[EARLY], height=width, color="#D9875A", label=PERIOD_LABELS[EARLY])
    ax.barh(y + width / 2, matrix[LATE], height=width, color="#3E7FA5", label=PERIOD_LABELS[LATE])
    ax.set_yticks(y, order); ax.invert_yaxis(); ax.set_xlim(0, max(matrix.max()) * 1.24)
    ax.set_xlabel("Relative TreeSHAP contribution (%)")
    process_title = "Forest-cover change" if process == "forest_fraction_change" else "Persistent-forest AGBD change"
    ax.set_title(f"{process_title}\nPredictor-domain reorganization", fontsize=8.3, pad=4, linespacing=1.1)
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.6)
    ax.legend(fontsize=6.3, loc="lower right", frameon=False)
    panel_label(ax, label)


def plot_delta(ax, delta, process, label):
    data = delta.copy()
    colours = np.where(data.delta_percent_point >= 0, "#3E7FA5", "#D9875A")
    ax.barh([DISPLAY.get(item, item) for item in data.predictor], data.delta_percent_point, color=colours, edgecolor="#4A5560", linewidth=0.35)
    ax.axvline(0, color="#555555", linewidth=0.7)
    ax.set_xlabel("Late − early importance (percentage points)")
    process_title = "Forest-cover change" if process == "forest_fraction_change" else "Persistent-forest AGBD change"
    ax.set_title(f"{process_title}\nPredictor-specific importance shift", fontsize=8.3, pad=4, linespacing=1.1)
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.6)
    panel_label(ax, label)


def model_from_parameters(row):
    params = {name: row[name] for name in ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    return XGBRegressor(objective="reg:squarederror", random_state=20260829, n_jobs=2, tree_method="hist", **params)


def ale_curve(model, features, predictor, quantiles):
    values = features[predictor].to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, len(quantiles) + 1)))
    if len(edges) < 4:
        raise ValueError(f"Insufficient unique values for ALE: {predictor}")
    increments, centres, counts = [], [], []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (values >= lower) & (values < upper)
        if upper == edges[-1]:
            mask = (values >= lower) & (values <= upper)
        if not mask.any():
            continue
        low = features.loc[mask].copy(); high = low.copy()
        low.loc[:, predictor] = lower; high.loc[:, predictor] = upper
        increments.append(float(np.mean(model.predict(high) - model.predict(low))))
        centres.append(float(np.mean(values[mask]))); counts.append(mask.sum())
    effects = np.cumsum(increments)
    effects = effects - np.average(effects, weights=counts)
    q = np.linspace(5, 95, len(effects))
    return q, effects


def foldwise_ale(data, parameters, process, period):
    config = PROCESS_CONFIG[process]
    feature_columns = [config["baseline"], *FINAL_PREDICTORS]
    fields = ["lon", "lat", config["response"], *feature_columns]
    frame = data[(data.period == period) & data[config["eligible"]]].loc[:, fields].dropna().copy()
    frame["spatial_block"] = spatial_blocks(frame)
    x = frame[feature_columns]; y = frame[config["response"]]
    splitter = GroupKFold(n_splits=5)
    curves = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, frame.spatial_block), start=1):
        parameter_row = parameters[(parameters.process == process) & (parameters.period == period) & (parameters.model == "full") & (parameters.outer_fold == fold)].iloc[0]
        model = model_from_parameters(parameter_row)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        q, effect = ale_curve(model, x.iloc[test_idx], config["ale_predictor"], np.arange(10))
        curves.append(pd.DataFrame({"process": process, "period": period, "outer_fold": fold, "percentile": q, "ale": effect}))
    return pd.concat(curves, ignore_index=True)


def plot_ale(ax, ale, process, label):
    subset = ale[ale.process == process]
    colours = {EARLY: "#D9875A", LATE: "#3E7FA5"}
    for period in [EARLY, LATE]:
        summary = subset[subset.period == period].groupby("percentile").ale.agg(["mean", "std"]).reset_index()
        ax.plot(summary.percentile, summary["mean"], color=colours[period], linewidth=1.7, label=PERIOD_LABELS[period])
        ax.fill_between(summary.percentile, summary["mean"] - summary["std"], summary["mean"] + summary["std"], color=colours[period], alpha=0.18, linewidth=0)
    ax.axhline(0, color="#555555", linewidth=0.7)
    predictor = DISPLAY[PROCESS_CONFIG[process]["ale_predictor"]]
    unit = "Forest-fraction slope" if process == "forest_fraction_change" else "AGBD slope (Mg ha$^{-1}$ yr$^{-1}$)"
    ax.set_xlabel(f"{predictor} percentile")
    ax.set_ylabel(f"Centered ALE ({unit})")
    ax.set_title(f"Fold-wise ALE: {predictor}", fontsize=8.8, pad=4)
    ax.grid(color="#E7E7E7", linewidth=0.6)
    ax.legend(fontsize=6.5, frameon=False, loc="best")
    panel_label(ax, label)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLE); shap = pd.read_csv(SHAP); parameters = pd.read_csv(PARAMETERS)
    group_forest, delta_forest = prepare_importance(shap, "forest_fraction_change")
    group_agbd, delta_agbd = prepare_importance(shap, "persistent_forest_agbd_change")
    ale = pd.concat([
        foldwise_ale(data, parameters, "forest_fraction_change", period) for period in [EARLY, LATE]
    ] + [
        foldwise_ale(data, parameters, "persistent_forest_agbd_change", period) for period in [EARLY, LATE]
    ], ignore_index=True)

    fig = plt.figure(figsize=(7.35, 5.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.0])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    plot_group_shift(axes[0], group_forest, "forest_fraction_change", "a")
    plot_delta(axes[1], delta_forest, "forest_fraction_change", "b")
    plot_ale(axes[2], ale, "forest_fraction_change", "c")
    plot_group_shift(axes[3], group_agbd, "persistent_forest_agbd_change", "d")
    plot_delta(axes[4], delta_agbd, "persistent_forest_agbd_change", "e")
    plot_ale(axes[5], ale, "persistent_forest_agbd_change", "f")

    pd.concat([group_forest.assign(process="forest_fraction_change"), group_agbd.assign(process="persistent_forest_agbd_change")], ignore_index=True).to_csv(FIG_DIR / "Fig5_source_predictor_domain_shift.csv", index=False)
    pd.concat([delta_forest.assign(process="forest_fraction_change"), delta_agbd.assign(process="persistent_forest_agbd_change")], ignore_index=True).to_csv(FIG_DIR / "Fig5_source_predictor_importance_shift.csv", index=False)
    ale.to_csv(FIG_DIR / "Fig5_source_foldwise_ale.csv", index=False)
    base = FIG_DIR / "Fig5_temporal_reorganization_and_ale"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
