from __future__ import annotations

"""
Create Fig. 5: early-to-late reorganization of cross-validated predictor associations.

Figure output:
outputs/figures

Source-data and script output:
data/process_associations
"""

from pathlib import Path

import sys

_CODE_DIR = Path(__file__).resolve().parents[2]
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
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


# =========================================================
# 1. Paths and analysis settings
# =========================================================
ROOT = PROCESS_DATA_DIR
FIG_OUT_DIR = FIGURE_OUTPUT_DIR
DATA_OUT_DIR = ROOT / "figures_process_associations"

TABLE = ROOT / "process_response_complete_predictor_table_0p1deg.csv"
SHAP = ROOT / "crossvalidated_treeshap_importance.csv"
PARAMETERS = ROOT / "nested_spatial_xgboost_selected_parameters.csv"

EARLY = "early_1990_1999"
LATE = "late_2000_2022"
PERIOD_LABELS = {
    EARLY: "1990–1999",
    LATE: "2000–2022",
}

PROCESS_LABELS = {
    "forest_fraction_change": "Forest-cover change",
    "persistent_forest_agbd_change": "Persistent-forest AGBD change",
}

PROCESS_CONFIG = {
    "forest_fraction_change": {
        "response": "forest_fraction_slope",
        "eligible": "forest_change_state_eligible",
        "baseline": "forest_fraction_baseline",
        "ale_predictor": "PRE_mean",
    },
    "persistent_forest_agbd_change": {
        "response": "persistent_forest_agbd_slope",
        "eligible": "persistent_forest_eligible",
        "baseline": "persistent_forest_agbd_baseline",
        "ale_predictor": "Tmean_mean",
    },
}

FINAL_PREDICTORS = [
    "Tmean_mean", "PRE_mean", "WS_mean", "Tmean_slope", "PRE_slope", "SLP",
    "LHGI_mean", "LHGI_sen_slope", "NTL_log1p_mean", "POP_log1p_density_change_yr",
    "cropland_fraction", "shrubland_fraction", "grassland_fraction",
]

DISPLAY = {
    "forest_fraction_baseline": "Initial forest fraction",
    "persistent_forest_agbd_baseline": "Initial forest AGBD",
    "Tmean_mean": "Mean temperature",
    "PRE_mean": "Precipitation",
    "WS_mean": "Wind speed",
    "Tmean_slope": "Temperature trend",
    "PRE_slope": "Precipitation trend",
    "SLP": "Slope",
    "LHGI_mean": "Grazing intensity",
    "LHGI_sen_slope": "Grazing-intensity trend",
    "NTL_log1p_mean": "Nighttime light",
    "POP_log1p_density_change_yr": "Population-density change",
    "cropland_fraction": "Cropland fraction",
    "shrubland_fraction": "Shrubland fraction",
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

FIG_WIDTH_IN = 180 / 25.4
FIG_HEIGHT_IN = 5.50
ABDE_WIDTH_SCALE = 0.88  # panels a, b, d and e keep 88% of their original width
BE_SHIFT_RIGHT_CM = 0.60   # move panels b and e right by 0.50 cm
DEF_SHIFT_DOWN_CM = 0.30   # move panels d, e and f down by 0.25 cm
BE_SHIFT_RIGHT = BE_SHIFT_RIGHT_CM / (FIG_WIDTH_IN * 2.54)
DEF_SHIFT_DOWN = DEF_SHIFT_DOWN_CM / (FIG_HEIGHT_IN * 2.54)

# Morandi-style muted palette.
EARLY_COLOR = "#D2A08F"      # muted dusty salmon
LATE_COLOR = "#8FAFC0"       # muted powder blue
POSITIVE_COLOR = "#8FAFC0"   # late > early
NEGATIVE_COLOR = "#D2A08F"   # late < early
EDGE_COLOR = "#59636B"
GRID_COLOR = "#E7E4E0"
ZERO_COLOR = "#666666"


# =========================================================
# 2. Global style — matched to Fig. 4
# =========================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def add_panel_title(ax: plt.Axes, label: str, title: str) -> None:
    """Fig. 4-style panel number + concise process label outside the frame."""
    ax.set_title(
        f"({label}) {title}",
        loc="left",
        x=0.0,
        y=1.02,
        pad=3.0,
        fontsize=9.0,
        fontweight="normal",
    )


def style_panel(ax: plt.Axes, grid_axis: str | None = None) -> None:
    """Add complete four-sided frame and restrained journal-style grid."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#444444")

    ax.tick_params(direction="out", pad=1.5, length=2.5, width=0.7)
    ax.set_axisbelow(True)

    if grid_axis == "x":
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.50, zorder=0)
    elif grid_axis == "y":
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.50, zorder=0)
    elif grid_axis == "both":
        ax.grid(color=GRID_COLOR, linewidth=0.45, zorder=0)


def add_column_headers(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Shared column headers, following the Fig. 4 visual hierarchy."""
    fig.canvas.draw()
    headers = [
        "Predictor-domain reorganization",
        "Predictor-specific importance shift",
        "Fold-wise ALE",
    ]
    for col, header in enumerate(headers):
        pos = axes[col].get_position()
        x = (pos.x0 + pos.x1) / 2
        y = pos.y1 + 0.042
        fig.text(x, y, header, ha="center", va="bottom", fontsize=9.0)


def save_bundle(fig: plt.Figure, stem: str) -> None:
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_OUT_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=800,
        bbox_inches="tight",
        pad_inches=0.01,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base.with_suffix(".jpg"), dpi=300, bbox_inches="tight", pad_inches=0.01)


def copy_running_script() -> None:
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        src = Path(__file__).resolve()
        dst = (DATA_OUT_DIR / src.name).resolve()
        if src != dst:
            shutil.copy2(src, dst)
    except Exception as exc:
        print(f"Warning: could not copy the plotting script: {exc}")


# =========================================================
# 3. Data preparation
# =========================================================
def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def spatial_blocks(frame: pd.DataFrame) -> np.ndarray:
    lat_edges = np.linspace(frame.lat.min(), frame.lat.max(), 6)
    lon_edges = np.linspace(frame.lon.min(), frame.lon.max(), 7)
    lat_block = np.clip(np.digitize(frame.lat.to_numpy(), lat_edges[1:-1]), 0, 4)
    lon_block = np.clip(np.digitize(frame.lon.to_numpy(), lon_edges[1:-1]), 0, 5)
    return lat_block * 6 + lon_block


def prepare_importance(shap: pd.DataFrame, process: str):
    data = shap[(shap.process == process) & (shap.period.isin([EARLY, LATE]))].copy()
    data["group"] = data.predictor.map(
        lambda value: next(
            (group for group, members in GROUPS.items() if value in members),
            "Other",
        )
    )

    group = (
        data.groupby(["group", "period"], as_index=False)
        .relative_importance_percent.sum()
    )

    delta = (
        data.pivot(index="predictor", columns="period", values="relative_importance_percent")
        .fillna(0)
    )
    delta["delta_percent_point"] = delta[LATE] - delta[EARLY]
    delta = delta.reset_index().sort_values("delta_percent_point")
    return group, delta


# =========================================================
# 4. Plotting functions
# =========================================================
def plot_group_shift(ax: plt.Axes, group: pd.DataFrame, process: str, label: str) -> None:
    order = list(GROUPS)
    matrix = (
        group.pivot(index="group", columns="period", values="relative_importance_percent")
        .reindex(order)
        .fillna(0)
    )

    y = np.arange(len(order))
    width = 0.30

    ax.barh(
        y - width / 2,
        matrix[EARLY],
        height=width,
        color=EARLY_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.45,
        label=PERIOD_LABELS[EARLY],
        zorder=3,
    )
    ax.barh(
        y + width / 2,
        matrix[LATE],
        height=width,
        color=LATE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.45,
        label=PERIOD_LABELS[LATE],
        zorder=3,
    )

    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0, float(matrix.to_numpy().max()) * 1.22)
    ax.set_xlabel("Relative TreeSHAP contribution (%)")
    ax.legend(
        fontsize=6.5,
        loc="lower right",
        ncol=1,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.4,
        labelspacing=0.25,
    )

    style_panel(ax, "x")
    add_panel_title(ax, label, PROCESS_LABELS[process])


def plot_delta(ax: plt.Axes, delta: pd.DataFrame, process: str, label: str) -> None:
    data = delta.copy()
    colors = np.where(data.delta_percent_point >= 0, POSITIVE_COLOR, NEGATIVE_COLOR)
    labels = [DISPLAY.get(item, item) for item in data.predictor]

    ax.barh(
        labels,
        data.delta_percent_point,
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.40,
        height=0.62,
        zorder=3,
    )
    ax.axvline(0, color=ZERO_COLOR, linewidth=0.65, zorder=2)
    ax.set_xlabel("Late − early importance (percentage points)")
    ax.tick_params(axis="y", labelsize=6.5)

    style_panel(ax, "x")
    add_panel_title(ax, label, PROCESS_LABELS[process])


def model_from_parameters(row: pd.Series) -> XGBRegressor:
    params = {
        name: row[name]
        for name in [
            "n_estimators", "learning_rate", "max_depth", "min_child_weight",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        ]
    }
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=20260829,
        n_jobs=2,
        tree_method="hist",
        **params,
    )


def ale_curve(
    model: XGBRegressor,
    features: pd.DataFrame,
    predictor: str,
    quantiles: np.ndarray,
):
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

        low = features.loc[mask].copy()
        high = low.copy()
        low.loc[:, predictor] = lower
        high.loc[:, predictor] = upper

        increments.append(float(np.mean(model.predict(high) - model.predict(low))))
        centres.append(float(np.mean(values[mask])))
        counts.append(int(mask.sum()))

    effects = np.cumsum(increments)
    effects = effects - np.average(effects, weights=counts)
    q = np.linspace(5, 95, len(effects))
    return q, effects


def foldwise_ale(
    data: pd.DataFrame,
    parameters: pd.DataFrame,
    process: str,
    period: str,
) -> pd.DataFrame:
    config = PROCESS_CONFIG[process]
    feature_columns = [config["baseline"], *FINAL_PREDICTORS]
    fields = ["lon", "lat", config["response"], *feature_columns]

    eligible = as_bool(data[config["eligible"]])
    frame = data[(data.period == period) & eligible].loc[:, fields].dropna().copy()
    frame["spatial_block"] = spatial_blocks(frame)

    x = frame[feature_columns]
    y = frame[config["response"]]
    splitter = GroupKFold(n_splits=5)
    curves = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(x, y, frame.spatial_block), start=1
    ):
        parameter_row = parameters[
            (parameters.process == process)
            & (parameters.period == period)
            & (parameters.model == "full")
            & (parameters.outer_fold == fold)
        ].iloc[0]

        model = model_from_parameters(parameter_row)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])

        q, effect = ale_curve(
            model,
            x.iloc[test_idx],
            config["ale_predictor"],
            np.arange(10),
        )
        curves.append(
            pd.DataFrame({
                "process": process,
                "period": period,
                "outer_fold": fold,
                "percentile": q,
                "ale": effect,
            })
        )

    return pd.concat(curves, ignore_index=True)


def plot_ale(ax: plt.Axes, ale: pd.DataFrame, process: str, label: str) -> None:
    subset = ale[ale.process == process]
    colors = {EARLY: EARLY_COLOR, LATE: LATE_COLOR}

    for period in [EARLY, LATE]:
        summary = (
            subset[subset.period == period]
            .groupby("percentile")
            .ale.agg(["mean", "std"])
            .reset_index()
        )
        ax.plot(
            summary.percentile,
            summary["mean"],
            color=colors[period],
            linewidth=1.35,
            label=PERIOD_LABELS[period],
            zorder=4,
        )
        ax.fill_between(
            summary.percentile,
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color=colors[period],
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

    ax.axhline(0, color=ZERO_COLOR, linewidth=0.65, zorder=3)

    predictor = DISPLAY[PROCESS_CONFIG[process]["ale_predictor"]]
    unit = (
        "Forest-fraction slope"
        if process == "forest_fraction_change"
        else r"AGBD slope (Mg ha$^{-1}$ yr$^{-1}$)"
    )

    ax.set_xlabel(f"{predictor} percentile")
    ax.set_ylabel(f"Centered ALE ({unit})")
    ax.legend(
        fontsize=6.5,
        frameon=False,
        loc="best",
        handlelength=1.6,
        handletextpad=0.4,
    )

    style_panel(ax, "both")
    add_panel_title(ax, label, PROCESS_LABELS[process])


# =========================================================
# 5. Main
# =========================================================
def main() -> None:
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(TABLE)
    shap = pd.read_csv(SHAP)
    parameters = pd.read_csv(PARAMETERS)

    group_forest, delta_forest = prepare_importance(shap, "forest_fraction_change")
    group_agbd, delta_agbd = prepare_importance(shap, "persistent_forest_agbd_change")

    ale = pd.concat(
        [
            foldwise_ale(data, parameters, "forest_fraction_change", period)
            for period in [EARLY, LATE]
        ]
        + [
            foldwise_ale(data, parameters, "persistent_forest_agbd_change", period)
            for period in [EARLY, LATE]
        ],
        ignore_index=True,
    )

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.090,
        top=0.895,
        wspace=0.36,
        hspace=0.24,
        width_ratios=[1.00, 1.15, 1.00],
        height_ratios=[1.0, 1.0],
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]

    plot_group_shift(axes[0], group_forest, "forest_fraction_change", "a")
    plot_delta(axes[1], delta_forest, "forest_fraction_change", "b")
    plot_ale(axes[2], ale, "forest_fraction_change", "c")

    plot_group_shift(axes[3], group_agbd, "persistent_forest_agbd_change", "d")
    plot_delta(axes[4], delta_agbd, "persistent_forest_agbd_change", "e")
    plot_ale(axes[5], ale, "persistent_forest_agbd_change", "f")

    # Make the frames in each row exactly share top and bottom positions.
    # Panels a, b, d and e are slightly narrowed to create more horizontal
    # separation and reduce label overlap; c and f remain unchanged.
    fig.canvas.draw()
    top_ref = axes[2].get_position()
    bottom_ref = axes[5].get_position()

    for idx in (0, 1):
        pos = axes[idx].get_position()
        axes[idx].set_position([
            pos.x0,
            top_ref.y0,
            pos.width * ABDE_WIDTH_SCALE,
            top_ref.height,
        ])

    for idx in (3, 4):
        pos = axes[idx].get_position()
        axes[idx].set_position([
            pos.x0,
            bottom_ref.y0,
            pos.width * ABDE_WIDTH_SCALE,
            bottom_ref.height,
        ])

    # Move panels b and e to the right. Panel e therefore receives both
    # horizontal and vertical adjustments, as requested.
    for idx in (1, 4):
        pos = axes[idx].get_position()
        axes[idx].set_position([
            pos.x0 + BE_SHIFT_RIGHT,
            pos.y0,
            pos.width,
            pos.height,
        ])

    # Move the entire lower row d/e/f downward while preserving their heights
    # and their shared top/bottom alignment.
    for idx in (3, 4, 5):
        pos = axes[idx].get_position()
        axes[idx].set_position([
            pos.x0,
            pos.y0 - DEF_SHIFT_DOWN,
            pos.width,
            pos.height,
        ])

    add_column_headers(fig, axes)

    # ---------- Source-data output ----------
    pd.concat(
        [
            group_forest.assign(process="forest_fraction_change"),
            group_agbd.assign(process="persistent_forest_agbd_change"),
        ],
        ignore_index=True,
    ).to_csv(DATA_OUT_DIR / "Fig5_source_predictor_domain_shift.csv", index=False)

    pd.concat(
        [
            delta_forest.assign(process="forest_fraction_change"),
            delta_agbd.assign(process="persistent_forest_agbd_change"),
        ],
        ignore_index=True,
    ).to_csv(DATA_OUT_DIR / "Fig5_source_predictor_importance_shift.csv", index=False)

    ale.to_csv(DATA_OUT_DIR / "Fig5_source_foldwise_ale.csv", index=False)

    save_bundle(fig, "Fig5_temporal_reorganization_and_ale_refined")
    copy_running_script()
    plt.close(fig)

    print(f"Figure files written to: {FIG_OUT_DIR}")
    print(f"Source data and script written to: {DATA_OUT_DIR}")


if __name__ == "__main__":
    main()
