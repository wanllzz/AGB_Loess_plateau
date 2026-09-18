"""Create Fig. 4: full-period predictive associations for two AGB processes."""

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


ROOT = PROCESS_DATA_DIR
FIG_DIR = ROOT / "figures_process_associations"
PERFORMANCE = ROOT / "nested_spatial_xgboost_performance.csv"
SHAP = ROOT / "crossvalidated_treeshap_importance.csv"
OOF = ROOT / "nested_spatial_xgboost_oof_predictions.csv"
PERIOD = "full_1990_2022"
PROCESS_LABELS = {
    "forest_fraction_change": "Forest-cover change",
    "persistent_forest_agbd_change": "Persistent-forest AGBD change",
}
DISPLAY = {
    "forest_fraction_baseline": "Initial forest fraction",
    "persistent_forest_agbd_baseline": "Initial forest AGBD",
    "Tmean_mean": "Mean temperature", "PRE_mean": "Precipitation", "WS_mean": "Wind speed",
    "Tmean_slope": "Temperature trend", "PRE_slope": "Precipitation trend", "SLP": "Slope",
    "LHGI_mean": "Grazing intensity", "LHGI_sen_slope": "Grazing-intensity trend",
    "NTL_log1p_mean": "Nighttime light", "POP_log1p_density_change_yr": "Population-density change",
    "cropland_fraction": "Cropland fraction", "shrubland_fraction": "Shrubland fraction",
    "grassland_fraction": "Grassland fraction",
}

mpl.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def panel_label(ax, label):
    ax.text(0.015, 0.985, f"({label})", transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=10.5,
            zorder=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.2})


def plot_performance(ax, performance, process, label):
    row = performance[(performance.process == process) & (performance.period == PERIOD)].set_index("model")
    values = row.loc[["state_only", "full"], "R2"]
    bars = ax.bar([0, 1], values, width=0.58, color=["#B8C2CC", "#2D6A8E"], edgecolor="#2F3941", linewidth=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035 if value >= 0 else value - 0.045,
                f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7.3)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks([0, 1], ["State-only", "Full"])
    ax.set_ylim(-0.10, 0.92)
    response = float(row.loc["full", "response_moran_I"])
    residual = float(row.loc["full", "residual_moran_I"])
    ax.set_ylabel("Spatial OOF R$^2$")
    ax.set_title(f"{PROCESS_LABELS[process]}\nMoran's I: {response:.3f} → {residual:.3f}", fontsize=8.8, pad=5, linespacing=1.1)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    panel_label(ax, label)


def plot_importance(ax, shap, process, label):
    data = shap[(shap.process == process) & (shap.period == PERIOD)].copy()
    data = data.sort_values("relative_importance_percent", ascending=True).tail(10)
    labels = [DISPLAY.get(item, item) for item in data.predictor]
    bars = ax.barh(labels, data.mean_abs_treeshap, xerr=data.sd_across_outer_folds, capsize=2,
                   color="#4682A9", edgecolor="#2F4A5C", linewidth=0.45)
    xmax = (data.mean_abs_treeshap + data.sd_across_outer_folds).max() * 1.42
    ax.set_xlim(0, xmax)
    for bar, percentage in zip(bars, data.relative_importance_percent):
        ax.text(bar.get_width() + xmax * 0.025, bar.get_y() + bar.get_height() / 2, f"{percentage:.1f}%",
                va="center", ha="left", fontsize=6.3)
    ax.set_xlabel("Mean absolute TreeSHAP value")
    ax.tick_params(axis="y", labelsize=6.8)
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.6)
    ax.set_title(f"{PROCESS_LABELS[process]} predictors", fontsize=8.8, pad=4)
    panel_label(ax, label)
    return data


def plot_residual_map(ax, oof, process, label):
    data = oof[(oof.process == process) & (oof.period == PERIOD) & (oof.model == "full")].copy()
    limit = np.quantile(np.abs(data.oof_residual), 0.98)
    scatter = ax.scatter(data.lon, data.lat, c=data.oof_residual, s=10.5, marker="s", linewidths=0,
                         cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
    ax.set_aspect("equal"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"{PROCESS_LABELS[process]} OOF residuals", fontsize=8.8, pad=4)
    cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.065, pad=0.10)
    unit = "Forest-fraction slope residual (yr$^{-1}$)" if process == "forest_fraction_change" else "AGBD-slope residual (Mg ha$^{-1}$ yr$^{-1}$)"
    cbar.set_label(unit, fontsize=6.3); cbar.ax.tick_params(labelsize=6.3)
    panel_label(ax, label)
    return data


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    performance = pd.read_csv(PERFORMANCE)
    shap = pd.read_csv(SHAP)
    oof = pd.read_csv(OOF)

    fig = plt.figure(figsize=(7.35, 5.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[0.88, 1.12, 1.18], height_ratios=[1, 1])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]

    plot_performance(axes[0], performance, "forest_fraction_change", "a")
    importance_forest = plot_importance(axes[1], shap, "forest_fraction_change", "b")
    residual_forest = plot_residual_map(axes[2], oof, "forest_fraction_change", "c")
    plot_performance(axes[3], performance, "persistent_forest_agbd_change", "d")
    importance_agbd = plot_importance(axes[4], shap, "persistent_forest_agbd_change", "e")
    residual_agbd = plot_residual_map(axes[5], oof, "persistent_forest_agbd_change", "f")

    performance[performance.period == PERIOD].to_csv(FIG_DIR / "Fig4_source_full_period_performance.csv", index=False)
    pd.concat([importance_forest, importance_agbd], ignore_index=True).to_csv(FIG_DIR / "Fig4_source_full_period_treeshap.csv", index=False)
    pd.concat([residual_forest, residual_agbd], ignore_index=True).to_csv(FIG_DIR / "Fig4_source_oof_residuals_full_period.csv", index=False)
    base = FIG_DIR / "Fig4_full_period_predictive_associations"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
