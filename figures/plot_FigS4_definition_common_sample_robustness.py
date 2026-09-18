"""Plot Fig. S4: forest-definition and common-sample robustness."""

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = PROCESS_DATA_DIR
OUTDIR = FIGURE_OUTPUT_DIR / "supplementary"
THRESHOLD = ROOT / "persistent_forest_threshold_sensitivity_reference_comparison.csv"
COMMON = ROOT / "Table_S14_common_sample_performance_and_concordance.csv"
PRIMARY_PERFORMANCE = ROOT / "nested_spatial_xgboost_performance.csv"

PERIOD_ORDER = ["early_1990_1999", "late_2000_2022", "full_1990_2022"]
PERIOD_LABELS = {
    "early_1990_1999": "1990–1999",
    "late_2000_2022": "2000–2022",
    "full_1990_2022": "1990–2022",
}
PERIOD_COLORS = {"early_1990_1999": "#C98E86", "late_2000_2022": "#7DA6BE", "full_1990_2022": "#7EA28A"}


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7.5,
        "axes.titlesize": 9.0, "axes.labelsize": 8.0, "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0, "legend.fontsize": 6.7, "axes.linewidth": 0.65,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def style_axis(ax, grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#444444")
    ax.tick_params(direction="out", pad=1.5, length=2.5, width=0.7)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#E6E8E9", linewidth=0.5)


def title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", pad=3, fontsize=9.0)


def configuration_axis(ax, threshold, metric, ylabel, ylim, multiplier=1.0):
    configurations = [(0.05, 0.80), (0.05, 0.90), (0.05, 1.00),
                      (0.10, 0.80), (0.10, 0.90), (0.10, 1.00),
                      (0.20, 0.80), (0.20, 0.90), (0.20, 1.00)]
    x = np.arange(len(configurations))
    for period in PERIOD_ORDER:
        subset = threshold.loc[threshold["period"].eq(period)].copy()
        values = []
        for forest, persistence in configurations:
            value = subset.loc[
                subset["forest_presence_threshold"].eq(forest)
                & subset["temporal_persistence_threshold"].eq(persistence), metric
            ].iloc[0]
            values.append(value * multiplier)
        ax.plot(x, values, marker="o", markersize=3.1, lw=1.0, color=PERIOD_COLORS[period], label=PERIOD_LABELS[period])
    for boundary in [2.5, 5.5]:
        ax.axvline(boundary, color="#BFC4C7", lw=0.55, ls="--", zorder=0)
    ax.axvline(4, color="#555B60", lw=0.7, ls=":", zorder=0)
    ax.set_xticks(x, [f"{int(forest * 100)}\n{int(persistence * 100)}" for forest, persistence in configurations])
    ax.set_xlabel("Forest threshold (%) / temporal persistence (%)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.text(4, ylim[0] + 0.04 * (ylim[1] - ylim[0]), "Primary", ha="center", va="bottom", fontsize=6.4, color="#40464B")
    style_axis(ax)


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    threshold = pd.read_csv(THRESHOLD)
    common = pd.read_csv(COMMON)
    threshold.to_csv(OUTDIR / "source_data_FigS4_threshold_definition.csv", index=False)
    common.to_csv(OUTDIR / "source_data_FigS4_common_sample.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 120 / 25.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    configuration_axis(ax_a, threshold, "spearman_rho_vs_reference", "Spearman $\\rho$ vs. primary", (0.975, 1.001))
    title(ax_a, "a", "Forest-definition sensitivity")
    ax_a.legend(loc="lower left", ncol=3, columnspacing=0.9, handletextpad=0.35)

    configuration_axis(ax_b, threshold, "slope_sign_agreement_vs_reference", "Slope-sign agreement (%)", (97.5, 100.1), multiplier=100)
    title(ax_b, "b", "Sign agreement under alternative definitions")

    common = common.loc[common["model"].eq("full")].copy()
    primary = pd.read_csv(PRIMARY_PERFORMANCE)
    common["period_label"] = common["period"].map(PERIOD_LABELS)
    primary_r2 = primary.loc[
        (primary["process"].eq("persistent_forest_agbd_change")) & primary["model"].eq("full"),
        ["period", "R2"],
    ].set_index("period")["R2"].to_dict()
    y = np.arange(len(common))
    for i, row in common.reset_index(drop=True).iterrows():
        main_value = primary_r2[row["period"]]
        common_value = row["R2"]
        ax_c.plot([main_value, common_value], [i, i], color="#9DA5AA", lw=1.1, zorder=1)
        ax_c.scatter(main_value, i, s=25, color="#6E7880", edgecolor="white", linewidth=0.45, zorder=3, label="Primary sample" if i == 0 else None)
        ax_c.scatter(common_value, i, s=31, color="#7DA6BE", edgecolor="white", linewidth=0.45, zorder=3, label="Common sample" if i == 0 else None)
        ax_c.text(max(main_value, common_value) + 0.018, i, f"{main_value:.3f} / {common_value:.3f}", va="center", fontsize=6.6)
    ax_c.set_yticks(y, common["period_label"])
    ax_c.set_xlim(0.20, 0.54)
    ax_c.set_xlabel("OOF $R^2$")
    ax_c.invert_yaxis()
    style_axis(ax_c, "x")
    title(ax_c, "c", "Primary versus common-sample performance")
    ax_c.text(
        0.03, 0.51, "Grey: primary sample\nBlue: common sample",
        transform=ax_c.transAxes, ha="left", va="center", fontsize=6.3,
        color="#4D555B",
    )

    x = np.arange(len(common))
    bars = ax_d.bar(x, common["ranking_spearman_rho_vs_primary"], width=0.45, color="#93B3A1", edgecolor="#44515A", linewidth=0.55)
    for i, (bar, row) in enumerate(zip(bars, common.itertuples())):
        ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.012, f"{bar.get_height():.3f}", ha="center", va="top", fontsize=6.7)
        ax_d.text(bar.get_x() + bar.get_width() / 2, 0.815, f"Top-5: {row.top5_overlap_count}/5", ha="center", va="bottom", fontsize=6.4)
    ax_d.set_xticks(x, common["period_label"])
    ax_d.set_ylim(0.80, 1.01)
    ax_d.set_ylabel("Rank Spearman $\\rho$ vs. primary")
    style_axis(ax_d)
    title(ax_d, "d", "Predictor-ranking concordance")

    stem = OUTDIR / "FigS4_persistent_forest_definition_and_common_sample_robustness"
    for suffix, kwargs in {".svg": {}, ".pdf": {}, ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}, ".png": {"dpi": 450}}.items():
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", pad_inches=0.01, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
