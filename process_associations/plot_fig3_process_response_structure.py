"""Create Fig. 3: spatial coupling of two process-separated responses.

Panel-specific sampling is intentional: forest-cover change uses every
forest-change-eligible cell; persistent-forest AGBD change uses every
persistent-forest-eligible cell; only joint panels use their intersection.
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch
from scipy.stats import spearmanr


ROOT = PROCESS_DATA_DIR
FIG_DIR = ROOT / "figures_process_associations"
TABLE = ROOT / "process_response_model_table_0p1deg.csv"
GRID = ROOT / "process_responses_native_0p1deg_1990_2022.nc"
PERIOD = "full_1990_2022"
ZONE_NAMES = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C", 6: "D"}
ZONE_COLORS = {1: "#4677A9", 2: "#74A6D4", 3: "#4B9C78", 4: "#86BF91", 5: "#D8A75B", 6: "#A8756D"}
JOINT_COLORS = {1: "#4A90C2", 2: "#D7866A", 3: "#2F7D59", 4: "#B5534D"}

mpl.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8.2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def panel_label(ax, label):
    ax.text(0.015, 0.985, f"({label})", transform=ax.transAxes, va="top", ha="left",
            fontsize=10.5, fontweight="bold", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.2})


def boxplot_by_zone(ax, data, column, ylabel, label):
    groups = [data.loc[data.ecological_zone_id == zone, column].dropna().values for zone in range(1, 7)]
    box = ax.boxplot(groups, patch_artist=True, widths=0.62, showfliers=False,
                     medianprops={"color": "black", "linewidth": 1.0})
    for patch, zone in zip(box["boxes"], range(1, 7)):
        patch.set_facecolor(ZONE_COLORS[zone]); patch.set_alpha(0.82); patch.set_edgecolor("#333333")
    ax.axhline(0, color="#555555", linewidth=0.7, zorder=0)
    ax.set_xticks(range(1, 7), [ZONE_NAMES[z] for z in range(1, 7)])
    ax.set_ylabel(ylabel); ax.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    panel_label(ax, label)


def draw_joint_map(ax, lon, lat, forest, agbd, forest_eligible, agbd_eligible):
    joint = np.full(forest.shape, np.nan)
    usable = forest_eligible & agbd_eligible & np.isfinite(forest) & np.isfinite(agbd)
    joint[usable & (forest >= 0) & (agbd >= 0)] = 1
    joint[usable & (forest >= 0) & (agbd < 0)] = 2
    joint[usable & (forest < 0) & (agbd >= 0)] = 3
    joint[usable & (forest < 0) & (agbd < 0)] = 4
    cmap = mpl.colors.ListedColormap([JOINT_COLORS[i] for i in range(1, 5)])
    ax.pcolormesh(lon, lat, joint, cmap=cmap,
                  norm=mpl.colors.BoundaryNorm(np.arange(0.5, 5.5, 1), cmap.N), shading="nearest")
    ax.set_aspect("equal"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    handles = [Patch(facecolor=JOINT_COLORS[i], edgecolor="none", label=label) for i, label in [
        (1, "Expansion + accumulation"), (2, "Expansion + decline"),
        (3, "Contraction + accumulation"), (4, "Contraction + decline"),
    ]]
    ax.legend(handles=handles, fontsize=6.2, loc="lower left", frameon=True, facecolor="white",
              edgecolor="#BBBBBB", handlelength=1.1, borderpad=0.35)
    return joint


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_data = pd.read_csv(TABLE)
    all_data = all_data[all_data.period == PERIOD].copy()
    forest_data = all_data[all_data.forest_change_state_eligible].dropna(subset=["forest_fraction_slope"]).copy()
    agbd_data = all_data[all_data.persistent_forest_eligible].dropna(subset=["persistent_forest_agbd_slope"]).copy()
    joint_data = all_data[all_data.forest_change_state_eligible & all_data.persistent_forest_eligible].dropna(
        subset=["forest_fraction_slope", "persistent_forest_agbd_slope"]
    ).copy()

    rho, _ = spearmanr(joint_data.forest_fraction_slope, joint_data.persistent_forest_agbd_slope)
    zone_rho = []
    for zone in range(1, 7):
        subset = joint_data[joint_data.ecological_zone_id == zone]
        zone_rho.append((zone, len(subset), spearmanr(subset.forest_fraction_slope, subset.persistent_forest_agbd_slope).statistic))

    fig = plt.figure(figsize=(7.25, 7.05), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.08], height_ratios=[1, 1.07])
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[0, col]) for col in range(3)]
    ax_d, ax_e, ax_f = [fig.add_subplot(grid[1, col]) for col in range(3)]

    boxplot_by_zone(ax_a, forest_data, "forest_fraction_slope", "Forest-fraction slope (yr$^{-1}$)", "a")
    boxplot_by_zone(ax_b, agbd_data, "persistent_forest_agbd_slope",
                    "Persistent-forest AGBD slope (Mg ha$^{-1}$ yr$^{-1}$)", "b")
    for zone in range(1, 7):
        subset = joint_data[joint_data.ecological_zone_id == zone]
        ax_c.scatter(subset.forest_fraction_slope, subset.persistent_forest_agbd_slope,
                     s=8, alpha=0.52, color=ZONE_COLORS[zone], edgecolor="none", label=ZONE_NAMES[zone])
    ax_c.axvline(0, color="#555555", linewidth=0.7); ax_c.axhline(0, color="#555555", linewidth=0.7)
    ax_c.set_xlabel("Forest-fraction slope (yr$^{-1}$)")
    ax_c.set_ylabel("Persistent-forest AGBD slope (Mg ha$^{-1}$ yr$^{-1}$)")
    ax_c.legend(ncol=3, loc="upper right", fontsize=6.4, handletextpad=0.2, columnspacing=0.6)
    ax_c.text(0.04, 0.08, f"n = {len(joint_data):,}\nSpearman's $\\rho$ = {rho:.3f}\n$p$ < 0.001",
              transform=ax_c.transAxes, fontsize=7.1, va="bottom",
              bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.9, "pad": 0.35})
    ax_c.grid(color="#EBEBEB", linewidth=0.55); panel_label(ax_c, "c")

    master = xr.open_dataset(GRID)
    period_index = list(master.period.values).index(PERIOD)
    forest = master.forest_fraction_slope.isel(period=period_index).values
    agbd = master.persistent_forest_agbd_slope.isel(period=period_index).values
    baseline = master.forest_fraction_baseline.isel(period=period_index).values
    agbd_count = master.persistent_forest_1km_count.isel(period=period_index).values
    lon, lat = np.meshgrid(master.lon.values, master.lat.values)
    forest_eligible = np.isfinite(baseline) & (baseline < 0.90)
    agbd_eligible = agbd_count >= 10
    joint = draw_joint_map(ax_d, lon, lat, forest, agbd, forest_eligible, agbd_eligible)
    panel_label(ax_d, "d")

    zones, counts, rhos = zip(*zone_rho)
    bars = ax_e.bar(range(1, 7), rhos, color=[ZONE_COLORS[z] for z in zones], edgecolor="#333333", linewidth=0.45)
    ax_e.axhline(0, color="#555555", linewidth=0.7); ax_e.set_ylim(-0.05, 0.72)
    ax_e.set_xticks(range(1, 7), [ZONE_NAMES[z] for z in zones])
    ax_e.set_ylabel("Within-zone Spearman's $\\rho$"); ax_e.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    for bar, count, value in zip(bars, counts, rhos):
        ax_e.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}\nn={count}", ha="center", va="bottom", fontsize=6.4)
    panel_label(ax_e, "e")

    joint_names = ["Expansion +\naccumulation", "Expansion +\ndecline", "Contraction +\naccumulation", "Contraction +\ndecline"]
    joint_data["joint_type"] = np.select(
        [
            (joint_data.forest_fraction_slope >= 0) & (joint_data.persistent_forest_agbd_slope >= 0),
            (joint_data.forest_fraction_slope >= 0) & (joint_data.persistent_forest_agbd_slope < 0),
            (joint_data.forest_fraction_slope < 0) & (joint_data.persistent_forest_agbd_slope >= 0),
        ],
        joint_names[:3], default=joint_names[3],
    )
    composition = (joint_data.groupby(["ecological_zone_id", "joint_type"]).size()
                   .unstack(fill_value=0).reindex(index=range(1, 7), columns=joint_names, fill_value=0))
    composition = composition.div(composition.sum(axis=1), axis=0) * 100
    bottom = np.zeros(6)
    for class_index, class_name in enumerate(joint_names):
        values = composition[class_name].to_numpy()
        ax_f.bar(range(1, 7), values, bottom=bottom, color=JOINT_COLORS[class_index + 1],
                 edgecolor="white", linewidth=0.35, label=class_name.replace("\n", " "))
        for x_value, y_value, base in zip(range(1, 7), values, bottom):
            if y_value >= 4:
                ax_f.text(x_value, base + y_value / 2, f"{y_value:.0f}", ha="center", va="center", fontsize=6.4,
                          color="white" if class_index in (0, 3) else "#1F2933")
        bottom += values
    ax_f.set_xticks(range(1, 7), [ZONE_NAMES[z] for z in range(1, 7)])
    ax_f.set_ylim(0, 100); ax_f.set_ylabel("Joint-process composition (%)")
    ax_f.grid(axis="y", color="#E7E7E7", linewidth=0.6)
    ax_f.legend(fontsize=5.8, loc="lower center", ncol=2, handlelength=1.0, columnspacing=0.65)
    panel_label(ax_f, "f")
    master.close()

    joint_data.to_csv(FIG_DIR / "Fig3_source_data_1990_2022_joint_cells.csv", index=False)
    pd.DataFrame(zone_rho, columns=["ecological_zone_id", "n_joint_cells", "spearman_rho"]).to_csv(
        FIG_DIR / "Fig3_source_zone_coupling.csv", index=False
    )
    composition.reset_index().to_csv(FIG_DIR / "Fig3_source_joint_type_composition_by_zone.csv", index=False)
    base = FIG_DIR / "Fig3_process_separated_response_structure"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
