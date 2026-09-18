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

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = PROCESS_DATA_DIR
FIG_OUT_DIR = FIGURE_OUTPUT_DIR
DATA_OUT_DIR = ROOT / "figures_process_associations"
PERFORMANCE = ROOT / "nested_spatial_xgboost_performance.csv"
SHAP = ROOT / "crossvalidated_treeshap_importance.csv"
OOF = ROOT / "nested_spatial_xgboost_oof_predictions.csv"
ZONE_PATH = RAW_DATA_DIR / "boundaries" / "huangtugaoyuan" / "Ecological_regionalization.shp"
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
FIG_WIDTH_IN = 180 / 25.4
FIG_HEIGHT_IN = 5.55
MAP_EXTENT = (100.7, 114.7, 33.4, 42.0)
MAP_BOX_ASPECT = (MAP_EXTENT[3] - MAP_EXTENT[2]) / (MAP_EXTENT[1] - MAP_EXTENT[0])
MAP_XTICKS = [102, 105, 108, 111, 114]
MAP_YTICKS = [34, 36, 38, 40, 42]
ZONE_ORDER = ["A1", "A2", "B1", "B2", "C", "D"]
BE_SHIFT_RIGHT = 0.7 / 18.0  # 0.7 cm on a 180-mm-wide figure (figure-coordinate fraction)
ROW_GAP = 0.10  # vertical gap between the top-row frames and bottom-row frames

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


def load_zones() -> gpd.GeoDataFrame:
    zones = gpd.read_file(ZONE_PATH)
    bounds = zones.total_bounds
    if (-180 <= bounds[0] <= 180 and -180 <= bounds[2] <= 180 and -90 <= bounds[1] <= 90 and -90 <= bounds[3] <= 90 and (zones.crs is None or not zones.crs.is_geographic)):
        zones = zones.set_crs("EPSG:4326", allow_override=True)
    if zones.crs is None:
        raise ValueError("The ecological-zone shapefile has no CRS information.")
    zones = zones[["CODE", "geometry"]].to_crs("EPSG:4326")
    zones["CODE"] = zones["CODE"].astype(str).str.strip()
    return zones


def add_panel_title(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", x=0.0, y=1.02, pad=3.0, fontsize=9.0, fontweight="normal")


def style_nonmap(ax, grid_axis=None, keep_box_aspect=True):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#444444")
    ax.tick_params(direction="out", pad=1.5, length=2.5, width=0.7)
    ax.set_axisbelow(True)
    if keep_box_aspect:
        ax.set_box_aspect(MAP_BOX_ASPECT)
        ax.set_anchor("N")
    else:
        ax.set_box_aspect(None)
    if grid_axis == "x":
        ax.grid(axis="x", color="#E7E7E7", linewidth=0.50, zorder=0)
    elif grid_axis == "y":
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.50, zorder=0)
    elif grid_axis == "both":
        ax.grid(color="#ECECEC", linewidth=0.45, zorder=0)


def style_map(ax, zones):
    zones.plot(ax=ax, facecolor="#F2F2F2", edgecolor="none", zorder=0)
    zones.boundary.plot(ax=ax, color="#666666", linewidth=0.35, zorder=5)
    outer = zones.geometry.union_all() if hasattr(zones.geometry, "union_all") else zones.unary_union
    gpd.GeoSeries([outer], crs=zones.crs).boundary.plot(ax=ax, color="#1A1A1A", linewidth=0.80, zorder=6)
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1]); ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_xticks(MAP_XTICKS); ax.set_yticks(MAP_YTICKS)
    ax.set_xticklabels([f"{x}°E" for x in MAP_XTICKS], fontsize=8.0)
    ax.set_yticklabels([f"{y}°N" for y in MAP_YTICKS], fontsize=8.0)
    ax.tick_params(direction="out", pad=1.5, length=2.5, width=0.7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#444444")
    ax.set_box_aspect(MAP_BOX_ASPECT)
    ax.set_aspect("auto")
    ax.set_anchor("N")


def label_ecological_zones(ax, zones):
    zone_geom = zones[["CODE", "geometry"]].dissolve(by="CODE").reset_index()
    points = zone_geom.geometry.representative_point()
    for code, point in zip(zone_geom["CODE"], points):
        if str(code) not in ZONE_ORDER:
            continue
        x, y = point.x, point.y
        if str(code) == "D":
            x += 0.25
        ax.text(x, y, str(code), ha="center", va="center", fontsize=9.0, color="#1A1A1A", zorder=25,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])


def copy_running_script():
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        src = Path(__file__).resolve(); dst = (DATA_OUT_DIR / src.name).resolve()
        if src != dst:
            shutil.copy2(src, dst)
    except Exception as exc:
        print(f"Warning: could not copy the plotting script to output directory: {exc}")


def save_bundle(fig, stem):
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_OUT_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base.with_suffix(".tiff"), dpi=800, bbox_inches="tight", pad_inches=0.01, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base.with_suffix(".jpg"), dpi=300, bbox_inches="tight", pad_inches=0.01)


def plot_performance(ax, performance, process, label):
    row = performance[(performance.process == process) & (performance.period == PERIOD)].set_index("model")
    values = row.loc[["state_only", "full"], "R2"].astype(float)
    bars = ax.bar(
        [0, 1], values, width=0.28,
        color=["#C9D3DC", "#5D88A8"], edgecolor="#44515A", linewidth=0.55, zorder=3
    )
    for i, (bar, value) in enumerate(zip(bars, values)):
        # In panel a, place the Full-model value just inside the bar top so it
        # does not overlap the Moran's I annotation above.
        if process == "forest_fraction_change" and i == 1:
            y = value - 0.012
            va = "top"
        else:
            y = value + 0.028 if value >= 0 else value - 0.030
            va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2, y, f"{value:.3f}",
            ha="center", va=va, fontsize=6.8
        )
    response = float(row.loc["full", "response_moran_I"])
    residual = float(row.loc["full", "residual_moran_I"])
    ax.axhline(0, color="#666666", linewidth=0.65, zorder=1)
    ax.set_xticks([0, 1], ["State-only", "Full"])
    if process == "forest_fraction_change":
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylim(-0.1, 0.5)
    ax.set_ylabel("Spatial OOF R$^2$")
    style_nonmap(ax, "y", keep_box_aspect=False)
    add_panel_title(ax, label, PROCESS_LABELS[process])
    ax.text(
        0.02, 0.95, f"Moran's I: {response:.3f} → {residual:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
        fontfamily="Times New Roman", fontstyle="italic"
    )


def plot_importance(ax, shap, process, label):
    data = shap[(shap.process == process) & (shap.period == PERIOD)].copy().sort_values("relative_importance_percent", ascending=True).tail(10)
    labels = [DISPLAY.get(item, item) for item in data.predictor]
    bars = ax.barh(
        labels, data.mean_abs_treeshap, xerr=data.sd_across_outer_folds, capsize=1.6,
        color="#8FB2C9", edgecolor="#44515A", linewidth=0.45, zorder=3,
        error_kw={"elinewidth": 0.6, "capthick": 0.6, "ecolor": "#44515A"}
    )
    xmax = float((data.mean_abs_treeshap + data.sd_across_outer_folds).max() * 1.42)
    ax.set_xlim(0, xmax)
    for bar, mean_v, sd_v, percentage in zip(bars, data.mean_abs_treeshap, data.sd_across_outer_folds, data.relative_importance_percent):
        x_text = float(mean_v + sd_v) + xmax * 0.015
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{percentage:.1f}%", va="center", ha="left", fontsize=6.1)
    ax.set_xlabel("Mean absolute TreeSHAP value")
    ax.tick_params(axis="y", labelsize=6.6)
    style_nonmap(ax, "x", keep_box_aspect=False)
    add_panel_title(ax, label, PROCESS_LABELS[process])
    return data


def plot_residual_map(ax, fig, oof, process, label, zones):
    data = oof[(oof.process == process) & (oof.period == PERIOD) & (oof.model == "full")].copy()
    limit = float(np.nanquantile(np.abs(data.oof_residual), 0.98))
    scatter = ax.scatter(data.lon, data.lat, c=data.oof_residual, s=10.5, marker="s", linewidths=0, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), rasterized=True, zorder=3)
    style_map(ax, zones)
    label_ecological_zones(ax, zones)
    add_panel_title(ax, label, PROCESS_LABELS[process])
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.008, pos.y0 + pos.height * 0.06, 0.012, pos.height * 0.86])
    cbar = fig.colorbar(scatter, cax=cax, orientation="vertical")
    unit = r"Residual (yr$^{-1}$)" if process == "forest_fraction_change" else r"Residual (Mg ha$^{-1}$ yr$^{-1}$)"
    cbar.set_label(unit, fontsize=8.0, labelpad=3)
    cbar.ax.tick_params(length=2.2, width=0.7, pad=1.5, labelsize=7.0)
    return data, cax


def add_column_headers(fig, ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
    fig.canvas.draw()
    pos_a, pos_b, pos_c = ax_a.get_position(), ax_b.get_position(), ax_c.get_position()
    x1 = (pos_a.x0 + pos_a.x1) / 2
    x2 = (pos_b.x0 + pos_b.x1) / 2
    x3 = (pos_c.x0 + pos_c.x1) / 2
    y_top = max(pos_a.y1, pos_b.y1, pos_c.y1) + 0.040
    y_mid = (ax_d.get_position().y1 + pos_a.y0) / 2
    fig.text(x1, y_top, "Model performance", ha="center", va="bottom", fontsize=9.0)
    fig.text(x2, y_top, "Relative TreeSHAP importance", ha="center", va="bottom", fontsize=9.0)
    fig.text(x3, y_top, "OOF residuals", ha="center", va="bottom", fontsize=9.0)


def main():
    FIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    zones = load_zones()
    performance = pd.read_csv(PERFORMANCE)
    shap = pd.read_csv(SHAP)
    oof = pd.read_csv(OOF)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    grid = fig.add_gridspec(
        2, 3,
        left=0.072, right=0.915, bottom=0.075, top=0.90,
        wspace=0.42, hspace=0.10,
        width_ratios=[0.68, 1.12, 1.18],
        height_ratios=[1.0, 1.0],
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]

    plot_performance(axes[0], performance, "forest_fraction_change", "a")
    importance_forest = plot_importance(axes[1], shap, "forest_fraction_change", "b")
    residual_forest, cax_c = plot_residual_map(axes[2], fig, oof, "forest_fraction_change", "c", zones)
    plot_performance(axes[3], performance, "persistent_forest_agbd_change", "d")
    importance_agbd = plot_importance(axes[4], shap, "persistent_forest_agbd_change", "e")
    residual_agbd, cax_f = plot_residual_map(axes[5], fig, oof, "persistent_forest_agbd_change", "f", zones)

    # Shift only panels b and e to the right by about 0.7 cm; keep a/d left-aligned and c/f unchanged.
    fig.canvas.draw()
    pos_b = axes[1].get_position()
    axes[1].set_position([pos_b.x0 + BE_SHIFT_RIGHT, pos_b.y0, pos_b.width, pos_b.height])
    pos_e = axes[4].get_position()
    axes[4].set_position([pos_e.x0 + BE_SHIFT_RIGHT, pos_e.y0, pos_e.width, pos_e.height])

    # Exact row-wise frame alignment:
    # a and b use exactly the same bottom/top coordinates as c;
    # d and e use exactly the same bottom/top coordinates as f.
    # Only x position and panel width are retained from each non-map panel.
    fig.canvas.draw()
    pos_c = axes[2].get_position()
    pos_f = axes[5].get_position()
    for idx in (0, 1):
        pos = axes[idx].get_position()
        axes[idx].set_position([pos.x0, pos_c.y0, pos.width, pos_c.height])
    for idx in (3, 4):
        pos = axes[idx].get_position()
        axes[idx].set_position([pos.x0, pos_f.y0, pos.width, pos_f.height])

    # Remove the large blank band between rows by translating the entire
    # bottom row upward. The top row (a-c) and panel c/f horizontal positions
    # remain unchanged. Move the f colorbar by the same amount.
    fig.canvas.draw()
    pos_c = axes[2].get_position()
    pos_f = axes[5].get_position()
    target_f_top = pos_c.y0 - ROW_GAP
    row_shift_y = target_f_top - pos_f.y1
    for idx in (3, 4, 5):
        pos = axes[idx].get_position()
        axes[idx].set_position([pos.x0, pos.y0 + row_shift_y, pos.width, pos.height])
    pos_cax_f = cax_f.get_position()
    cax_f.set_position([
        pos_cax_f.x0,
        pos_cax_f.y0 + row_shift_y,
        pos_cax_f.width,
        pos_cax_f.height,
    ])

    add_column_headers(fig, axes[0], axes[1], axes[2], axes[3], axes[4], axes[5])

    performance[performance.period == PERIOD].to_csv(DATA_OUT_DIR / "Fig4_source_full_period_performance.csv", index=False)
    pd.concat([importance_forest, importance_agbd], ignore_index=True).to_csv(DATA_OUT_DIR / "Fig4_source_full_period_treeshap.csv", index=False)
    pd.concat([residual_forest, residual_agbd], ignore_index=True).to_csv(DATA_OUT_DIR / "Fig4_source_oof_residuals_full_period.csv", index=False)
    save_bundle(fig, "Fig4_full_period_predictive_associations_refined")
    copy_running_script()
    plt.close(fig)
    print(f"Figure files written to: {FIG_OUT_DIR}")
    print(f"Source data and script written to: {DATA_OUT_DIR}")


if __name__ == "__main__":
    main()
