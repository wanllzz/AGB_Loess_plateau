from __future__ import annotations

"""
Fig. 3 | Spatial coupling and regional heterogeneity of forest-cover change
and persistent-forest AGBD change during 1990–2022.

Visual refinements:
1) Panel labels + short titles are placed outside the plotting frames.
2) Panels a–c and f use complete four-sided frames.
3) Panel d uses the same Loess Plateau map styling as Fig. 1:
   - Plateau outer boundary
   - ecological-zone boundaries
   - A1–D zone labels
   - 102/105/108/111/114°E and 34/36/38/40/42°N ticks
   - pale background for the complete Plateau, so cells without joint-process
     data are visually distinguished from areas outside the study region.
4) The joint-process legend for panel d is placed below the longitude ticks with reduced bottom whitespace.
5) Panels a and b use their own valid samples; panels c–f use common-valid
   cells, consistent with the manuscript definition.
6) Panel d maps joint process direction and panel e grades the intensity of
   the dominant increase–accumulation process as low, medium, or high.

All figures, source-data summaries, and a copy of this script are written to:
outputs/figures
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

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import spearmanr


# =========================================================
# 1. Paths and analysis settings
# =========================================================
RESULT_DIR = DATA_DIR
PROCESS_DIR = PROCESS_DATA_DIR
OUTPUT_DIR = FIGURE_OUTPUT_DIR
TABLE = PROCESS_DIR / "process_response_model_table_0p1deg.csv"
GRID = PROCESS_DIR / "process_responses_native_0p1deg_1990_2022.nc"
ZONE_PATH = RAW_DATA_DIR / "boundaries" / "huangtugaoyuan" / "Ecological_regionalization.shp"

PERIOD = "full_1990_2022"
FIG_WIDTH_IN = 180 / 25.4
FIG_HEIGHT_IN = 5.45

MAP_EXTENT = (100.7, 114.7, 33.4, 42.0)
MAP_BOX_ASPECT = (MAP_EXTENT[3] - MAP_EXTENT[2]) / (MAP_EXTENT[1] - MAP_EXTENT[0])
MAP_XTICKS = [102, 105, 108, 111, 114]
MAP_YTICKS = [34, 36, 38, 40, 42]

ZONE_ORDER = ["A1", "A2", "B1", "B2", "C", "D"]
ZONE_ID_TO_CODE = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 5: "C", 6: "D"}
ZONE_COLORS = {
    1: "#8EA6BF",
    2: "#B8C9D8",
    3: "#93B3A1",
    4: "#B8CDBB",
    5: "#D8C3A0",
    6: "#C9A79C",
}

# Muted low-to-high palette for the joint increase-intensity map.
JOINT_INTENSITY_COLORS = ["#DCE8ED", "#8FB7C9", "#3E7896"]

JOINT_COLORS = {
    1: "#8FB2C9",  # Increase + accumulation
    2: "#D8A18D",  # Increase + decline
    3: "#93B7A3",  # Decrease + accumulation
    4: "#C98E86",  # Decrease + decline
}
JOINT_LABELS = {
    1: "Increase + accumulation",
    2: "Increase + decline",
    3: "Decrease + accumulation",
    4: "Decrease + decline",
}


# =========================================================
# 2. Global style
# =========================================================
def apply_style() -> None:
    mpl.rcParams.update(
        {
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
        }
    )


# =========================================================
# 3. General helpers
# =========================================================
def load_zones() -> gpd.GeoDataFrame:
    zones = gpd.read_file(ZONE_PATH)
    if "CODE" not in zones.columns:
        raise KeyError(f"Field 'CODE' was not found in {ZONE_PATH}")

    bounds = zones.total_bounds
    if (
        -180 <= bounds[0] <= 180
        and -180 <= bounds[2] <= 180
        and -90 <= bounds[1] <= 90
        and -90 <= bounds[3] <= 90
        and (zones.crs is None or not zones.crs.is_geographic)
    ):
        zones = zones.set_crs("EPSG:4326", allow_override=True)

    if zones.crs is None:
        raise ValueError("The ecological-zone shapefile has no CRS information.")

    zones = zones[["CODE", "geometry"]].to_crs("EPSG:4326")
    zones["CODE"] = zones["CODE"].astype(str).str.strip()
    return zones


def as_bool(series: pd.Series) -> pd.Series:
    """Robust conversion of persistent_forest_eligible to Boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def add_panel_title(ax: plt.Axes, label: str, title: str) -> None:
    """Panel number + label outside the plot frame, aligned to the left edge."""
    ax.set_title(
        f"({label}) {title}",
        loc="left",
        x=0.0,
        y=1.02,
        pad=3.0,
        fontsize=9.0,
        fontweight="normal",
    )


def style_nonmap(ax: plt.Axes, grid_axis: str | None = None) -> None:
    """Full four-sided frame for panels a–c and e–f."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#444444")

    ax.tick_params(direction="out", pad=1.5, length=2.5, width=0.7)
    ax.set_axisbelow(True)

    if grid_axis == "x":
        ax.grid(axis="x", color="#E7E7E7", linewidth=0.50, zorder=0)
    elif grid_axis == "y":
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.50, zorder=0)
    elif grid_axis == "both":
        ax.grid(color="#ECECEC", linewidth=0.45, zorder=0)


def style_map(ax: plt.Axes, zones: gpd.GeoDataFrame) -> None:
    """Fig. 1-style Loess Plateau map frame, ticks, and boundaries."""
    # Pale fill makes the complete Loess Plateau visible even where joint data are absent.
    zones.plot(ax=ax, facecolor="#F2F2F2", edgecolor="none", zorder=0)

    zones.boundary.plot(
        ax=ax,
        color="#666666",
        linewidth=0.35,
        zorder=5,
    )

    outer = zones.geometry.union_all() if hasattr(zones.geometry, "union_all") else zones.unary_union
    gpd.GeoSeries([outer], crs=zones.crs).boundary.plot(
        ax=ax,
        color="#1A1A1A",
        linewidth=0.80,
        zorder=6,
    )

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_xticks(MAP_XTICKS)
    ax.set_yticks(MAP_YTICKS)
    ax.set_xticklabels([f"{x}°E" for x in MAP_XTICKS], fontsize=8.0)
    ax.set_yticklabels([f"{y}°N" for y in MAP_YTICKS], fontsize=8.0)
    ax.tick_params(direction="out", pad=1.5, length=2.5, width=0.7)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#444444")

    # Same geographic frame proportion used in Fig. 1.
    ax.set_box_aspect(MAP_BOX_ASPECT)
    ax.set_aspect("auto")
    ax.set_anchor("N")


def label_ecological_zones(ax: plt.Axes, zones: gpd.GeoDataFrame) -> None:
    """Label A1–D using internal representative points, matching Fig. 1 style."""
    zone_geom = zones[["CODE", "geometry"]].dissolve(by="CODE").reset_index()
    points = zone_geom.geometry.representative_point()

    for code, point in zip(zone_geom["CODE"], points):
        if str(code) not in ZONE_ORDER:
            continue
        x, y = point.x, point.y

        # Keep the D label away from its eastern boundary, following the Fig. 1 layout.
        if str(code) == "D":
            x += 0.25

        ax.text(
            x,
            y,
            str(code),
            ha="center",
            va="center",
            fontsize=9.0,
            color="#1A1A1A",
            zorder=25,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
        )


def save_bundle(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=800,
        bbox_inches="tight",
        pad_inches=0.01,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base.with_suffix(".jpg"), dpi=300, bbox_inches="tight", pad_inches=0.01)


def copy_running_script() -> None:
    """Keep a reproducible copy of the plotting script beside the output figures."""
    try:
        src = Path(__file__).resolve()
        dst = (OUTPUT_DIR / src.name).resolve()
        if src != dst:
            shutil.copy2(src, dst)
    except Exception as exc:
        print(f"Warning: could not copy the plotting script to output directory: {exc}")


# =========================================================
# 4. Data helpers
# =========================================================
def classify_joint(df: pd.DataFrame) -> pd.Series:
    forest = df["forest_fraction_slope"].to_numpy(float)
    agbd = df["persistent_forest_agbd_slope"].to_numpy(float)
    return pd.Series(
        np.select(
            [
                (forest >= 0) & (agbd >= 0),
                (forest >= 0) & (agbd < 0),
                (forest < 0) & (agbd >= 0),
                (forest < 0) & (agbd < 0),
            ],
            [1, 2, 3, 4],
            default=np.nan,
        ),
        index=df.index,
        dtype="float",
    )


def set_unique_two_decimal_yticks(ax: plt.Axes) -> None:
    """Keep panel-a y ticks readable at two decimals without duplicate labels."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ticks = ax.get_yticks()
    keep_ticks, keep_labels, seen = [], [], set()
    for tick in ticks:
        label = f"{tick:.2f}"
        if label == "-0.00":
            label = "0.00"
        if label in seen:
            continue
        seen.add(label)
        keep_ticks.append(tick)
        keep_labels.append(label)
    ax.set_yticks(keep_ticks)
    ax.set_yticklabels(keep_labels)


def boxplot_by_zone(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    ylabel: str,
    label: str,
    title: str,
) -> None:
    groups = [
        data.loc[data["ecological_zone_id"] == zone, column].dropna().to_numpy()
        for zone in range(1, 7)
    ]

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.56,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 0.95},
        whiskerprops={"color": "#333333", "linewidth": 0.80},
        capprops={"color": "#333333", "linewidth": 0.80},
        boxprops={"edgecolor": "#444444", "linewidth": 0.70},
    )

    for patch, zone in zip(bp["boxes"], range(1, 7)):
        patch.set_facecolor(ZONE_COLORS[zone])
        patch.set_alpha(0.85)

    ax.axhline(0, color="#666666", linewidth=0.65, zorder=1)
    ax.set_xticks(range(1, 7), [ZONE_ID_TO_CODE[z] for z in range(1, 7)])
    ax.set_xlabel("Ecological zone")
    ax.set_ylabel(ylabel)
    if label == "a":
        set_unique_two_decimal_yticks(ax)
    style_nonmap(ax, "y")
    add_panel_title(ax, label, title)


def draw_intensity_legend_below_map(fig: plt.Figure, ax_map: plt.Axes) -> plt.Axes:
    """Place a compact one-row legend below the intensity map, aligned with panel d."""
    fig.canvas.draw()
    pos = ax_map.get_position()
    legend_ax = fig.add_axes([pos.x0, pos.y0 - 0.070, pos.width, 0.032])
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(JOINT_INTENSITY_COLORS, ["Low", "Medium", "High"])
    ]
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        fontsize=7.2,
        handlelength=1.45,
        handleheight=0.9,
        handletextpad=0.45,
        columnspacing=0.90,
        labelspacing=0.25,
        borderaxespad=0.0,
        frameon=False,
    )
    return legend_ax


def draw_joint_legend_below_map(fig: plt.Figure, ax_map: plt.Axes) -> plt.Axes:
    """Place a compact four-row legend below the joint-direction map."""
    fig.canvas.draw()
    pos = ax_map.get_position()
    legend_ax = fig.add_axes([pos.x0, pos.y0 - 0.138, pos.width, 0.112])
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=JOINT_COLORS[i], edgecolor="none", label=JOINT_LABELS[i])
        for i in [1, 2, 3, 4]
    ]
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        ncol=1,
        fontsize=7.2,
        handlelength=1.45,
        handleheight=0.9,
        handletextpad=0.45,
        labelspacing=0.32,
        borderaxespad=0.0,
        frameon=False,
    )
    return legend_ax


# =========================================================
# 5. Main figure
# =========================================================
def main() -> None:
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zones = load_zones()

    # ---------- Read response table ----------
    data_all = pd.read_csv(TABLE)
    required = {
        "period",
        "ecological_zone_id",
        "forest_fraction_slope",
        "persistent_forest_agbd_slope",
        "persistent_forest_eligible",
    }
    missing = required.difference(data_all.columns)
    if missing:
        raise KeyError(f"Missing required columns in {TABLE}: {sorted(missing)}")

    data_all["ecological_zone_id"] = pd.to_numeric(data_all["ecological_zone_id"], errors="coerce")
    data_all["persistent_forest_eligible"] = as_bool(data_all["persistent_forest_eligible"])
    data = data_all[data_all["period"].astype(str) == PERIOD].copy()

    # Panel a: its own valid forest-cover-response sample.
    forest_data = data[np.isfinite(data["forest_fraction_slope"])].copy()

    # Panel b: its own persistent-forest-response sample.
    agbd_data = data[
        data["persistent_forest_eligible"]
        & np.isfinite(data["persistent_forest_agbd_slope"])
    ].copy()

    # Panels c–f: common-valid cells only.
    common = data[
        data["persistent_forest_eligible"]
        & np.isfinite(data["forest_fraction_slope"])
        & np.isfinite(data["persistent_forest_agbd_slope"])
        & data["ecological_zone_id"].isin(range(1, 7))
    ].copy()
    common["joint_type"] = classify_joint(common)

    # ---------- Figure layout ----------
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.165,
        top=0.955,
        wspace=0.34,
        hspace=0.42,
        height_ratios=[1.0, 1.0],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    # ---------- (a) Forest-fraction slope ----------
    boxplot_by_zone(
        ax_a,
        forest_data,
        "forest_fraction_slope",
        r"Forest-fraction slope (yr$^{-1}$)",
        "a",
        "Forest-cover change",
    )

    # ---------- (b) Persistent-forest AGBD slope ----------
    boxplot_by_zone(
        ax_b,
        agbd_data,
        "persistent_forest_agbd_slope",
        r"Persistent-forest AGBD slope (Mg ha$^{-1}$ yr$^{-1}$)",
        "b",
        "Persistent-forest AGBD change",
    )

    # ---------- (c) Grid-cell coupling and zonal centres ----------
    for zone in range(1, 7):
        subset = common[common["ecological_zone_id"] == zone]
        ax_c.scatter(
            subset["forest_fraction_slope"],
            subset["persistent_forest_agbd_slope"],
            s=9,
            alpha=0.34,
            color=ZONE_COLORS[zone],
            edgecolor="none",
            label=ZONE_ID_TO_CODE[zone],
            zorder=3,
        )

    rho_all, _ = spearmanr(
        common["forest_fraction_slope"],
        common["persistent_forest_agbd_slope"],
    )
    ax_c.axvline(0, color="#666666", linewidth=0.65, zorder=2)
    ax_c.axhline(0, color="#666666", linewidth=0.65, zorder=2)
    ax_c.set_xlabel(r"Forest-fraction slope (yr$^{-1}$)")
    ax_c.set_ylabel(r"Persistent-forest AGBD slope (Mg ha$^{-1}$ yr$^{-1}$)")
    ax_c.legend(
        ncol=3,
        loc="upper right",
        fontsize=6.3,
        markerscale=1.0,
        handletextpad=0.20,
        columnspacing=0.55,
        borderaxespad=0.30,
        frameon=False,
    )
    ax_c.text(
        0.55,
        0.05,
        rf"Spearman's $\rho$ = {rho_all:.3f}" + "\n" + rf"$n$ = {len(common):,}",
        transform=ax_c.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.30},
        zorder=8,
    )
    style_nonmap(ax_c, "both")
    add_panel_title(ax_c, "c", "Grid-cell coupling")

    # ---------- (d) Joint-process direction map ----------
    with xr.open_dataset(GRID) as ds:
        period_values = [str(v) for v in ds["period"].values]
        if PERIOD not in period_values:
            raise ValueError(f"{PERIOD} not found in NetCDF period coordinate: {period_values}")
        pidx = period_values.index(PERIOD)

        forest = ds["forest_fraction_slope"].isel(period=pidx).values.astype(float)
        agbd = ds["persistent_forest_agbd_slope"].isel(period=pidx).values.astype(float)
        eligible = ds["persistent_forest_1km_count"].isel(period=pidx).values >= 10
        lons = ds["lon"].values.astype(float)
        lats = ds["lat"].values.astype(float)

    lon2d, lat2d = np.meshgrid(lons, lats)
    valid_joint = eligible & np.isfinite(forest) & np.isfinite(agbd)
    joint = np.full(forest.shape, np.nan, dtype=float)
    joint[valid_joint & (forest >= 0) & (agbd >= 0)] = 1
    joint[valid_joint & (forest >= 0) & (agbd < 0)] = 2
    joint[valid_joint & (forest < 0) & (agbd >= 0)] = 3
    joint[valid_joint & (forest < 0) & (agbd < 0)] = 4

    joint_cmap = mpl.colors.ListedColormap([JOINT_COLORS[i] for i in [1, 2, 3, 4]])
    joint_norm = mpl.colors.BoundaryNorm(np.arange(0.5, 5.5, 1.0), joint_cmap.N)
    zones.plot(ax=ax_d, facecolor="#F2F2F2", edgecolor="none", zorder=0)
    ax_d.pcolormesh(lon2d, lat2d, joint, cmap=joint_cmap, norm=joint_norm,
                    shading="nearest", zorder=2)
    style_map(ax_d, zones)
    label_ecological_zones(ax_d, zones)
    add_panel_title(ax_d, "d", "Joint process types")
    ax_d_leg = draw_joint_legend_below_map(fig, ax_d)

    # ---------- (e) Joint increase-intensity map ----------
    # Within increase–accumulation cells, calculate each response's empirical
    # rank and use the lower of the two ranks as the joint score. Consequently,
    # a high class requires simultaneously high positive forest-cover change and AGBD
    # accumulation; a single high response cannot determine a high joint class.
    increase_accumulation_mask = valid_joint & (joint == 1)
    forest_values = forest[increase_accumulation_mask]
    agbd_values = agbd[increase_accumulation_mask]
    forest_rank = pd.Series(forest_values).rank(method="average", pct=True).to_numpy()
    agbd_rank = pd.Series(agbd_values).rank(method="average", pct=True).to_numpy()
    joint_score = np.minimum(forest_rank, agbd_rank)
    intensity_breaks = np.nanquantile(joint_score, [1 / 3, 2 / 3])
    intensity_values = np.digitize(joint_score, intensity_breaks, right=False) + 1
    intensity = np.full(forest.shape, np.nan, dtype=float)
    intensity[increase_accumulation_mask] = intensity_values.astype(float)
    intensity_cmap = mpl.colors.ListedColormap(JOINT_INTENSITY_COLORS)
    intensity_norm = mpl.colors.BoundaryNorm(np.arange(0.5, 4.5, 1.0), intensity_cmap.N)

    # Non-dominant joint types remain pale background; only the dominant process
    # is graded here, while all four directions are retained in panel d.
    zones.plot(ax=ax_e, facecolor="#F2F2F2", edgecolor="none", zorder=0)
    ax_e.pcolormesh(
        lon2d,
        lat2d,
        intensity,
        cmap=intensity_cmap,
        norm=intensity_norm,
        shading="nearest",
        zorder=2,
    )
    style_map(ax_e, zones)
    label_ecological_zones(ax_e, zones)
    add_panel_title(ax_e, "e", "Joint increase intensity")
    ax_e_leg = draw_intensity_legend_below_map(fig, ax_e)

    # ---------- (f) Within-zone coupling ----------
    rho_vals: list[float] = []
    n_vals: list[int] = []
    for zone in range(1, 7):
        subset = common[common["ecological_zone_id"] == zone]
        n_vals.append(len(subset))
        rho = spearmanr(
            subset["forest_fraction_slope"],
            subset["persistent_forest_agbd_slope"],
        ).statistic if len(subset) >= 3 else np.nan
        rho_vals.append(float(rho))

    x = np.arange(6)
    bars = ax_f.bar(
        x,
        rho_vals,
        width=0.62,
        color=[ZONE_COLORS[z] for z in range(1, 7)],
        edgecolor="#444444",
        linewidth=0.60,
        zorder=3,
    )
    ax_f.axhline(0, color="#666666", linewidth=0.65, zorder=2)
    ax_f.set_xticks(x, ZONE_ORDER)
    ax_f.set_xlabel("Ecological zone")
    ax_f.set_ylabel(r"Spearman's $\rho$")
    ax_f.set_ylim(0, 0.8)

    for rect, rho, n in zip(bars, rho_vals, n_vals):
        if not np.isfinite(rho):
            continue
        offset = 0.02
        ax_f.text(
            rect.get_x() + rect.get_width() / 2,
            min(rho + offset, 0.77),
            f"{rho:.2f}\n$n$={n}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )

    style_nonmap(ax_f, "y")
    add_panel_title(ax_f, "f", "Zonal coupling strength")

    # Shrink panel f slightly and lower it so the bottom x-axis title aligns
    # visually with the lowest edge of the panel d legend block.
    fig.canvas.draw()
    pos_f = ax_f.get_position()
    pos_d_leg = ax_d_leg.get_position()
    new_width = pos_f.width * 0.90
    new_height = pos_f.height * 0.82
    ax_f.set_position([
        pos_f.x0 + (pos_f.width - new_width) / 2,
        pos_d_leg.y0 + 0.072,
        new_width,
        new_height,
    ])

    # ---------- Save figure and source summaries ----------
    source_forest = forest_data.copy()
    source_agbd = agbd_data.copy()
    source_common = common.copy()
    source_forest.to_csv(OUTPUT_DIR / "Fig3_source_forest_cover_change_1990_2022.csv", index=False)
    source_agbd.to_csv(OUTPUT_DIR / "Fig3_source_persistent_forest_agbd_1990_2022.csv", index=False)
    source_common.to_csv(OUTPUT_DIR / "Fig3_source_common_valid_cells_1990_2022.csv", index=False)

    zonal_summary = pd.DataFrame(
        {
            "zone": ZONE_ORDER,
            "spearman_rho": rho_vals,
            "n_common_cells": n_vals,
        }
    )
    zonal_summary.to_csv(OUTPUT_DIR / "Fig3_source_zonal_coupling_strength.csv", index=False)

    save_bundle(fig, "Fig3_process_separated_response_structure_refined")
    copy_running_script()
    plt.close(fig)

    print(f"Figure and source data written to: {OUTPUT_DIR}")
    print(f"Common-valid cells: {len(common):,}")
    print(f"Overall Spearman rho: {rho_all:.3f}")


if __name__ == "__main__":
    main()
