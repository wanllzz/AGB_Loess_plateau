from __future__ import annotations

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

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.stats import theilslopes

# =========================================================
# 1. Paths
# =========================================================
RESULT_DIR = DATA_DIR
SPATIAL_DIR = FIGURE_SOURCE_DIR / "spatial_pattern"
TEMPORAL_DIR = TEMPORAL_DATA_DIR
FIGURE_DIR = FIGURE_OUTPUT_DIR

ZONE_PATH = RAW_DATA_DIR / "boundaries" / "huangtugaoyuan" / "Ecological_regionalization.shp"

DECOMP_PATH = TEMPORAL_DATA_DIR / "agb_stock_change_decomposition.csv"
BOOTSTRAP_PATH = TEMPORAL_DATA_DIR / "agb_stock_change_decomposition_bootstrap.csv"

# =========================================================
# 2. Global figure settings
# =========================================================
FIG_WIDTH_IN = 180 / 25.4

MAP_EXTENT = (100.7, 114.7, 33.4, 42.0)
MAP_BOX_ASPECT = (
    (MAP_EXTENT[3] - MAP_EXTENT[2])
    / (MAP_EXTENT[1] - MAP_EXTENT[0])
)

ZONE_ORDER = ["A1", "A2", "B1", "B2", "C", "D"]
PERIODS = ["1990-1999", "2000-2023", "1990-2023"]

COLORS = {
    "blue": "#2F6690",
    "teal": "#3A8D8B",
    "green": "#4C956C",
    "green_light": "#A6C48A",
    "gold": "#D6A84B",
    "red": "#C65D57",
    "red_dark": "#9E3D3D",
    "purple": "#7563A8",
    "gray": "#8A8A8A",
    "gray_light": "#D9D9D9",
    "black": "#252525",
    "area_contribution": "#93AA8E",
    "density_contribution": "#C5A46D",
    "total": "#333333",
    "grid": "#E4E4E4",
}

TREND_COLORS = [
    COLORS["red_dark"],
    "#E6B0AA",
    "#EFEFEF",
    "#A9D8C3",
    "#2E6E45",
]
TREND_LABELS = [
    "Sig. decrease",
    "Decrease",
    "Stable",
    "Increase",
    "Sig. increase",
]
TREND_CMAP = ListedColormap(TREND_COLORS)
TREND_NORM = BoundaryNorm(
    [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
    TREND_CMAP.N,
)

FOREST_CHANGE_COLORS = [
    "#2E6E45",
    "#8BCF8B",
    "#C65D57",
    "#D6D6D6",
]
FOREST_CHANGE_LABELS = [
    "Persistent",
    "Expansion",
    "Contraction",
    "Sparse/other",
]

# Fig. 4 decomposition CI remains off in the merged Fig. 2.
SHOW_COMPONENT_CI = False

# =========================================================
# 3. Typography
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
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    sns.set_context("paper", rc={"font.family": "Times New Roman"})

# =========================================================
# 4. Shared data / map helpers
# =========================================================
def canonical_period(value) -> str:
    return str(value).replace("–", "-").replace("—", "-").strip()

def load_zones() -> gpd.GeoDataFrame:
    zones = gpd.read_file(ZONE_PATH)
    bounds = zones.total_bounds

    if (
        -180 <= bounds[0] <= 180
        and -180 <= bounds[2] <= 180
        and -90 <= bounds[1] <= 90
        and -90 <= bounds[3] <= 90
        and (zones.crs is None or not zones.crs.is_geographic)
    ):
        zones = zones.set_crs("EPSG:4326", allow_override=True)

    keep = ["CODE", "geometry"]
    return zones[keep].to_crs("EPSG:4326")

def raster_lonlat(
    path: Path,
    categorical: bool = False,
    resolution: float = 0.015,
):
    with rasterio.open(path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            "EPSG:4326",
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution,
        )

        destination = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        source = (
            src.read(1, masked=True)
            .astype(np.float32)
            .filled(np.nan)
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=(
                Resampling.nearest
                if categorical
                else Resampling.bilinear
            ),
        )

    left = transform.c
    top = transform.f
    right = left + width * transform.a
    bottom = top + height * transform.e

    return destination, (left, right, bottom, top)

def style_map(
    ax: plt.Axes,
    zones: gpd.GeoDataFrame,
    show_y: bool = True,
    show_x: bool = True,
    tick_labelsize: float = 6.0,
) -> None:
    zones.boundary.plot(
        ax=ax,
        color="#555555",
        linewidth=0.35,
        zorder=5,
    )

    gpd.GeoSeries(
        [zones.unary_union],
        crs=zones.crs,
    ).boundary.plot(
        ax=ax,
        color="#1A1A1A",
        linewidth=0.75,
        zorder=6,
    )

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])

    xticks = [102, 105, 108, 111, 114]
    yticks = [34, 36, 38, 40, 42]

    ax.set_xticks(xticks if show_x else [])
    ax.set_yticks(yticks if show_y else [])

    if show_x:
        ax.set_xticklabels(
            [f"{x}°E" for x in xticks],
            fontsize=tick_labelsize,
        )

    if show_y:
        ax.set_yticklabels(
            [f"{y}°N" for y in yticks],
            fontsize=tick_labelsize,
        )

    ax.tick_params(
        direction="out",
        pad=1.5,
        length=2.5,
        width=0.7,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#444444")

    ax.set_box_aspect(MAP_BOX_ASPECT)
    ax.set_aspect("auto")

def label_ecological_zones(
    ax: plt.Axes,
    zones: gpd.GeoDataFrame,
) -> None:
    zone_geom = (
        zones[["CODE", "geometry"]]
        .dissolve(by="CODE")
        .reset_index()
    )

    # Representative points are guaranteed to lie inside polygons.
    points = zone_geom.geometry.representative_point()

    for code, point in zip(zone_geom["CODE"], points):
        x = point.x
        y = point.y

        # Fine-tuned placement for D: keep it inside the zone and away from the boundary.
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
            path_effects=[
                pe.withStroke(
                    linewidth=1.8,
                    foreground="white",
                )
            ],
        )

def style_nonmap(ax: plt.Axes, full_box: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#444444")

    if not full_box:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax.tick_params(direction="out", pad=1.5, length=2.5, width=0.7, labelsize=7)

def save_bundle(
    fig: plt.Figure,
    stem: str,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_DIR / f"{stem}.svg",
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_DIR / f"{stem}.tiff",
        dpi=800,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )

    fig.savefig(
        FIGURE_DIR / f"{stem}.jpg",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

# =========================================================
# 5. Trend and temporal helpers
# =========================================================
def stage_fit(
    years: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, float, float]:
    mask = (
        (years >= start)
        & (years <= end)
        & np.isfinite(values)
    )

    xx = years[mask].astype(float)
    yy = values[mask].astype(float)

    slope, intercept, _, _ = theilslopes(
        yy,
        xx,
    )

    yhat = intercept + slope * xx
    ss_res = np.sum((yy - yhat) ** 2)
    ss_tot = np.sum((yy - np.mean(yy)) ** 2)

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return slope, intercept, r2

def add_stage_fits_and_stats(
    ax: plt.Axes,
    years: np.ndarray,
    values: np.ndarray,
) -> None:
    specs = [
        (1990, 1999, COLORS["red"]),
        (2000, 2023, COLORS["teal"]),
    ]

    stats = []

    for start, end, color in specs:
        slope, intercept, r2 = stage_fit(
            years,
            values,
            start,
            end,
        )

        xx = np.array([start, end])

        ax.plot(
            xx,
            intercept + slope * xx,
            color=color,
            linewidth=1.45,
            zorder=4,
        )

        stats.append(
            (start, end, slope, r2, color)
        )

    line1 = (
        f"1990–1999: slope={stats[0][2]:.2f}, "
        f"R²={stats[0][3]:.2f}"
    )

    line2 = (
        f"2000–2023: slope={stats[1][2]:.2f}, "
        f"R²={stats[1][3]:.2f}"
    )

    ax.text(
        0.03,
        0.965,
        line1 + "\n" + line2,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#333333",
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
    )

def add_period_background(ax: plt.Axes) -> None:
    ax.axvspan(
        1990,
        1999.5,
        color="#E6DDD6",
        alpha=0.70,
        zorder=0,
    )

    ax.axvspan(
        1999.5,
        2023.8,
        color="#D5E1D8",
        alpha=0.70,
        zorder=0,
    )

    ax.axvline(
        1999.5,
        color="white",
        linewidth=1.0,
        zorder=1,
    )

# =========================================================
# 6. Symmetric AGB-stock decomposition helpers
# =========================================================
def load_decomposition() -> tuple[pd.DataFrame, pd.DataFrame]:
    decomp = pd.read_csv(DECOMP_PATH)
    decomp["_period"] = decomp["period"].map(
        canonical_period
    )

    if BOOTSTRAP_PATH.exists():
        bootstrap = pd.read_csv(BOOTSTRAP_PATH)
        bootstrap["_period"] = bootstrap["period"].map(
            canonical_period
        )
    else:
        bootstrap = pd.DataFrame()

    return decomp, bootstrap

def ci_lookup(
    bootstrap: pd.DataFrame,
    zone: str,
    period: str,
    component: str,
) -> tuple[float, float]:
    if bootstrap.empty:
        return np.nan, np.nan

    local = bootstrap[
        (bootstrap["zone"] == zone)
        & (bootstrap["_period"] == canonical_period(period))
        & (bootstrap["component"] == component)
    ]

    if local.empty:
        return np.nan, np.nan

    record = local.iloc[0]
    estimate = float(record["estimate_Pg"])

    return (
        estimate - float(record["ci_lower_Pg"]),
        float(record["ci_upper_Pg"]) - estimate,
    )

def decomposition_bottoms(
    area_values: np.ndarray,
    density_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(area_values)

    area_bottom = np.zeros(n)
    density_bottom = np.zeros(n)

    pos = np.zeros(n)
    neg = np.zeros(n)

    for i, value in enumerate(area_values):
        if value >= 0:
            area_bottom[i] = pos[i]
            pos[i] += value
        else:
            area_bottom[i] = neg[i]
            neg[i] += value

    for i, value in enumerate(density_values):
        if value >= 0:
            density_bottom[i] = pos[i]
            pos[i] += value
        else:
            density_bottom[i] = neg[i]
            neg[i] += value

    return area_bottom, density_bottom

def draw_decomposition(
    ax: plt.Axes,
    rows: pd.DataFrame,
    groups: list[str],
    mode: str,
    period: str | None = None,
    title: str = "",
    show_ylabel: bool = True,
    annotate_total: bool = False,
    bar_width: float = 0.58,
    area_color: str | None = None,
    density_color: str | None = None,
    total_color: str | None = None,
    title_fs: float = 9.0,
    axis_label_fs: float = 8.0,
    tick_fs: float = 7.0,
    annotation_fs: float = 6.2,
    xmargin: float = 0.10,
    full_box: bool = False,
) -> None:
    area = rows["area_contribution_Pg"].to_numpy(dtype=float)
    area_color = COLORS["area_contribution"] if area_color is None else area_color
    density_color = COLORS["density_contribution"] if density_color is None else density_color
    total_color = COLORS["total"] if total_color is None else total_color

    density = rows["density_contribution_Pg"].to_numpy(dtype=float)
    total = rows["stock_change_Pg"].to_numpy(dtype=float)

    x = np.arange(len(groups), dtype=float)

    area_bottom, density_bottom = decomposition_bottoms(
        area,
        density,
    )

    ax.bar(
        x,
        area,
        bottom=area_bottom,
        width=bar_width,
        color=area_color,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )

    ax.bar(
        x,
        density,
        bottom=density_bottom,
        width=bar_width,
        color=density_color,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )

    ax.scatter(
        x,
        total,
        marker="D",
        s=22,
        facecolor=total_color,
        edgecolor="white",
        linewidth=0.45,
        zorder=7,
    )

    if SHOW_COMPONENT_CI:
        for i, group in enumerate(groups):
            if mode == "period":
                zone_i = "Loess Plateau"
                period_i = group
            else:
                zone_i = group
                period_i = period

            for component, values, bottoms in [
                (
                    "area_contribution",
                    area,
                    area_bottom,
                ),
                (
                    "density_contribution",
                    density,
                    density_bottom,
                ),
            ]:
                lo, hi = ci_lookup(
                    bootstrap=GLOBAL_BOOTSTRAP,
                    zone=zone_i,
                    period=period_i,
                    component=component,
                )

                if not np.isfinite(lo) or not np.isfinite(hi):
                    continue

                endpoint = (
                    bottoms[i] + values[i]
                )

                ax.errorbar(
                    x[i],
                    endpoint,
                    yerr=np.array([[lo], [hi]]),
                    fmt="none",
                    ecolor="#4A4A4A",
                    elinewidth=0.65,
                    capsize=1.8,
                    capthick=0.65,
                    zorder=6,
                )

    if annotate_total:
        span = np.nanmax(total) - np.nanmin(total)
        if not np.isfinite(span) or span == 0:
            span = max(
                np.nanmax(np.abs(total)),
                0.1,
            )

        offset = 0.05 * span

        for xi, yi in zip(x, total):
            ax.text(
                xi,
                yi + (
                    offset
                    if yi >= 0
                    else -offset
                ),
                f"{yi:.2f}",
                ha="center",
                va=(
                    "bottom"
                    if yi >= 0
                    else "top"
                ),
                fontsize=annotation_fs,
                color=total_color,
            )

    ax.axhline(
        0,
        color="#555555",
        linewidth=0.75,
        zorder=2,
    )

    ax.grid(
        axis="y",
        color=COLORS["grid"],
        linewidth=0.5,
        zorder=0,
    )

    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)

    ax.set_title(title, loc="left", pad=2.5, fontsize=title_fs, fontweight="normal")
    if show_ylabel:
        ax.set_ylabel("Contribution to ΔAGB stock (Pg)", fontsize=axis_label_fs)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.margins(x=xmargin)
    style_nonmap(ax, full_box=full_box)
    ax.tick_params(axis="both", labelsize=tick_fs)

    closure = total - (area + density)
    max_diff = np.nanmax(np.abs(closure))

    if max_diff > 1e-5:
        print(
            f"Warning [{title}]: "
            f"decomposition closure difference = "
            f"{max_diff:.6f} Pg"
        )

def decomp_rows_periods(
    decomp: pd.DataFrame,
) -> pd.DataFrame:
    local = (
        decomp[
            decomp["zone"] == "Loess Plateau"
        ]
        .set_index("_period")
        .loc[PERIODS]
        .reset_index()
    )
    return local

def decomp_rows_zones(
    decomp: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    local = (
        decomp[
            (decomp["_period"] == period)
            & (decomp["zone"].isin(ZONE_ORDER))
        ]
        .set_index("zone")
        .loc[ZONE_ORDER]
        .reset_index()
    )
    return local

# Global bootstrap holder used only if SHOW_COMPONENT_CI=True.
GLOBAL_BOOTSTRAP = pd.DataFrame()

# =========================================================
# 7. NEW FIGURE 1
# Spatial patterns + process separation + early/late trends
# =========================================================
def plot_new_fig1(
    zones: gpd.GeoDataFrame,
) -> None:
    # Original logical order retained:
    # row 1 = state maps; row 2 = process separation; row 3 = trend contrasts.
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, 5.72)
    )

    gs = fig.add_gridspec(
        3,
        3,
        left=0.065,
        right=0.935,
        bottom=0.120,
        top=0.968,
        wspace=0.085,
        hspace=0.08,
    )

    axes = [
        fig.add_subplot(gs[i, j])
        for i in range(3)
        for j in range(3)
    ]

    ax_a, ax_b, ax_c = axes[0], axes[1], axes[2]
    ax_d, ax_e, ax_f = axes[3], axes[4], axes[5]
    ax_g, ax_h, ax_i = axes[6], axes[7], axes[8]

    # Manually nudge lower rows upward for a tighter layout.
    for ax in [ax_d, ax_e, ax_f]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + 0.034, pos.width, pos.height])

    for ax in [ax_g, ax_h, ax_i]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + 0.012, pos.width, pos.height])

    # ----- unified typography within Fig.1 -----
    map_tick_fs = 8.0
    panel_title_fs = 9.0
    cbar_label_fs = 8.0
    cbar_tick_fs = 6.8
    legend_font_fs = 8.0
    legend_patch_len = 1.35

    # -----------------------------------------------------
    # a-c: AGBD in 1990, 2000, 2023
    # -----------------------------------------------------
    agb_years = [1990, 2000, 2023]
    agb_data = [
        raster_lonlat(
            SPATIAL_DIR / f"forest_mean_agbd_{year}.tif"
        )
        for year in agb_years
    ]

    valid = np.concatenate(
        [arr[np.isfinite(arr)] for arr, _ in agb_data]
    )

    vmax = np.percentile(valid, 98.5)

    cmap_agb = mpl.colormaps["YlGn"].copy()
    cmap_agb.set_bad("white")

    agb_images = []

    for idx, (ax, year, (array, extent)) in enumerate(
        zip([ax_a, ax_b, ax_c], agb_years, agb_data)
    ):
        im = ax.imshow(
            array,
            extent=extent,
            origin="upper",
            cmap=cmap_agb,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        agb_images.append(im)

        style_map(
            ax,
            zones,
            show_y=(idx == 0),
            show_x=False,
            tick_labelsize=map_tick_fs,
        )
        label_ecological_zones(ax, zones)

        ax.set_title(
            f"({chr(97 + idx)}) Mean AGBD, {year}",
            loc="left",
            pad=2.2,
            fontsize=panel_title_fs,
            fontweight="normal",
        )

    fig.canvas.draw()
    pos_c = ax_c.get_position()

    cbar_h = pos_c.height * 0.90
    cbar_y0 = pos_c.y0 + (pos_c.height - cbar_h) / 2

    cax_agb = fig.add_axes(
        [pos_c.x1 + 0.0055, cbar_y0, 0.0105, cbar_h]
    )

    cbar = fig.colorbar(
        agb_images[-1],
        cax=cax_agb,
        orientation="vertical",
    )
    cbar.set_label(
        "Mean AGBD (Mg ha⁻¹)",
        fontsize=cbar_label_fs,
        labelpad=2.2,
    )
    cbar.ax.tick_params(
        length=2.0,
        width=0.7,
        pad=1.0,
        labelsize=cbar_tick_fs,
    )

    # -----------------------------------------------------
    # d: Forest change classes
    # -----------------------------------------------------
    classes, extent = raster_lonlat(
        SPATIAL_DIR / "forest_change_class_early_vs_late.tif",
        categorical=True,
    )

    forest_cmap = ListedColormap(FOREST_CHANGE_COLORS)
    forest_norm = BoundaryNorm(
        [0.5, 1.5, 2.5, 3.5, 4.5],
        forest_cmap.N,
    )

    ax_d.imshow(
        classes,
        extent=extent,
        origin="upper",
        cmap=forest_cmap,
        norm=forest_norm,
        interpolation="nearest",
    )
    style_map(
        ax_d,
        zones,
        show_y=True,
        show_x=False,
        tick_labelsize=map_tick_fs,
    )
    label_ecological_zones(ax_d, zones)
    ax_d.set_title(
        "(d) Forest change classes",
        loc="left",
        pad=2.2,
        fontsize=panel_title_fs,
        fontweight="normal",
    )

    # -----------------------------------------------------
    # e: Persistent-forest AGBD slope
    # -----------------------------------------------------
    path_e = TEMPORAL_DIR / "persistent_forest_agbd_theil_sen_slope_1990_2023.tif"
    arr_e, ext_e = raster_lonlat(path_e)
    bound_e = np.nanpercentile(np.abs(arr_e), 98)

    im_e = ax_e.imshow(
        arr_e,
        extent=ext_e,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound_e, vcenter=0, vmax=bound_e),
        interpolation="nearest",
    )
    style_map(
        ax_e,
        zones,
        show_y=False,
        show_x=False,
        tick_labelsize=map_tick_fs,
    )
    label_ecological_zones(ax_e, zones)
    ax_e.set_title(
        "(e) Persistent-forest AGBD slope",
        loc="left",
        pad=2.2,
        fontsize=panel_title_fs,
        fontweight="normal",
    )

    # -----------------------------------------------------
    # f: Forest-fraction slope
    # -----------------------------------------------------
    path_f = TEMPORAL_DIR / "forest_fraction_theil_sen_slope_1990_2023.tif"
    arr_f, ext_f = raster_lonlat(path_f)
    bound_f = np.nanpercentile(np.abs(arr_f), 98)

    im_f = ax_f.imshow(
        arr_f,
        extent=ext_f,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound_f, vcenter=0, vmax=bound_f),
        interpolation="nearest",
    )
    style_map(
        ax_f,
        zones,
        show_y=False,
        show_x=False,
        tick_labelsize=map_tick_fs,
    )
    label_ecological_zones(ax_f, zones)
    ax_f.set_title(
        "(f) Forest-fraction slope",
        loc="left",
        pad=2.2,
        fontsize=panel_title_fs,
        fontweight="normal",
    )

    # -----------------------------------------------------
    # g-h: AGBD trend classes
    # -----------------------------------------------------
    trend_specs = [
        (ax_g, "1990_1999", "(g) AGBD trend class, 1990–1999", True),
        (ax_h, "2000_2023", "(h) AGBD trend class, 2000–2023", False),
    ]

    for ax, period, title, show_y in trend_specs:
        arr, ext = raster_lonlat(
            TEMPORAL_DIR / f"landscape_agbd_trend_class_{period}.tif",
            categorical=True,
        )
        ax.imshow(
            arr,
            extent=ext,
            origin="upper",
            cmap=TREND_CMAP,
            norm=TREND_NORM,
            interpolation="nearest",
        )
        style_map(
            ax,
            zones,
            show_y=show_y,
            show_x=True,
            tick_labelsize=map_tick_fs,
        )
        label_ecological_zones(ax, zones)
        ax.set_title(
            title,
            loc="left",
            pad=2.2,
            fontsize=panel_title_fs,
            fontweight="normal",
        )

    # -----------------------------------------------------
    # i: AGBD slope change, late-early
    # -----------------------------------------------------
    diff_path = TEMPORAL_DIR / "landscape_agbd_slope_difference_late_minus_early.tif"
    diff, diff_ext = raster_lonlat(diff_path)
    diff_bound = np.nanpercentile(np.abs(diff), 98)

    im_i = ax_i.imshow(
        diff,
        extent=diff_ext,
        origin="upper",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-diff_bound, vcenter=0, vmax=diff_bound),
        interpolation="nearest",
    )
    style_map(
        ax_i,
        zones,
        show_y=False,
        show_x=True,
        tick_labelsize=map_tick_fs,
    )
    label_ecological_zones(ax_i, zones)
    ax_i.set_title(
        "(i) AGBD slope change, late–early",
        loc="left",
        pad=2.2,
        fontsize=panel_title_fs,
        fontweight="normal",
    )

    # -----------------------------------------------------
    # Legends / colorbars
    # -----------------------------------------------------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    pos_d = ax_d.get_position()
    pos_e = ax_e.get_position()
    pos_f = ax_f.get_position()
    pos_g = ax_g.get_position()
    pos_h = ax_h.get_position()
    pos_i = ax_i.get_position()

    tight_g = ax_g.get_tightbbox(renderer)
    tight_h = ax_h.get_tightbbox(renderer)
    tight_g_bottom = tight_g.y0 / fig.bbox.height
    tight_h_bottom = tight_h.y0 / fig.bbox.height

    # e/f horizontal colorbars: keep close to panel bottoms and match panel-c typography.
    ef_cbar_y_shift = 0.030
    ef_cbar_h = 0.0110

    for map_ax, image, label in [
        (ax_e, im_e, "Mg ha⁻¹ yr⁻¹"),
        (ax_f, im_f, "yr⁻¹"),
    ]:
        pos = map_ax.get_position()
        width = pos.width * 0.82
        left = pos.x0 + (pos.width - width) / 2

        cax = fig.add_axes(
            [left, pos.y0 - ef_cbar_y_shift, width, ef_cbar_h]
        )

        cb = fig.colorbar(
            image,
            cax=cax,
            orientation="horizontal",
        )
        cb.set_label(
            label,
            fontsize=cbar_label_fs,
            labelpad=0.7,
        )
        cb.ax.tick_params(
            length=1.8,
            width=0.7,
            pad=0.6,
            labelsize=cbar_tick_fs,
        )

    # Reference y position: center of e/f colorbar bars.
    ef_bar_center_y = pos_e.y0 - ef_cbar_y_shift + ef_cbar_h / 2

    # d legend: two aligned rows; same patch size/font as g-h legend.
    forest_handles = [Patch(facecolor=color, edgecolor="none", label=label) for color, label in zip(FOREST_CHANGE_COLORS, FOREST_CHANGE_LABELS)]
    d_row_h, d_row_gap = 0.020, 0.0080
    row1_y0 = ef_bar_center_y - d_row_h / 2
    row2_y0 = row1_y0 - d_row_h - d_row_gap

    # Row 2 remains unchanged; use its second-column start as the alignment target.
    d_leg_row2 = fig.add_axes([pos_d.x0, row2_y0, pos_d.width, d_row_h])
    d_leg_row2.axis("off")
    leg2 = d_leg_row2.legend(handles=[forest_handles[1], forest_handles[3]], labels=["Expansion", "Sparse/other"], loc="center", ncol=2, mode="expand", fontsize=legend_font_fs, handlelength=legend_patch_len, handleheight=0.95, handletextpad=0.32, columnspacing=0.0, borderaxespad=0.0, borderpad=0.0, frameon=False)

    # Draw once so the exact Sparse/other patch/text positions can be read.
    fig.canvas.draw()
    renderer_d = fig.canvas.get_renderer()
    row2_patches = leg2.get_patches()
    row2_texts = leg2.get_texts()
    sparse_patch_box = row2_patches[1].get_window_extent(renderer=renderer_d)
    sparse_text_box = row2_texts[1].get_window_extent(renderer=renderer_d)

    # Convert the Sparse/other left edges from display coordinates to row-1 axes coordinates.
    patch_x_target = d_leg_row2.transAxes.inverted().transform((sparse_patch_box.x0, sparse_patch_box.y0))[0]
    text_x_target = d_leg_row2.transAxes.inverted().transform((sparse_text_box.x0, sparse_text_box.y0))[0]

    # Row 1: keep Persistent at its existing position; manually place Contraction so that
    # both its color patch and text are left-aligned with Sparse/other below.
    d_leg_row1 = fig.add_axes([pos_d.x0, row1_y0, pos_d.width, d_row_h])
    d_leg_row1.axis("off")
    leg1 = d_leg_row1.legend(handles=[forest_handles[0]], labels=["Persistent"], loc="center left", fontsize=legend_font_fs, handlelength=legend_patch_len, handleheight=0.95, handletextpad=0.32, borderaxespad=0.0, borderpad=0.0, frameon=False)

    # Match the rendered patch size of Sparse/other exactly.
    patch_w_axes = d_leg_row2.transAxes.inverted().transform((sparse_patch_box.x1, sparse_patch_box.y0))[0] - patch_x_target
    patch_h_axes = d_leg_row2.transAxes.inverted().transform((sparse_patch_box.x0, sparse_patch_box.y1))[1] - d_leg_row2.transAxes.inverted().transform((sparse_patch_box.x0, sparse_patch_box.y0))[1]
    contraction_patch = mpl.patches.Rectangle((patch_x_target, 0.5 - patch_h_axes / 2), patch_w_axes, patch_h_axes, transform=d_leg_row1.transAxes, facecolor=FOREST_CHANGE_COLORS[2], edgecolor="none")
    d_leg_row1.add_patch(contraction_patch)
    d_leg_row1.text(text_x_target, 0.5, "Contraction", transform=d_leg_row1.transAxes, ha="left", va="center", fontsize=legend_font_fs, color="#222222")

    # g-h trend legend: same fontsize and handle size as panel d legend.
    trend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(TREND_COLORS, TREND_LABELS)
    ]

    trend_leg_h = 0.030
    trend_leg_y = min(tight_g_bottom, tight_h_bottom) - 0.036

    trend_leg_ax = fig.add_axes(
        [pos_g.x0, trend_leg_y, pos_h.x1 - pos_g.x0, trend_leg_h]
    )
    trend_leg_ax.axis("off")
    trend_leg_ax.legend(
        handles=trend_handles,
        loc="center",
        ncol=5,
        mode="expand",
        fontsize=legend_font_fs,
        handlelength=legend_patch_len,
        handleheight=0.95,
        handletextpad=0.32,
        columnspacing=0.0,
        borderaxespad=0.0,
        borderpad=0.0,
        frameon=False,
    )

    # Vertical colorbar for i: unify typography with the other colorbars.
    cbar_i_h = pos_i.height * 0.90
    cbar_i_y = pos_i.y0 + (pos_i.height - cbar_i_h) / 2

    cax_i = fig.add_axes(
        [pos_i.x1 + 0.0055, cbar_i_y, 0.0105, cbar_i_h]
    )

    cb_i = fig.colorbar(
        im_i,
        cax=cax_i,
        orientation="vertical",
    )
    cb_i.set_label(
        "Δ slope (Mg ha⁻¹ yr⁻¹)",
        fontsize=cbar_label_fs,
        labelpad=2.3,
    )
    cb_i.ax.tick_params(
        length=1.8,
        width=0.7,
        pad=0.7,
        labelsize=cbar_tick_fs,
    )

    save_bundle(
        fig,
        "Fig1_combined_spatial_patterns_and_trends",
    )

# =========================================================
# 8. NEW FIGURE 2
# Temporal trajectories + decomposition
# =========================================================
def plot_new_fig2(
    decomp: pd.DataFrame,
) -> None:
    annual = pd.read_csv(RESULT_DIR / "agb_plateau_year_summary_1990_2023.csv")

    if "forest_area_10k_km2" not in annual.columns:
        annual["forest_area_10k_km2"] = annual["forest_area_km2"] / 1e4

    years = annual["year"].to_numpy()

    # Unified typography hierarchy for Fig.2
    panel_title_fs = 9.0
    axis_label_fs = 8.0
    tick_fs = 7.0
    legend_fs = 7.0
    stat_fs = 6.2

    # Morandi-like soft palette coordinated with Fig.1
    morandi_area = "#AFC2A3"
    morandi_density = "#D7C29B"
    morandi_total = "#5C5C5C"

    fig = plt.figure(figsize=(FIG_WIDTH_IN, 5.10))
    gs = fig.add_gridspec(2, 3, left=0.080, right=0.965, top=0.955, bottom=0.125, wspace=0.26, hspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

    # a-c: annual trajectories
    series = [
        (ax_a, "total_agb_Pg", "(a) Total AGB stock", "AGB stock (Pg)", COLORS["blue"]),
        (ax_b, "forest_area_10k_km2", "(b) Forest area", "Forest area (10⁴ km²)", COLORS["green"]),
        (ax_c, "forest_mean_agbd_Mg_ha", "(c) Mean AGBD", "Mean AGBD (Mg ha⁻¹)", COLORS["gold"]),
    ]

    for ax, column, title, ylabel, color in series:
        values = annual[column].to_numpy(dtype=float)
        add_period_background(ax)
        ax.plot(years, values, "o", color=color, markerfacecolor="none", markeredgecolor=color, markeredgewidth=0.75, markersize=2.9, linestyle="none", zorder=3)
        add_stage_fits_and_stats(ax, years, values)
        ax.set_xlim(1990.0, 2023.8)
        ax.set_ylabel(ylabel, fontsize=axis_label_fs)
        ax.set_xlabel("")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5, zorder=0)
        ax.set_title(title, loc="left", pad=2.5, fontsize=panel_title_fs, fontweight="normal")
        style_nonmap(ax, full_box=True)
        ax.tick_params(axis="both", labelsize=tick_fs)

    for ax in [ax_a, ax_b, ax_c]:
        for txt in ax.texts:
            txt.set_fontsize(stat_fs)

    # d: Indexed trajectories
    indexed = {
        "Stock": annual["total_agb_Pg"] / annual["total_agb_Pg"].iloc[0] * 100,
        "Area": annual["forest_area_km2"] / annual["forest_area_km2"].iloc[0] * 100,
        "Mean AGBD": annual["forest_mean_agbd_Mg_ha"] / annual["forest_mean_agbd_Mg_ha"].iloc[0] * 100,
    }

    add_period_background(ax_d)
    for label, values, color in zip(indexed.keys(), indexed.values(), [COLORS["blue"], COLORS["green"], COLORS["gold"]]):
        ax_d.plot(years, values, label=label, color=color, linewidth=1.25, zorder=3)

    ax_d.axhline(100, color="#888888", linewidth=0.6)
    ax_d.set_xlim(1990.0, 2023.8)
    ax_d.set_ylabel("Relative index (1990 = 100)", fontsize=axis_label_fs)
    ax_d.set_xlabel("Year", fontsize=axis_label_fs)
    ax_d.grid(axis="y", color="#E8E8E8", linewidth=0.5, zorder=0)
    ax_d.set_title("(d) Indexed trajectories", loc="left", pad=2.5, fontsize=panel_title_fs, fontweight="normal")
    ax_d.legend(loc="upper left", fontsize=legend_fs, ncol=1, handlelength=1.4, handletextpad=0.4, borderaxespad=0.25, frameon=False)
    style_nonmap(ax_d, full_box=True)
    ax_d.tick_params(axis="both", labelsize=tick_fs)

    # e-f decomposition: same bar width and same palette
    shared_bar_width = 0.38

    # e: Loess Plateau decomposition
    plateau_rows = decomp_rows_periods(decomp)
    draw_decomposition(
        ax=ax_e,
        rows=plateau_rows,
        groups=["1990–1999", "2000–2023", "1990–2023"],
        mode="period",
        title="(e) Stock-change decomposition",
        show_ylabel=True,
        annotate_total=True,
        bar_width=shared_bar_width,
        area_color=morandi_area,
        density_color=morandi_density,
        total_color=morandi_total,
        title_fs=panel_title_fs,
        axis_label_fs=axis_label_fs,
        tick_fs=tick_fs,
        annotation_fs=stat_fs,
        xmargin=0.18,
        full_box=True,
    )
    ax_e.set_xlabel("Period", fontsize=axis_label_fs, labelpad=0.5)
    ax_e.set_ylim(0, 1.6)
    ax_e.tick_params(axis="both", labelsize=tick_fs)

    # f: Zonal decomposition, 2000-2023
    zonal_2000 = decomp_rows_zones(decomp, "2000-2023")
    draw_decomposition(
        ax=ax_f,
        rows=zonal_2000,
        groups=ZONE_ORDER,
        mode="zone",
        period="2000-2023",
        title="(f) 2000–2023 zonal decomposition",
        show_ylabel=False,
        annotate_total=False,
        bar_width=shared_bar_width,
        area_color=morandi_area,
        density_color=morandi_density,
        total_color=morandi_total,
        title_fs=panel_title_fs,
        axis_label_fs=axis_label_fs,
        tick_fs=tick_fs,
        annotation_fs=stat_fs,
        xmargin=0.08,
        full_box=True,
    )
    ax_f.set_xlabel("Ecological zone", fontsize=axis_label_fs)
    ax_f.set_ylim(0, 0.5)
    ax_f.tick_params(axis="both", labelsize=tick_fs)

    # Separate legends for e and f: 3 rows × 1 column, placed in upper-left whitespace.
    legend_handles = [
        Patch(facecolor=morandi_area, edgecolor="none", label="Forest area"),
        Patch(facecolor=morandi_density, edgecolor="none", label="Mean AGBD"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=morandi_total, markeredgecolor="white", markeredgewidth=0.45, markersize=5.0, label="Total ΔAGB"),
    ]

    legend_kw = dict(
        handles=legend_handles,
        ncol=1,
        fontsize=legend_fs,
        frameon=False,
        handlelength=1.15,
        handletextpad=0.45,
        labelspacing=0.35,
        borderaxespad=0.25,
    )

    ax_e.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), **legend_kw)
    ax_f.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), **legend_kw)

    save_bundle(fig, "Fig2_combined_temporal_trajectories_and_decomposition")

# =========================================================
# 9. SUPPLEMENTARY FIGURE S1
# Removed / additional spatial diagnostics
# =========================================================
def plot_supplementary_combined(zones: gpd.GeoDataFrame, decomp: pd.DataFrame) -> None:
    """Merged Supplementary Fig. S1 in a compact 3-row layout."""
    trends = pd.read_csv(RESULT_DIR / "agb_zone_and_plateau_period_trends.csv")
    panel_title_fs, axis_label_fs, tick_fs, legend_fs = 9.0, 8.0, 7.0, 7.0
    map_tick_fs, cbar_label_fs, cbar_tick_fs = 8.0, 8.0, 6.8
    morandi_area, morandi_density, morandi_total = "#AFC2A3", "#D7C29B", "#5C5C5C"

    fig = plt.figure(figsize=(FIG_WIDTH_IN, 6.90))
    gs = fig.add_gridspec(3, 6, left=0.070, right=0.965, bottom=0.085, top=0.965, wspace=0.32, hspace=0.40, height_ratios=[1.08, 0.92, 0.92])
    ax_a = fig.add_subplot(gs[0, 0:2]); ax_b = fig.add_subplot(gs[0, 2:4]); ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:2]); ax_e = fig.add_subplot(gs[1, 2:4]); ax_f = fig.add_subplot(gs[1, 4:6])
    ax_g = fig.add_subplot(gs[2, 0:3]); ax_h = fig.add_subplot(gs[2, 3:6])

    # Lift the whole d-f row upward to reduce the large blank gap below a-c.
    for ax_tmp in [ax_d, ax_e, ax_f]:
        pos_tmp = ax_tmp.get_position()
        ax_tmp.set_position([pos_tmp.x0, pos_tmp.y0 + 0.020, pos_tmp.width, pos_tmp.height])

    def draw_trend_two_row_legend(ax_row1, ax_row2):
        # Row 2 positions are the anchors. Row 1 Increase / Sig. increase align to them.
        patch_w, patch_h = 0.060, 0.48
        left_patch_x, left_text_x = 0.07, 0.15
        right_patch_x, right_text_x = 0.60, 0.68
        center_patch_x, center_text_x = 0.36, 0.44
        for ax_ in [ax_row1, ax_row2]:
            ax_.axis("off")
        row_items = {
            "row1": [
                (left_patch_x, left_text_x, TREND_COLORS[3], "Increase"),
                (center_patch_x, center_text_x, TREND_COLORS[2], "Stable"),
                (right_patch_x, right_text_x, TREND_COLORS[4], "Sig. increase"),
            ],
            "row2": [
                (left_patch_x, left_text_x, TREND_COLORS[1], "Decrease"),
                (right_patch_x, right_text_x, TREND_COLORS[0], "Sig. decrease"),
            ],
        }
        for patch_x, text_x, color, label in row_items["row1"]:
            ax_row1.add_patch(mpl.patches.Rectangle((patch_x, 0.26), patch_w, patch_h, transform=ax_row1.transAxes, facecolor=color, edgecolor="none"))
            ax_row1.text(text_x, 0.50, label, transform=ax_row1.transAxes, ha="left", va="center", fontsize=legend_fs, color="#222222")
        for patch_x, text_x, color, label in row_items["row2"]:
            ax_row2.add_patch(mpl.patches.Rectangle((patch_x, 0.26), patch_w, patch_h, transform=ax_row2.transAxes, facecolor=color, edgecolor="none"))
            ax_row2.text(text_x, 0.50, label, transform=ax_row2.transAxes, ha="left", va="center", fontsize=legend_fs, color="#222222")

    # (a) AGBD difference, late-early
    change, extent = raster_lonlat(SPATIAL_DIR / "change_landscape_agbd_late_minus_early.tif")
    bound = np.nanpercentile(np.abs(change), 98)
    im_a = ax_a.imshow(change, extent=extent, origin="upper", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), interpolation="nearest")
    style_map(ax_a, zones, show_y=True, show_x=True, tick_labelsize=map_tick_fs)
    label_ecological_zones(ax_a, zones)
    ax_a.set_title("(a) AGBD difference, late–early", loc="left", pad=2.5, fontsize=panel_title_fs, fontweight="normal")
    fig.canvas.draw()
    pos_a = ax_a.get_position()
    cbar_h = 0.0105
    cax_a = fig.add_axes([pos_a.x0 + pos_a.width * 0.09, pos_a.y0 - 0.047, pos_a.width * 0.82, cbar_h])
    cb_a = fig.colorbar(im_a, cax=cax_a, orientation="horizontal")
    cb_a.set_label("Difference (Mg ha⁻¹)", fontsize=cbar_label_fs, labelpad=0.8)
    cb_a.ax.tick_params(labelsize=cbar_tick_fs, length=2, pad=0.8)

    # (b) AGBD trend class, 1990-2023
    arr_b, ext_b = raster_lonlat(TEMPORAL_DIR / "landscape_agbd_trend_class_1990_2023.tif", categorical=True)
    ax_b.imshow(arr_b, extent=ext_b, origin="upper", cmap=TREND_CMAP, norm=TREND_NORM, interpolation="nearest")
    style_map(ax_b, zones, show_y=False, show_x=True, tick_labelsize=map_tick_fs)
    label_ecological_zones(ax_b, zones)
    ax_b.set_title("(b) AGBD trend class, 1990–2023", loc="left", pad=2.5, fontsize=panel_title_fs, fontweight="normal")
    fig.canvas.draw()
    pos_b = ax_b.get_position()
    row_h, row_gap = 0.018, 0.006
    row1_y0 = cax_a.get_position().y0 - (row_h - cbar_h) / 2.0
    row2_y0 = row1_y0 - row_h - row_gap
    b_row1 = fig.add_axes([pos_b.x0, row1_y0, pos_b.width, row_h])
    b_row2 = fig.add_axes([pos_b.x0, row2_y0, pos_b.width, row_h])
    draw_trend_two_row_legend(b_row1, b_row2)

    # (c) Detrended AGBD CV
    cv, cv_ext = raster_lonlat(SPATIAL_DIR / "landscape_agbd_detrended_cv_1990_2023.tif")
    cv_vmax = np.nanpercentile(cv, 98)
    im_c = ax_c.imshow(cv, extent=cv_ext, origin="upper", cmap="magma_r", norm=mpl.colors.Normalize(vmin=0, vmax=cv_vmax), interpolation="nearest")
    style_map(ax_c, zones, show_y=False, show_x=True, tick_labelsize=map_tick_fs)
    label_ecological_zones(ax_c, zones)
    ax_c.set_title("(c) Detrended AGBD CV", loc="left", pad=2.5, fontsize=panel_title_fs, fontweight="normal")
    fig.canvas.draw()
    pos_c = ax_c.get_position()
    cax_c = fig.add_axes([pos_c.x0 + pos_c.width * 0.09, pos_c.y0 - 0.047, pos_c.width * 0.82, cbar_h])
    cb_c = fig.colorbar(im_c, cax=cax_c, orientation="horizontal")
    cb_c.set_label("Detrended CV", fontsize=cbar_label_fs, labelpad=0.8)
    cb_c.ax.tick_params(labelsize=cbar_tick_fs, length=2, pad=0.8)

    # (d) 2023 zonal AGB stock and forest area
    summary = pd.read_csv(RESULT_DIR / "agb_spatial_pattern_summary_by_zone.csv")
    summary = summary[summary["period_or_year"].astype(str) == "2023"].set_index("zone").loc[ZONE_ORDER]
    x = np.arange(len(ZONE_ORDER))
    stock = summary["total_agb_Pg"].to_numpy(dtype=float)
    area_10k = summary["forest_area_km2"].to_numpy(dtype=float) / 1e4
    ax_d.bar(x, stock, color="#B7C9A8", edgecolor="#444444", linewidth=0.55, width=0.46, zorder=2)
    ax_d.set_xticks(x); ax_d.set_xticklabels(ZONE_ORDER)
    ax_d.set_xlabel("Ecological zone", fontsize=axis_label_fs)
    ax_d.set_ylabel("AGB stock (Pg)", fontsize=axis_label_fs)
    ax_d.set_ylim(0, np.nanmax(stock) * 1.18)
    ax_d.set_title("(d) 2023 zonal AGB stock and forest area", loc="left", pad=2.5, fontsize=panel_title_fs, fontweight="normal")
    ax_d.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
    style_nonmap(ax_d, full_box=True)
    ax_d.tick_params(axis="both", labelsize=tick_fs)
    ax_d2 = ax_d.twinx()
    ax_d2.scatter(x, area_10k, s=24, color=COLORS["gold"], edgecolor="white", linewidth=0.5, zorder=4)
    ax_d2.set_ylim(0, np.nanmax(area_10k) * 1.28)
    ax_d2.set_ylabel("Forest area (10⁴ km²)", fontsize=axis_label_fs, labelpad=2.0)
    ax_d2.tick_params(axis="y", direction="out", pad=1.2, length=2.5, width=0.7, labelsize=tick_fs)
    for spine in ax_d2.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.65); spine.set_color("#444444")
    fig.canvas.draw()
    d_y0, d_h = ax_d.get_position().y0, ax_d.get_position().height
    d_width = pos_a.width * 0.78
    for _ in range(2):
        new_pos_d = [pos_a.x0, d_y0, d_width, d_h]
        ax_d.set_position(new_pos_d); ax_d2.set_position(new_pos_d)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_right = ax_d2.yaxis.label.get_window_extent(renderer=renderer).x1 / fig.bbox.width
        d_width -= (label_right - pos_a.x1)
        d_width = np.clip(d_width, pos_a.width * 0.58, pos_a.width * 0.90)

    # Panel d legend: bar = AGB stock; circle = forest area.
    fig.canvas.draw()
    pos_d = ax_d.get_position()
    d_leg_ax = fig.add_axes([pos_d.x0, pos_d.y0 - 0.08, pos_d.width, 0.022])
    d_leg_ax.axis("off")
    d_handles = [
        Patch(facecolor="#B7C9A8", edgecolor="#444444", linewidth=0.55, label="AGB stock"),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=COLORS["gold"], markeredgecolor="white",
            markeredgewidth=0.5, markersize=5.0, label="Forest area"
        ),
    ]
    d_leg_ax.legend(
        handles=d_handles, loc="center", ncol=2,
        fontsize=legend_fs, handlelength=1.15,
        handletextpad=0.35, columnspacing=1.0,
        borderaxespad=0.0, frameon=False,
    )

    # (e) AGBD trend-class composition -> vertical stacked bars
    trend_area = pd.read_csv(RESULT_DIR / "agb_trend_class_area_plateau.csv")
    trend_area = trend_area[(trend_area["metric"] == "landscape_agbd") & trend_area["period"].isin(["1990_1999", "2000_2023", "1990_2023"])]
    classes = [2, 1, 0, -1, -2]
    comp_labels = ["Sig. increase", "Increase", "Stable", "Decrease", "Sig. decrease"]
    comp_colors = [COLORS["green"], "#A9D8C3", "#D9D9D9", "#E6B0AA", COLORS["red_dark"]]
    comp_periods = ["1990_1999", "2000_2023", "1990_2023"]
    comp_period_labels = ["1990–1999", "2000–2023", "1990–2023"]
    xpos = np.arange(len(comp_periods))
    bottom = np.zeros(len(comp_periods))
    for code, label, color in zip(classes, comp_labels, comp_colors):
        vals = [trend_area[(trend_area["period"] == period) & (trend_area["trend_class_code"] == code)]["percentage_of_valid_metric_area"].iloc[0] for period in comp_periods]
        ax_e.bar(xpos, vals, bottom=bottom, color=color, width=0.31, edgecolor="none", label=label)
        bottom += np.asarray(vals)
    ax_e.set_xticks(xpos)
    ax_e.set_xticklabels(comp_period_labels, rotation=0)
    ax_e.set_ylim(0, 100)
    ax_e.set_ylabel("Area proportion (%)", fontsize=axis_label_fs)
    ax_e.set_xlabel("Period", fontsize=axis_label_fs)
    ax_e.set_title("(e) AGBD trend-class composition", loc="left", pad=2.5, fontsize=panel_title_fs, fontweight="normal")
    ax_e.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    style_nonmap(ax_e, full_box=True)
    ax_e.tick_params(axis="both", labelsize=tick_fs)
    # Keep panel e span coordinated with panel b.
    fig.canvas.draw()
    e_y0, e_h = ax_e.get_position().y0, ax_e.get_position().height
    e_x0, e_x1 = pos_b.x0, pos_b.x1
    ax_e.set_position([e_x0, e_y0, e_x1 - e_x0, e_h])
    # True layout alignment for panel e:
    # panel-b left frame = left edge of ylabel; then ticks and left spine sit naturally to its right.
    ax_e.set_ylabel("Area proportion (%)", fontsize=axis_label_fs, labelpad=2.0)
    fig.canvas.draw()
    e_pos0 = ax_e.get_position()
    e_right = pos_b.x1
    e_left = pos_b.x0 + 0.045
    for _ in range(4):
        ax_e.set_position([e_left, e_pos0.y0, e_right - e_left, e_pos0.height])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ylabel_left = ax_e.yaxis.label.get_window_extent(renderer=renderer).x0 / fig.bbox.width
        e_left += pos_b.x0 - ylabel_left
        e_left = np.clip(e_left, pos_b.x0 + 0.020, e_right - 0.10)

    fig.canvas.draw()
    pos_e = ax_e.get_position()

    # Keep panel b fixed, but align the left title edge of panel e to panel b.
    ax_e._left_title.set_x((pos_b.x0 - pos_e.x0) / pos_e.width)

    # (f) 2000-2023 zonal relative changes with bottom horizontal colorbar
    metric_order = ["forest_area_km2", "forest_mean_agbd_Mg_ha", "total_agb_Pg"]
    zone_trends = trends[trends["unit"].isin(ZONE_ORDER) & (trends["period"] == "2000-2023") & trends["metric"].isin(metric_order)]
    matrix = zone_trends.pivot(index="unit", columns="metric", values="relative_change_pct").loc[ZONE_ORDER, metric_order].rename(columns={"forest_area_km2": "Area", "forest_mean_agbd_Mg_ha": "Mean AGBD", "total_agb_Pg": "Stock"})
    hm = sns.heatmap(matrix, ax=ax_f, cmap="BrBG", center=0, annot=True, fmt=".0f", annot_kws={"fontsize": 6.8}, cbar=False, linewidths=0.4, linecolor="white")
    ax_f.set_xlabel("")
    ax_f.set_ylabel("Ecological zone", fontsize=axis_label_fs)
    ax_f.set_title("(f) 2000–2023 zonal relative changes", loc="left", pad=2.5, fontsize=panel_title_fs, fontweight="normal")
    ax_f.tick_params(axis="x", rotation=0, labelsize=tick_fs, direction="out", pad=1.5, length=2.5, width=0.7)
    ax_f.tick_params(axis="y", rotation=0, labelsize=tick_fs, direction="out", pad=1.5, length=2.5, width=0.7)
    for spine in ax_f.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.65); spine.set_color("#444444")

    # True layout alignment for panel f:
    # keep the right frame aligned with panel c, but shrink from the left until
    # "Ecological zone" begins exactly at panel-c left frame. The A1–D labels
    # and y-axis spine then remain naturally between the ylabel and heatmap.
    ax_f.set_ylabel("Ecological zone", fontsize=axis_label_fs, labelpad=2.0)
    fig.canvas.draw()
    f_pos0 = ax_f.get_position()
    f_right = pos_c.x1
    f_left = pos_c.x0 + 0.055
    for _ in range(4):
        ax_f.set_position([f_left, f_pos0.y0, f_right - f_left, f_pos0.height])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ylabel_left = ax_f.yaxis.label.get_window_extent(renderer=renderer).x0 / fig.bbox.width
        f_left += pos_c.x0 - ylabel_left
        f_left = np.clip(f_left, pos_c.x0 + 0.025, f_right - 0.10)

    # Bottom horizontal colorbar, visually matching panels a and c.
    fig.canvas.draw()
    pos_f = ax_f.get_position()

    # Keep panel c fixed, but align the left title edge of panel f to panel c.
    ax_f._left_title.set_x((pos_c.x0 - pos_f.x0) / pos_f.width)

    cax_f = fig.add_axes([pos_f.x0 + pos_f.width * 0.09, pos_f.y0 - 0.047, pos_f.width * 0.82, cbar_h])
    cb_f = fig.colorbar(ax_f.collections[0], cax=cax_f, orientation="horizontal")
    cb_f.set_label("Relative change (%)", fontsize=cbar_label_fs, labelpad=0.8)
    cb_f.ax.tick_params(labelsize=cbar_tick_fs, length=2, pad=0.8, width=0.7)

    # Now place panel e's two-row legend lower, with the legend bottom aligned to
    # the bottom of panel f's colorbar title.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    f_cbar_label_bottom = cb_f.ax.xaxis.label.get_window_extent(renderer=renderer).y0 / fig.bbox.height
    e_legend_shift = 0.008
    e_row1_y0 = f_cbar_label_bottom + row_h + 0.006 - e_legend_shift
    e_row2_y0 = f_cbar_label_bottom - e_legend_shift

    e_row1 = fig.add_axes([pos_e.x0, e_row1_y0, pos_e.width, row_h])
    e_row2 = fig.add_axes([pos_e.x0, e_row2_y0, pos_e.width, row_h])
    draw_trend_two_row_legend(e_row1, e_row2)

    # (g-h) Zonal stock-change decomposition
    shared_bar_width = 0.38
    rows_g = decomp_rows_zones(decomp, "1990-1999")
    draw_decomposition(ax=ax_g, rows=rows_g, groups=ZONE_ORDER, mode="zone", period="1990-1999", title="(g) Zonal stock-change decomposition, 1990–1999", show_ylabel=True, annotate_total=False, bar_width=shared_bar_width, area_color=morandi_area, density_color=morandi_density, total_color=morandi_total, title_fs=panel_title_fs, axis_label_fs=axis_label_fs, tick_fs=tick_fs, annotation_fs=6.2, xmargin=0.08, full_box=True)
    ax_g.set_xlabel("Ecological zone", fontsize=axis_label_fs)

    rows_h = decomp_rows_zones(decomp, "1990-2023")
    draw_decomposition(ax=ax_h, rows=rows_h, groups=ZONE_ORDER, mode="zone", period="1990-2023", title="(h) Zonal stock-change decomposition, 1990–2023", show_ylabel=True, annotate_total=False, bar_width=shared_bar_width, area_color=morandi_area, density_color=morandi_density, total_color=morandi_total, title_fs=panel_title_fs, axis_label_fs=axis_label_fs, tick_fs=tick_fs, annotation_fs=6.2, xmargin=0.08, full_box=True)
    ax_h.set_xlabel("Ecological zone", fontsize=axis_label_fs)

    # Narrow both bottom panels and enlarge the gap between them.
    fig.canvas.draw()
    pos_g0, pos_h0 = ax_g.get_position(), ax_h.get_position()
    shrink_factor = 0.88
    new_g_w, new_h_w = pos_g0.width * shrink_factor, pos_h0.width * shrink_factor
    ax_g.set_position([pos_g0.x0, pos_g0.y0, new_g_w, pos_g0.height])
    ax_h.set_position([pos_h0.x1 - new_h_w, pos_h0.y0, new_h_w, pos_h0.height])

    # Use a common y-axis range for direct comparison.
    ax_g.set_ylim(0, 0.6)
    ax_h.set_ylim(0, 0.6)

    decomp_handles = [Patch(facecolor=morandi_area, edgecolor="none", label="Forest area"), Patch(facecolor=morandi_density, edgecolor="none", label="Mean AGBD"), Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=morandi_total, markeredgecolor="white", markeredgewidth=0.45, markersize=5.0, label="Total ΔAGB")]

    legend_kw = dict(handles=decomp_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=1, fontsize=legend_fs, handlelength=1.10, handletextpad=0.40, labelspacing=0.30, borderaxespad=0.0, frameon=False)
    ax_g.legend(**legend_kw)
    ax_h.legend(**legend_kw)

    save_bundle(fig, "FigS1_combined_supporting_results")

# =========================================================
# 11. Main
# =========================================================
def main() -> None:
    global GLOBAL_BOOTSTRAP

    apply_style()

    zones = load_zones()

    decomp, bootstrap = load_decomposition()
    GLOBAL_BOOTSTRAP = bootstrap

    plot_new_fig1(zones)
    plot_new_fig2(decomp)
    plot_supplementary_combined(zones, decomp)

    print(
        f"Figures written to: {FIGURE_DIR}"
    )

    print(
        "Generated:"
    )

    print(
        "  Fig1_combined_spatial_patterns_and_trends"
    )

    print(
        "  Fig2_combined_temporal_trajectories_and_decomposition"
    )

    print(
        "  FigS1_combined_supporting_results"
    )

    print(
        f"Component CI displayed: {SHOW_COMPONENT_CI}"
    )

if __name__ == "__main__":
    main()
