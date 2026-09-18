"""Loess Plateau study-area map for manuscript submission.
Panel (a): China in WGS84 Albers.
Panel (b): Loess Plateau in EPSG:6933.
Updated according to latest layout requests.
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
import warnings
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

STUDY_DIR = RAW_DATA_DIR / "study_area"
BOUNDARY_DIR = STUDY_DIR / "boundary"
COUNTIES_SHP = BOUNDARY_DIR / "黄土高原县域.shp"
YELLOW_RIVER_ROOT = RAW_DATA_DIR / "hydrography" / "yellow_river"
DEM_RASTER = STUDY_DIR / "DEM" / "DEM.img"
ZONES_SHP = RAW_DATA_DIR / "boundaries" / "huangtugaoyuan" / "Ecological_regionalization.shp"
CHINA_BOUNDARY_ROOT = RAW_DATA_DIR / "boundaries" / "china"

CHINA_PROVINCES_SHP = None
LAND_BOUNDARY_SHP = None
MARITIME_BOUNDARY_SHP = None
UNDEFINED_BOUNDARY_SHP = None
NINE_DASH_SHP = None

OUT_DIR = FIGURE_OUTPUT_DIR
OUT_STEM = OUT_DIR / "Fig1_study_area_ecological_regionalization"

CHINA_ALBERS = CRS.from_proj4("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs")
RIGHT_CRS = "EPSG:6933"

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
LEFT_MARGIN_IN = 0.28
RIGHT_MARGIN_IN = 0.22
TOP_MARGIN_IN = 0.34
BOTTOM_MARGIN_IN = 0.34
PANEL_GAP_IN = 0.17
MAX_B_TO_A_WIDTH = 1.16
RASTER_DPI = 800
SHOW_FIGURE = True

A_LEFT_PAD_FRAC = 0.026
A_RIGHT_PAD_FRAC = 0.120
A_TOP_PAD_FRAC = 0.026
A_BOTTOM_PAD_FRAC = 0.020

B_LEFT_PAD_M = 14000
B_RIGHT_PAD_M = 30000
B_TOP_PAD_M = 52000
B_BOTTOM_PAD_M = 32000
DEM_PAD_M = 40000

SCALEBAR_A_M = 500000
SCALEBAR_B_M = 100000

BASE_FONT = 7.5
TICK_FONT = 9.0
PANEL_FONT = 12.0
LEGEND_FONT = 8.5
LEGEND_TITLE_FONT = 8.5
ZONE_FONT = 9.2
NORTH_FONT = 9.0
CBAR_FONT = 7.0
SCALE_FONT = 7.0

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": BASE_FONT,
    "axes.linewidth": 0.75,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42
})

ZONE_FIELD = "CODE"
ZONE_ORDER = ["A1", "A2", "B1", "B2", "C", "D"]
ZONE_NAMES = {"A1": "A1", "A2": "A2", "B1": "B1", "B2": "B2", "C": "C", "D": "D"}
ZONE_COLOURS = {"A1": "#8EAEC6", "A2": "#AFC6DA", "B1": "#8FAF9B", "B2": "#B6CDBA", "C": "#D8C08F", "D": "#C99D8D"}

PROVINCE_EDGE = "#969696"
LAND_BORDER_CORE = "#62508D"
LAND_BORDER_HALO = "#C7BADD"
LAND_HALO_WIDTH = 1.20
LAND_CORE_WIDTH = 0.56
SEA_BORDER = "#159AC8"
SEA_BORDER_WIDTH = 0.76
UNDEFINED_BORDER = "#725891"
UNDEFINED_WIDTH = 0.62
NINE_DASH_COLOR = "#159AC8"
NINE_DASH_WIDTH = 0.76
YELLOW_RIVER_COLOR = "#174BD4"
STUDY_FACE = "#D9C3A3"
STUDY_EDGE = "#675493"
STUDY_ALPHA = 0.62
A2_X_SHIFT_FRAC = -0.070  # A2向左移动，占图b显示宽度的5.5%
A2_Y_SHIFT_FRAC = 0.000

def geometry_types(path):
    try:
        gdf = gpd.read_file(path)
        return set(gdf.geom_type.dropna().tolist())
    except Exception:
        return set()

def acceptable_polygon_layer(path):
    return bool(geometry_types(path) & {"Polygon", "MultiPolygon"})

def acceptable_boundary_layer(path):
    return bool(geometry_types(path) & {"LineString", "MultiLineString", "Polygon", "MultiPolygon"})

def find_shp(root, keywords, exclude=(), layer_type="any"):
    if not root.exists():
        return None
    candidates = []
    for p in root.rglob("*.shp"):
        name = p.stem.lower()
        if any(str(e).lower() in name for e in exclude):
            continue
        score = sum(str(k).lower() in name for k in keywords)
        if score <= 0:
            continue
        if layer_type == "polygon" and not acceptable_polygon_layer(p):
            continue
        if layer_type == "boundary" and not acceptable_boundary_layer(p):
            continue
        candidates.append((score, len(name), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]

def resolve_boundary_files():
    global CHINA_PROVINCES_SHP, LAND_BOUNDARY_SHP, MARITIME_BOUNDARY_SHP, UNDEFINED_BOUNDARY_SHP, NINE_DASH_SHP
    if CHINA_PROVINCES_SHP is None:
        CHINA_PROVINCES_SHP = find_shp(CHINA_BOUNDARY_ROOT, ["省级", "省界", "省"], ["国界", "九段", "海界", "市", "县"], "polygon")
    if LAND_BOUNDARY_SHP is None:
        LAND_BOUNDARY_SHP = find_shp(CHINA_BOUNDARY_ROOT, ["陆地国界", "陆界", "国界"], ["未定", "未定义", "海界", "九段"], "boundary")
    if MARITIME_BOUNDARY_SHP is None:
        MARITIME_BOUNDARY_SHP = find_shp(CHINA_BOUNDARY_ROOT, ["海界", "海域边界", "海岸线"], ["九段"], "boundary")
    if UNDEFINED_BOUNDARY_SHP is None:
        UNDEFINED_BOUNDARY_SHP = find_shp(CHINA_BOUNDARY_ROOT, ["未定国界", "未定义国界", "未定界", "未定"], [], "boundary")
    if NINE_DASH_SHP is None:
        NINE_DASH_SHP = find_shp(CHINA_BOUNDARY_ROOT, ["九段线", "九段", "南海断续线", "断续线"], [], "boundary")

def read_optional_vector(path, name):
    if path is None:
        warnings.warn(f"{name} not found.")
        return None
    gdf = gpd.read_file(path)
    if gdf.empty:
        return None
    if gdf.crs is None:
        b = gdf.total_bounds
        if -180 <= b[0] <= 180 and -180 <= b[2] <= 180 and -90 <= b[1] <= 90 and -90 <= b[3] <= 90:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            raise ValueError(f"{name} has no CRS: {path}")
    return gdf

def to_line_geoseries(gdf, name):
    if gdf is None:
        return None
    lines = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        gt = geom.geom_type
        if gt in ("LineString", "MultiLineString"):
            lines.append(geom)
        elif gt in ("Polygon", "MultiPolygon"):
            lines.append(geom.boundary)
        elif gt == "GeometryCollection":
            for subgeom in geom.geoms:
                if subgeom.geom_type in ("LineString", "MultiLineString"):
                    lines.append(subgeom)
                elif subgeom.geom_type in ("Polygon", "MultiPolygon"):
                    lines.append(subgeom.boundary)
    if not lines:
        warnings.warn(f"{name}: no usable line geometries; layer skipped.")
        return None
    return gpd.GeoSeries(lines, crs=gdf.crs)

def resolve_yellow_river_shp(root):
    """Resolve the Yellow River line shapefile from a file path or folder-like root.

    The supplied path may be a directory, a basename without .shp, or a direct
    shapefile. Line-type shapefiles are preferred; all features in the selected
    Yellow River file are retained so the river is not truncated by the former
    HYD1_4M_ID filter.
    """
    root = Path(root)
    candidates = []
    if root.suffix.lower() == ".shp" and root.exists():
        candidates.append(root)
    shp_same_name = root.with_suffix(".shp")
    if shp_same_name.exists():
        candidates.append(shp_same_name)
    if root.exists() and root.is_dir():
        direct = root / f"{root.name}.shp"
        if direct.exists():
            candidates.append(direct)
        candidates.extend(sorted(root.rglob("*.shp")))
    elif root.parent.exists():
        candidates.extend(sorted(root.parent.glob("*.shp")))

    # Deduplicate while preserving order.
    uniq = []
    seen = set()
    for p in candidates:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)

    scored = []
    for p in uniq:
        try:
            types = geometry_types(p)
        except Exception:
            types = set()
        if not (types & {"LineString", "MultiLineString"}):
            continue
        name = p.stem.lower()
        score = 0
        for kw, weight in [("黄河", 8), ("一级", 4), ("河流", 3), ("线状", 2)]:
            if kw in name:
                score += weight
        # Prefer an exact basename match if present.
        if p.stem == root.name:
            score += 20
        scored.append((score, len(p.stem), p))

    if not scored:
        raise FileNotFoundError(
            "未在给定路径中找到可用的黄河线状 shp 文件：\n"
            f"{root}\n"
            "请确认该目录下存在 LineString/MultiLineString 类型的 .shp 文件。"
        )

    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = scored[0][2]
    print("Yellow River shapefile:", chosen)
    print("Yellow River geometry:", geometry_types(chosen))
    return chosen


def load_yellow_river(root):
    """Read the complete Yellow River line layer and return it in EPSG:4326."""
    shp = resolve_yellow_river_shp(root)
    river = gpd.read_file(shp)
    if river.empty:
        raise ValueError(f"Yellow River shapefile is empty: {shp}")
    if river.crs is None:
        b = river.total_bounds
        if -180 <= b[0] <= 180 and -180 <= b[2] <= 180 and -90 <= b[1] <= 90 and -90 <= b[3] <= 90:
            river = river.set_crs("EPSG:4326")
        else:
            raise ValueError(f"Yellow River shapefile has no CRS: {shp}")
    # Keep only line/boundary geometries and retain ALL features in the new layer.
    river_lines = to_line_geoseries(river, "Yellow River")
    if river_lines is None or len(river_lines) == 0:
        raise ValueError(f"No line geometries found in Yellow River shapefile: {shp}")
    return gpd.GeoDataFrame(geometry=river_lines, crs=river_lines.crs).to_crs("EPSG:4326")

def add_north_arrow(ax, x=0.049, y=0.770, height=0.073, width=0.024):
    tip = (x, y + height)
    center = (x, y + height * 0.31)
    left = (x - width / 2, y)
    right = (x + width / 2, y)
    ax.add_patch(Polygon([tip, center, left], closed=True, transform=ax.transAxes, facecolor="black", edgecolor="black", linewidth=0.52, zorder=60, clip_on=False))
    ax.add_patch(Polygon([tip, right, center], closed=True, transform=ax.transAxes, facecolor="white", edgecolor="black", linewidth=0.52, zorder=61, clip_on=False))
    ax.plot([x, x], [y + height * 0.31, y + height], transform=ax.transAxes, color="black", lw=0.52, zorder=62, clip_on=False)
    ax.text(x, y + height + 0.008, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=NORTH_FONT, fontweight="normal", zorder=63)

def add_scale_bar(ax, x, y, length_m, label):
    h = max(length_m * 0.020, 2400)
    ax.plot([x, x + length_m], [y, y], color="0.10", lw=0.95, zorder=50)
    ax.plot([x, x], [y, y + h], color="0.10", lw=0.80, zorder=50)
    ax.plot([x + length_m, x + length_m], [y, y + h], color="0.10", lw=0.80, zorder=50)
    ax.text(x, y + h * 1.18, "0", ha="center", va="bottom", fontsize=SCALE_FONT)
    ax.text(x + length_m, y + h * 1.18, label, ha="center", va="bottom", fontsize=SCALE_FONT)

def add_graticule(ax, crs, lon_range, lat_range, lon_ticks, lat_ticks, x_side="top", y_side="left"):
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lats = np.linspace(lat_range[0], lat_range[1], 350)
    lons = np.linspace(lon_range[0], lon_range[1], 450)
    for lon in lon_ticks:
        x, y = tf.transform(np.full_like(lats, lon, dtype=float), lats)
        ax.plot(x, y, color="0.60", lw=0.24, alpha=0.32, zorder=1)
    for lat in lat_ticks:
        x, y = tf.transform(lons, np.full_like(lons, lat, dtype=float))
        ax.plot(x, y, color="0.60", lw=0.24, alpha=0.32, zorder=1)
    lon_mid, lat_mid = np.mean(lon_range), np.mean(lat_range)
    ax.set_xticks([tf.transform(float(v), float(lat_mid))[0] for v in lon_ticks])
    ax.set_yticks([tf.transform(float(lon_mid), float(v))[1] for v in lat_ticks])
    ax.set_xticklabels([f"{v}°" for v in lon_ticks], fontsize=TICK_FONT)
    ax.set_yticklabels([f"{v}°" for v in lat_ticks], fontsize=TICK_FONT)
    if x_side == "top":
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=2)
    else:
        ax.tick_params(axis="x", bottom=True, labelbottom=True, top=False, labeltop=False, pad=2)
    if y_side == "right":
        ax.tick_params(axis="y", right=True, labelright=True, left=False, labelleft=False, pad=2)
    else:
        ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, pad=2)

def read_dem_projected(src_path, target_crs, bounds, max_side=1800):
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("DEM has no CRS.")
        print("DEM native CRS:", src.crs)
        print(f"DEM native size: {src.width} x {src.height}")
        with WarpedVRT(src, crs=target_crs, resampling=Resampling.bilinear) as vrt:
            full = rasterio.windows.Window(0, 0, vrt.width, vrt.height)
            window = from_bounds(*bounds, transform=vrt.transform).round_offsets().round_lengths().intersection(full)
            if window.width <= 0 or window.height <= 0:
                raise ValueError("DEM window is empty after reprojection.")
            factor = max(max(window.width, window.height) / max_side, 1.0)
            ow = max(1, int(round(window.width / factor)))
            oh = max(1, int(round(window.height / factor)))
            dem = vrt.read(1, window=window, out_shape=(oh, ow), resampling=Resampling.bilinear, masked=True)
            trans = vrt.window_transform(window) * Affine.scale(window.width / ow, window.height / oh)
            west, south, east, north = array_bounds(oh, ow, trans)
    return dem, (west, south, east, north)

def get_china_view_bounds(provinces_a):
    minx, miny, maxx, maxy = provinces_a.total_bounds
    main_w = maxx - minx
    main_h = maxy - miny
    return minx - main_w * A_LEFT_PAD_FRAC, miny - main_h * A_BOTTOM_PAD_FRAC, maxx + main_w * A_RIGHT_PAD_FRAC, maxy + main_h * A_TOP_PAD_FRAC

def get_balanced_b_bounds(study_b, a_ratio):
    minx, miny, maxx, maxy = study_b.total_bounds
    xmin = minx - B_LEFT_PAD_M
    xmax = maxx + B_RIGHT_PAD_M
    ymin = miny - B_BOTTOM_PAD_M
    ymax = maxy + B_TOP_PAD_M
    xspan = xmax - xmin
    yspan = ymax - ymin
    max_ratio = a_ratio * MAX_B_TO_A_WIDTH
    current_ratio = xspan / yspan
    if current_ratio > max_ratio:
        required_yspan = xspan / max_ratio
        extra_y = required_yspan - yspan
        ymin -= extra_y * 0.45
        ymax += extra_y * 0.55
    return xmin, ymin, xmax, ymax

def add_nine_dash_inset(ax, provinces_ll, land_lines, sea_lines, undefined_lines, nine_lines):
    w, h = 0.165, 0.230
    inset = ax.inset_axes([1.0 - w, 0.0, w, h])
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    try:
        provinces_ll.boundary.plot(ax=inset, color="#8F8F8F", linewidth=0.22, zorder=2)
    except Exception:
        pass
    if land_lines is not None:
        land_lines.to_crs("EPSG:4326").plot(ax=inset, color=LAND_BORDER_CORE, linewidth=0.50, zorder=3)
    if sea_lines is not None:
        sea_lines.to_crs("EPSG:4326").plot(ax=inset, color=SEA_BORDER, linewidth=0.62, zorder=4)
    if undefined_lines is not None:
        undefined_lines.to_crs("EPSG:4326").plot(ax=inset, color=UNDEFINED_BORDER, linewidth=0.50, linestyle=(0, (4, 2.5)), zorder=5)
    if nine_lines is not None:
        nine_lines.to_crs("EPSG:4326").plot(ax=inset, color=NINE_DASH_COLOR, linewidth=0.68, linestyle=(0, (5, 3)), zorder=6)
    inset.set_xlim(105, 125)
    inset.set_ylim(3, 25)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.70)
        spine.set_color("#303030")

def save_figure(fig):
    fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(f"{OUT_STEM}.png", dpi=RASTER_DPI, bbox_inches="tight", pad_inches=0.015)
    try:
        fig.savefig(f"{OUT_STEM}.tif", dpi=RASTER_DPI, format="tiff", bbox_inches="tight", pad_inches=0.015, pil_kwargs={"compression": "tiff_lzw"})
    except TypeError:
        fig.savefig(f"{OUT_STEM}.tif", dpi=RASTER_DPI, format="tiff", bbox_inches="tight", pad_inches=0.015)
    print("\nFigure exported successfully:")
    for ext in ("svg", "pdf", "png", "tif"):
        print(f"  {OUT_STEM}.{ext}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolve_boundary_files()
    if CHINA_PROVINCES_SHP is None:
        raise FileNotFoundError("无法识别中国省级Polygon图层，请检查中国标准地图目录。")

    counties = gpd.read_file(COUNTIES_SHP)
    provinces = gpd.read_file(CHINA_PROVINCES_SHP)
    river_ll = load_yellow_river(YELLOW_RIVER_ROOT)
    zones = gpd.read_file(ZONES_SHP)

    if counties.crs is None or provinces.crs is None:
        raise ValueError("One or more core shapefiles have no CRS.")

    counties_ll = counties.to_crs("EPSG:4326")
    provinces_ll = provinces.to_crs("EPSG:4326")
    zones_ll = zones.set_crs("EPSG:4326", allow_override=True)

    if ZONE_FIELD not in zones_ll.columns:
        raise KeyError(f"Zone field '{ZONE_FIELD}' not found: {list(zones_ll.columns)}")

    zones_ll[ZONE_FIELD] = zones_ll[ZONE_FIELD].astype(str).str.strip()
    study_ll = counties_ll.dissolve()

    land_raw = read_optional_vector(LAND_BOUNDARY_SHP, "Land boundary")
    sea_raw = read_optional_vector(MARITIME_BOUNDARY_SHP, "Maritime boundary")
    undefined_raw = read_optional_vector(UNDEFINED_BOUNDARY_SHP, "Undefined boundary")
    nine_raw = read_optional_vector(NINE_DASH_SHP, "Nine-dash line")

    land_lines = to_line_geoseries(land_raw, "Land boundary")
    sea_lines = to_line_geoseries(sea_raw, "Maritime boundary")
    undefined_lines = to_line_geoseries(undefined_raw, "Undefined boundary")
    nine_lines = to_line_geoseries(nine_raw, "Nine-dash line")

    provinces_a = provinces_ll.to_crs(CHINA_ALBERS)
    province_boundary_a = provinces_a.boundary
    study_a = study_ll.to_crs(CHINA_ALBERS)
    river_a = river_ll.to_crs(CHINA_ALBERS)
    land_a = land_lines.to_crs(CHINA_ALBERS) if land_lines is not None else None
    sea_a = sea_lines.to_crs(CHINA_ALBERS) if sea_lines is not None else None
    undefined_a = undefined_lines.to_crs(CHINA_ALBERS) if undefined_lines is not None else None

    ax0, ay0, ax1, ay1 = get_china_view_bounds(provinces_a)
    a_ratio = (ax1 - ax0) / (ay1 - ay0)

    counties_b = counties_ll.to_crs(RIGHT_CRS)
    zones_b = zones_ll.to_crs(RIGHT_CRS)
    study_b = study_ll.to_crs(RIGHT_CRS)
    river_b = river_ll.to_crs(RIGHT_CRS)

    # 仅保留黄土高原边界范围内的黄河河段。
    # 使用几何相交而不是绘图裁剪，导出的SVG/PDF中也不会保留边界外河段。
    try:
        study_geom_b = study_b.geometry.union_all()
    except AttributeError:
        study_geom_b = study_b.geometry.unary_union
    river_b = river_b.copy()
    river_b["geometry"] = river_b.geometry.intersection(study_geom_b)
    river_b = river_b.loc[river_b.geometry.notna() & (~river_b.geometry.is_empty)].copy()

    bx0, by0, bx1, by1 = get_balanced_b_bounds(study_b, a_ratio)
    b_ratio = (bx1 - bx0) / (by1 - by0)

    dem_bounds = (bx0 - DEM_PAD_M, by0 - DEM_PAD_M, bx1 + DEM_PAD_M, by1 + DEM_PAD_M)
    dem, dem_extent = read_dem_projected(DEM_RASTER, RIGHT_CRS, dem_bounds)
    values = dem.compressed()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("DEM contains no valid values.")

    vmin = float(np.nanpercentile(values, 2))
    vmax = float(np.nanpercentile(values, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("dem", ["#173F7A", "#2E86B7", "#55B96A", "#A4D95E", "#F1E94D", "#F2A63A", "#B95C2B"], N=256)
    dem_float = dem.astype(np.float32).filled(np.nan)

    available_width = FIG_WIDTH_IN - LEFT_MARGIN_IN - RIGHT_MARGIN_IN - PANEL_GAP_IN
    panel_a_width = available_width / 2.0
    panel_b_width = available_width / 2.0
    common_height = min(panel_a_width / a_ratio, panel_b_width / b_ratio)
    fig_height = TOP_MARGIN_IN + common_height + BOTTOM_MARGIN_IN
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))

    pos_a = [LEFT_MARGIN_IN / FIG_WIDTH_IN, BOTTOM_MARGIN_IN / fig_height, panel_a_width / FIG_WIDTH_IN, common_height / fig_height]
    pos_b = [(LEFT_MARGIN_IN + panel_a_width + PANEL_GAP_IN) / FIG_WIDTH_IN, BOTTOM_MARGIN_IN / fig_height, panel_b_width / FIG_WIDTH_IN, common_height / fig_height]
    ax_a = fig.add_axes(pos_a)
    ax_b = fig.add_axes(pos_b)

    # Panel A
    ax_a.set_facecolor("white")
    province_boundary_a.plot(ax=ax_a, color=PROVINCE_EDGE, linewidth=0.30, zorder=4)
    if land_a is not None:
        land_a.plot(ax=ax_a, color=LAND_BORDER_HALO, linewidth=LAND_HALO_WIDTH, alpha=0.60, zorder=10)
        land_a.plot(ax=ax_a, color=LAND_BORDER_CORE, linewidth=LAND_CORE_WIDTH, zorder=11)
    if sea_a is not None:
        sea_a.plot(ax=ax_a, color=SEA_BORDER, linewidth=SEA_BORDER_WIDTH, zorder=12)
    if undefined_a is not None:
        undefined_a.plot(ax=ax_a, color=UNDEFINED_BORDER, linewidth=UNDEFINED_WIDTH, linestyle=(0, (4, 2.5)), zorder=13)
    study_a.plot(ax=ax_a, facecolor=STUDY_FACE, edgecolor=STUDY_EDGE, linewidth=0.32, alpha=STUDY_ALPHA, zorder=17)
    river_a.plot(ax=ax_a, color=YELLOW_RIVER_COLOR, linewidth=0.95, zorder=25)
    ax_a.set_xlim(ax0, ax1)
    ax_a.set_ylim(ay0, ay1)
    ax_a.set_aspect("equal", adjustable="box")
    add_graticule(ax_a, CHINA_ALBERS, (70, 136), (5, 55), [70, 80, 90, 100, 110, 120, 130], [20, 30, 40, 50], "top", "left")
    ax_a.set_xlim(ax0, ax1)
    ax_a.set_ylim(ay0, ay1)
    ax_a.text(0.018, 0.980, "(a)", transform=ax_a.transAxes, ha="left", va="top", fontsize=PANEL_FONT, fontweight="normal", zorder=80)
    add_north_arrow(ax_a, x=0.935, y=0.779, height=0.095, width=0.033)

    # National, maritime and undefined boundaries remain visible for map-policy
    # compliance, but do not need to compete with the three scientific cues.
    handles_a = [
        Patch(facecolor=STUDY_FACE, edgecolor=STUDY_EDGE, alpha=STUDY_ALPHA, label="Study area"),
        Line2D([0], [0], color=YELLOW_RIVER_COLOR, lw=0.9, label="Yellow River"),
        Line2D([0], [0], color=PROVINCE_EDGE, lw=0.55, label="Provincial boundary"),
    ]
    ax_a.legend(handles=handles_a, loc="lower left", bbox_to_anchor=(0.018, 0.0), fontsize=LEGEND_FONT, frameon=False, ncol=1, handlelength=2.0, handletextpad=0.42, labelspacing=0.16, borderaxespad=0)

    a_span_x = ax1 - ax0
    a_span_y = ay1 - ay0
    add_scale_bar(ax_a, ax0 + a_span_x * 0.64, ay0 + a_span_y * 0.04, SCALEBAR_A_M, "500 km")
    add_nine_dash_inset(ax_a, provinces_ll, land_lines, sea_lines, undefined_lines, nine_lines)

    # Panel B
    west, south, east, north = dem_extent
    ax_b.imshow(dem_float, extent=(west, east, south, north), origin="upper", cmap=cmap, norm=norm, interpolation="bilinear", zorder=0)
    counties_b.boundary.plot(ax=ax_b, color="#8D8D8D", linewidth=0.22, alpha=0.54, zorder=3)

    for code in ZONE_ORDER:
        zone = zones_b.loc[zones_b[ZONE_FIELD] == code]
        if zone.empty:
            continue
        zone.plot(ax=ax_b, facecolor=ZONE_COLOURS[code], edgecolor="none", alpha=0.10, zorder=4)
        zone.boundary.plot(ax=ax_b, color=ZONE_COLOURS[code], linewidth=0.95, zorder=5)
        try:
            merged = zone.geometry.union_all()
        except AttributeError:
            merged = zone.geometry.unary_union
        pt = merged.representative_point()
        x_text, y_text = pt.x, pt.y
        if code == "A2":
            # 使用图b显示范围的相对比例移动，避免固定公里数变化不明显。
            # 负值向左，正值向右；当前为向左5.5%。
            x_text += (bx1 - bx0) * A2_X_SHIFT_FRAC
            y_text += (by1 - by0) * A2_Y_SHIFT_FRAC
        ax_b.text(
            x_text, y_text, code, ha="center", va="center", fontsize=ZONE_FONT,
            fontweight="bold", color="#1E1E1E",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")], zorder=9,
        )

    study_b.boundary.plot(ax=ax_b, color="#333333", linewidth=0.90, zorder=7)
    river_b.plot(ax=ax_b, color=YELLOW_RIVER_COLOR, linewidth=1.00, zorder=8)

    ax_b.set_xlim(bx0, bx1)
    ax_b.set_ylim(by0, by1)
    ax_b.set_aspect("equal", adjustable="box")
    add_graticule(ax_b, RIGHT_CRS, (101, 115), (33, 42), [102, 105, 108, 111, 114], [33, 36, 39, 42], "top", "right")
    ax_b.set_xlim(bx0, bx1)
    ax_b.set_ylim(by0, by1)
    ax_b.text(0.018, 0.980, "(b)", transform=ax_b.transAxes, ha="left", va="top", fontsize=PANEL_FONT, fontweight="normal", zorder=80)

    cax = ax_b.inset_axes([0.030, 0.070, 0.220, 0.030])
    cax.set_facecolor((1, 1, 1, 0.72))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.0f}", f"{vmax:.0f}"])
    cbar.ax.tick_params(labelsize=CBAR_FONT, length=1.6, pad=1)
    cbar.outline.set_linewidth(0.42)
    ax_b.text(0.030, 0.109, "Elevation (m)", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=BASE_FONT,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.7))

    b_span_x = bx1 - bx0
    b_span_y = by1 - by0
    add_scale_bar(ax_b, bx1 - b_span_x * 0.07 - SCALEBAR_B_M, by0 + b_span_y * 0.045, SCALEBAR_B_M, "100 km")

    for ax in (ax_a, ax_b):
        ax.tick_params(direction="out", length=2.8, width=0.70)
        for spine in ax.spines.values():
            spine.set_linewidth(0.78)

    save_figure(fig)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
