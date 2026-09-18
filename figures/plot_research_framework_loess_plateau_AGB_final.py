
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# =========================================================
# 1. Output
# =========================================================
OUT_DIR = FIGURE_OUTPUT_DIR
OUT_STEM = OUT_DIR / "Fig2_research_framework"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. Figure style
# =========================================================
FIG_WIDTH_MM = 180
FIG_HEIGHT_MM = 126
DPI = 800

FONT = "Times New Roman"
BASE_FS = 8.0
SMALL_FS = 7.0
HEADER_FS = 9.0
STAGE_FS = 9.6
FOOT_FS = 6.5
HARM_FS = 6.55
ANALYSIS_FS = 6.55
ANALYSIS_FOOT_FS = 6.15

mpl.rcParams.update({
    "font.family": FONT,
    "font.size": BASE_FS,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# Muted, publication-oriented palette
C_DATA = "#E7EDF3"
C_HARM = "#E6EFE9"
C_STOCK = "#F3E7D4"
C_PROCESS = "#DDE8F2"
C_SYNTH = "#E4ECE3"
C_ROBUST = "#ECE7F0"
C_WHITE = "#FFFFFF"

EDGE = "#4C5257"
TEXT = "#25282B"
ARROW = "#666C70"
MUTED = "#686D71"

# =========================================================
# 3. Helpers
# =========================================================
def box(ax, x, y, w, h, text, fc=C_WHITE, ec=EDGE, lw=0.72,
        fs=BASE_FS, weight="normal", ha="center", pad=0.006, r=0.009, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z
    )
    ax.add_patch(p)
    tx = x + w/2 if ha == "center" else x + 0.011
    ax.text(tx, y + h/2, text, ha=ha, va="center",
            fontsize=fs, color=TEXT, fontweight=weight,
            linespacing=1.12, zorder=z+1)
    return p

def panel(ax, x, y, w, h, title, fc):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.007,rounding_size=0.011",
        facecolor=fc, edgecolor=EDGE, linewidth=0.85, zorder=1
    )
    ax.add_patch(p)
    ax.text(x + 0.011, y + h - 0.017, title,
            ha="left", va="top", fontsize=HEADER_FS,
            fontweight="bold", color=TEXT, zorder=4)
    return p

def arrow(ax, x1, y1, x2, y2, lw=0.82, ms=7.5,
          rad=0.0, color=ARROW, ls="-", z=7):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=z
    )
    ax.add_patch(a)
    return a

def stage(ax, x, y, n, label):
    c = Circle((x, y), 0.012, facecolor=EDGE, edgecolor=EDGE, lw=0.5, zorder=10)
    ax.add_patch(c)
    ax.text(x, y, str(n), ha="center", va="center",
            fontsize=6.8, color="white", fontweight="bold", zorder=11)
    ax.text(x + 0.018, y, label, ha="left", va="center",
            fontsize=STAGE_FS, fontweight="bold", color=TEXT, zorder=11)

# =========================================================
# 4. Canvas
# =========================================================
fig = plt.figure(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

X1, W1 = 0.020, 0.180
X2, W2 = 0.220, 0.175
X3, W3 = 0.415, 0.365
X4, W4 = 0.800, 0.180
Y_MAIN, H_MAIN = 0.225, 0.695

stage(ax, X1 + 0.014, 0.965, 1, "Data")
stage(ax, X2 + 0.014, 0.965, 2, "Harmonization")
stage(ax, X3 + 0.014, 0.965, 3, "Process-separated analyses")
stage(ax, X4 + 0.014, 0.965, 4, "Synthesis")

# =========================================================
# 5. Data
# =========================================================
panel(ax, X1, Y_MAIN, W1, H_MAIN, "Data sources", C_DATA)

data = [
    ("Forest biomass/state\nCFATD / CATCD\n30 m; 1990–2023", 0.785),
    ("Climate\nHCD01\n0.1°; 1990–2022", 0.660),
    ("Terrain & land cover\nMERIT DEM / CLCD", 0.535),
    ("Grazing pressure\nLHGI", 0.410),
    ("Human activity\nNTL / GHSL population", 0.285),
]
for txt, y in data:
    box(ax, X1 + 0.018, y, W1 - 0.036, 0.086, txt, fs=SMALL_FS)

# =========================================================
# 6. Harmonization
# =========================================================
panel(ax, X2, Y_MAIN, W2, H_MAIN, "Common support", C_HARM)

box(
    ax, X2 + 0.018, 0.700, W2 - 0.036, 0.145,
    "1 km equal-area grid\n(EPSG:6933)\n\nForest fraction\nForest mean AGBD\nLandscape AGBD",
    fs=HARM_FS
)
box(
    ax, X2 + 0.018, 0.500, W2 - 0.036, 0.155,
    "0.1° HCD01 lattice\n\nArea-weighted\naggregation\n0.005° land-cover sampling\nPeriod means / slopes",
    fs=HARM_FS
)
box(
    ax, X2 + 0.018, 0.290, W2 - 0.036, 0.165,
    "Periods & predictor\nscreening\n\n1990–1999\n2000–2022/2023\n1990–2022/2023\n\nSpearman + VIF",
    fs=HARM_FS
)

arrow(ax, X1 + W1, 0.565, X2, 0.565, lw=1.0, ms=8.5)

# =========================================================
# 7. Analysis A — stock pathway
# =========================================================
panel(ax, X3, 0.590, W3, 0.330, "A. Biomass-stock pathway", C_STOCK)

box(ax, 0.436, 0.755, 0.094, 0.090, "Forest-state\nmetrics", fs=ANALYSIS_FS, weight="bold")
box(ax, 0.546, 0.755, 0.105, 0.090, "Trend & change\nTheil–Sen + MK\nForest classes", fs=ANALYSIS_FS)
box(ax, 0.667, 0.755, 0.093, 0.090, "AGB stock\nS = kAB", fs=ANALYSIS_FS, weight="bold")
arrow(ax, 0.530, 0.800, 0.546, 0.800)
arrow(ax, 0.651, 0.800, 0.667, 0.800)

box(
    ax, 0.436, 0.635, 0.150, 0.080,
    "Symmetric decomposition\nArea vs mean-AGBD\ncontribution",
    fs=ANALYSIS_FS
)
box(
    ax, 0.610, 0.635, 0.150, 0.080,
    "Stage & zonal\ncomparison\nEarly / late / full periods",
    fs=ANALYSIS_FS
)
arrow(ax, 0.714, 0.755, 0.714, 0.718)
arrow(ax, 0.610, 0.675, 0.588, 0.675)

ax.text(
    0.600, 0.610,
    "50 km spatial-block bootstrap for decomposition uncertainty",
    ha="center", va="center", fontsize=ANALYSIS_FOOT_FS, color=MUTED
)

arrow(ax, X2 + W2, 0.745, X3, 0.745, lw=1.0, ms=8.5)

# =========================================================
# 8. Analysis B — process-specific modelling
# =========================================================
panel(ax, X3, 0.225, W3, 0.335, "B. Process-specific predictive associations", C_PROCESS)

box(
    ax, 0.436, 0.425, 0.145, 0.082,
    "Response 1\nForest-cover change\n(fraction slope)",
    fs=ANALYSIS_FS, weight="bold"
)
box(
    ax, 0.605, 0.425, 0.155, 0.082,
    "Response 2\nPersistent-forest\nAGBD slope",
    fs=ANALYSIS_FS, weight="bold"
)

box(
    ax, 0.436, 0.345, 0.324, 0.052,
    "Predictors: initial state | climate | terrain | land cover\ngrazing | human activity",
    fs=ANALYSIS_FOOT_FS
)

box(
    ax, 0.436, 0.260, 0.150, 0.068,
    "Nested spatial\nGroupKFold\n5 outer × 3 inner folds\nState-only vs full XGBoost",
    fs=ANALYSIS_FOOT_FS
)
box(
    ax, 0.610, 0.260, 0.150, 0.068,
    "OOF evaluation &\ninterpretation\nR² / RMSE / MAE / Moran's I\nTreeSHAP + ALE",
    fs=ANALYSIS_FOOT_FS
)

arrow(ax, 0.509, 0.425, 0.509, 0.398)
arrow(ax, 0.683, 0.425, 0.683, 0.398)
arrow(ax, 0.598, 0.345, 0.511, 0.326, rad=0.05)
arrow(ax, 0.598, 0.345, 0.685, 0.326, rad=-0.05)
arrow(ax, 0.586, 0.294, 0.610, 0.294)

arrow(ax, X2 + W2, 0.395, X3, 0.395, lw=1.0, ms=8.5)

# =========================================================
# 9. Synthesis
# =========================================================
panel(ax, X4, Y_MAIN, W4, H_MAIN, "Analytical targets", C_SYNTH)

targets = [
    ("1", "Area–density pathway\nreorganization"),
    ("2", "Spatial coupling &\necological heterogeneity"),
    ("3", "Predictability &\npredictor structure"),
    ("4", "Temporal stability of\npredictive associations"),
]
ys = [0.760, 0.615, 0.470, 0.325]

for (n, txt), y in zip(targets, ys):
    c = Circle((0.818, y + 0.038), 0.013, facecolor="#718676", edgecolor="#718676", zorder=6)
    ax.add_patch(c)
    ax.text(0.818, y + 0.038, n, ha="center", va="center",
            fontsize=6.8, color="white", fontweight="bold", zorder=7)
    box(ax, 0.838, y, 0.128, 0.076, txt, fs=6.5)

ax.text(
    0.890, 0.255,
    "Interpretation boundary:\npredictive associations,\nnot causal effects",
    ha="center", va="center", fontsize=FOOT_FS, color=MUTED, fontstyle="italic"
)

arrow(ax, X3 + W3, 0.745, X4, 0.745, lw=0.95, ms=8)
arrow(ax, X3 + W3, 0.390, X4, 0.390, lw=0.95, ms=8)

# =========================================================
# 10. Compact process-separation cue
# =========================================================
panel(ax, X1, 0.030, W1, 0.145, "Process separation", "#F2F0EA")

# Minimal three-node schematic: no explanatory paragraph, avoiding overlap.
cx_left, cx_right, cx_stock = X1 + 0.052, X1 + 0.128, X1 + 0.090
cy_top, cy_bottom = 0.105, 0.055

c1 = Circle((cx_left, cy_top), 0.020, facecolor="#D8E6D8", edgecolor=EDGE, lw=0.65, zorder=5)
c2 = Circle((cx_right, cy_top), 0.020, facecolor="#DCE6EF", edgecolor=EDGE, lw=0.65, zorder=5)
c3 = Circle((cx_stock, cy_bottom), 0.022, facecolor="#EEDFC8", edgecolor=EDGE, lw=0.65, zorder=5)
ax.add_patch(c1); ax.add_patch(c2); ax.add_patch(c3)

ax.text(cx_left, cy_top, "Area", ha="center", va="center", fontsize=6.7, color=TEXT, fontweight="bold", zorder=6)
ax.text(cx_right, cy_top, "AGBD", ha="center", va="center", fontsize=6.7, color=TEXT, fontweight="bold", zorder=6)
ax.text(cx_stock, cy_bottom, "AGB\nstock", ha="center", va="center", fontsize=6.5, color=TEXT, fontweight="bold", linespacing=0.95, zorder=6)

arrow(ax, cx_left + 0.008, cy_top - 0.018, cx_stock - 0.012, cy_bottom + 0.018, lw=0.75, ms=7.0, color="#777C80")
arrow(ax, cx_right - 0.008, cy_top - 0.018, cx_stock + 0.012, cy_bottom + 0.018, lw=0.75, ms=7.0, color="#777C80")

# =========================================================
# 11. Robustness strip
# =========================================================
panel(ax, 0.220, 0.030, 0.760, 0.145, "Robustness & sensitivity", C_ROBUST)

robust_boxes = [
    (0.238, "Definition / product\nForest threshold & persistence\nCFATD–CLCD consistency"),
    (0.425, "Trend / accounting\nTFPW–MK\nEqual-length decomposition"),
    (0.612, "Sample / baseline\nCommon spatial sample\nAntecedent baseline"),
    (0.799, "Temporal sensitivity\nAlternative late windows\nRepeated spatial validation"),
]
for x, txt in robust_boxes:
    box(ax, x, 0.052, 0.157, 0.075, txt, fs=6.1)

arrow(ax, 0.600, 0.225, 0.600, 0.175, lw=0.72, ms=7, color="#8B818F", ls="--")
arrow(ax, 0.890, 0.225, 0.890, 0.175, lw=0.72, ms=7, color="#8B818F", ls="--")

# =========================================================
# 12. Save
# =========================================================
fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT_STEM}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{OUT_STEM}.tif", dpi=DPI, format="tiff",
            bbox_inches="tight", pad_inches=0.02)

plt.show()
plt.close(fig)
