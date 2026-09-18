"""Plot Fig. S5: alternative late-window performance and TreeSHAP structure."""
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
from matplotlib.transforms import Bbox

ROOT=PROCESS_DATA_DIR
OUTDIR=FIGURE_OUTPUT_DIR / "supplementary"
PRIMARY_PERFORMANCE=ROOT/"nested_spatial_xgboost_performance.csv"
ALTERNATIVE_PERFORMANCE=ROOT/"temporal_window_sensitivity_performance.csv"
STRUCTURE=ROOT/"Table_S15_alternative_window_predictor_structure.csv"
RANKING=ROOT/"Table_S15_alternative_window_ranking_concordance.csv"

PROCESS=["forest_fraction_change","persistent_forest_agbd_change"]

PROCESS_LABELS={
    "forest_fraction_change":"Forest-cover change",
    "persistent_forest_agbd_change":"Persistent-forest AGBD change"
}

DISPLAY={
    "forest_fraction_baseline":"Initial forest fraction",
    "persistent_forest_agbd_baseline":"Initial forest AGBD",
    "Tmean_mean":"Mean temperature",
    "PRE_mean":"Precipitation",
    "WS_mean":"Wind speed",
    "Tmean_slope":"Temperature trend",
    "PRE_slope":"Precipitation trend",
    "SLP":"Slope",
    "LHGI_mean":"Grazing intensity",
    "LHGI_sen_slope":"Grazing-intensity trend",
    "NTL_log1p_mean":"Nighttime light",
    "POP_log1p_density_change_yr":"Population-density change",
    "cropland_fraction":"Cropland fraction",
    "shrubland_fraction":"Shrubland fraction",
    "grassland_fraction":"Grassland fraction"
}

WINDOWS=[
    ("early","1990–1999"),
    ("equal","2000–2009"),
    ("main","2000–2022"),
    ("recent","2013–2022")
]

WINDOW_COLORS=["#C69289","#AAB7A2","#7896A5","#A493AF"]

MORANDI_CMAP=mpl.colors.LinearSegmentedColormap.from_list(
    "morandi_treeshap",
    ["#F1EEE8","#D5DDD6","#ABC0B8","#819FA0","#637F8D","#4B6475"]
)

HEATMAP_WIDTH_SCALE=0.80
HEATMAP_CBAR_GAP=0.006
HEATMAP_CBAR_WIDTH=0.014
CBAR_LABEL_PAD=4
SHARED_TITLE_OFFSET=0.043

# =========================================================
# c、d底部文字参数
# =========================================================
RANK_TEXT_GAP=0.013
RANK_TEXT_FONTSIZE=6.5
RANK_TEXT_LINESPACING=1.20

def apply_style():
    mpl.rcParams.update({
        "font.family":"Times New Roman",
        "font.size":8.8,
        "axes.titlesize":9.8,
        "axes.labelsize":9.0,
        "xtick.labelsize":8.0,
        "ytick.labelsize":8.0,
        "legend.fontsize":7.8,
        "axes.linewidth":0.7,
        "xtick.major.width":0.7,
        "ytick.major.width":0.7,
        "xtick.major.size":2.8,
        "ytick.major.size":2.8,
        "svg.fonttype":"none",
        "pdf.fonttype":42,
        "ps.fonttype":42,
        "axes.unicode_minus":False,
        "figure.facecolor":"white",
        "axes.facecolor":"white",
        "savefig.facecolor":"white",
        "mathtext.fontset":"custom",
        "mathtext.rm":"Times New Roman",
        "mathtext.it":"Times New Roman:italic",
        "mathtext.bf":"Times New Roman:bold"
    })

def style_axis(ax,grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#444444")
    ax.tick_params(direction="out",pad=2,length=2.8,width=0.7)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis,color="#E4E6E5",linewidth=0.5)

def title(ax,letter,text,pad=5):
    ax.set_title(
        f"({letter}) {text}",
        loc="left",
        pad=pad,
        fontsize=9.8,
        fontweight="normal",
        fontname="Times New Roman"
    )

def full_model_r2(primary,alternative,process):
    primary=primary.loc[
        (primary["process"].eq(process))&
        primary["model"].eq("full")
    ].set_index("period")

    alternative=alternative.loc[
        (alternative["process"].eq(process))&
        alternative["model"].eq("full")
    ].set_index("period")

    return np.array([
        primary.loc["early_1990_1999","R2"],
        alternative.loc["late_equal_2000_2009","R2"],
        primary.loc["late_2000_2022","R2"],
        alternative.loc["late_recent_2013_2022","R2"]
    ])

def plot_performance(ax,values,letter,process):
    x=np.arange(4)

    bars=ax.bar(
        x,
        values,
        width=0.46,
        color=WINDOW_COLORS,
        edgecolor="#596168",
        linewidth=0.55
    )

    for bar,value in zip(bars,values):
        ax.text(
            bar.get_x()+bar.get_width()/2,
            value+0.022,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontname="Times New Roman"
        )

    ax.set_xticks(x,[label for _,label in WINDOWS])
    ax.set_ylim(0,0.90)
    ax.set_ylabel("Full-model OOF R²",fontname="Times New Roman")

    style_axis(ax)
    title(ax,letter,PROCESS_LABELS[process])

def matrix_for_process(structure,process):
    subset=structure.loc[structure["process"].eq(process)].copy()

    equal=subset.loc[
        subset["alternative_window"].eq("late_equal_2000_2009")
    ].set_index("predictor")

    recent=subset.loc[
        subset["alternative_window"].eq("late_recent_2013_2022")
    ].set_index("predictor")

    predictors=sorted(
        set(equal.index)&set(recent.index),
        key=lambda v:-max(
            equal.loc[v,"early_relative_importance_percent"],
            equal.loc[v,"main_late_relative_importance_percent"],
            equal.loc[v,"alternative_relative_importance_percent"],
            recent.loc[v,"alternative_relative_importance_percent"]
        )
    )

    matrix=pd.DataFrame({
        "1990–1999":
            equal.loc[predictors,"early_relative_importance_percent"],
        "2000–2009":
            equal.loc[predictors,"alternative_relative_importance_percent"],
        "2000–2022":
            equal.loc[predictors,"main_late_relative_importance_percent"],
        "2013–2022":
            recent.loc[predictors,"alternative_relative_importance_percent"]
    })

    matrix.index=[
        DISPLAY.get(predictor,predictor)
        for predictor in predictors
    ]

    return matrix

def format_p_value(value):
    return "p < 0.001" if value<0.001 else f"p = {value:.3f}"

def make_rank_text(ranking,process):
    rank=ranking.loc[
        ranking["process"].eq(process)
    ].set_index("alternative_window")

    equal=rank.loc["late_equal_2000_2009"]
    recent=rank.loc["late_recent_2013_2022"]

    # -----------------------------------------------------
    # 图 c：固定两行
    # -----------------------------------------------------
    if process=="forest_fraction_change":
        return (
            f"ρ vs. main late: 2000–2009 = "
            f"{equal['ranking_rho_vs_main_late']:.3f} "
            f"({format_p_value(equal['ranking_p_vs_main_late'])})\n"
            f"2013–2022 = "
            f"{recent['ranking_rho_vs_main_late']:.3f} "
            f"({format_p_value(recent['ranking_p_vs_main_late'])})"
        )

    # -----------------------------------------------------
    # 图 d：固定两行
    # -----------------------------------------------------
    return (
        f"ρ vs. main late: 2000–2009 = "
        f"{equal['ranking_rho_vs_main_late']:.3f} "
        f"({format_p_value(equal['ranking_p_vs_main_late'])}); "
        f"2013–2022 = "
        f"{recent['ranking_rho_vs_main_late']:.3f} "
        f"({format_p_value(recent['ranking_p_vs_main_late'])})\n"
        f"2013–2022 vs. early = "
        f"{recent['ranking_rho_vs_early']:.3f} "
        f"({format_p_value(recent['ranking_p_vs_early'])})"
    )

def plot_heatmap(ax,matrix,process,letter,vmax):
    image=ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap=MORANDI_CMAP,
        vmin=0,
        vmax=vmax,
        interpolation="nearest"
    )

    ax.set_xticks(
        np.arange(matrix.shape[1]),
        matrix.columns
    )

    ax.set_yticks(
        np.arange(matrix.shape[0]),
        matrix.index
    )

    ax.tick_params(
        axis="y",
        labelsize=6.5,
        pad=1.2
    )

    ax.tick_params(
        axis="x",
        labelsize=7.5,
        rotation=0,
        pad=1.2
    )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value=matrix.iloc[row,col]

            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6.1,
                color="white" if value>=0.58*vmax else "#303638",
                fontname="Times New Roman"
            )

    ax.set_xticks(
        np.arange(-0.5,matrix.shape[1],1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5,matrix.shape[0],1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=0.65
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#5A5A5A")

    title(
        ax,
        letter,
        PROCESS_LABELS[process],
        pad=7
    )

    return image

def force_times_font(ax):
    ax.title.set_fontname("Times New Roman")
    ax.xaxis.label.set_fontname("Times New Roman")
    ax.yaxis.label.set_fontname("Times New Roman")

    for text in ax.get_xticklabels()+ax.get_yticklabels():
        text.set_fontname("Times New Roman")

    for text in ax.texts:
        text.set_fontname("Times New Roman")

def shift_axis_x(ax,dx):
    pos=ax.get_position()

    ax.set_position([
        pos.x0+dx,
        pos.y0,
        pos.width,
        pos.height
    ])

def align_d_colorbar_group_to_b(fig,ax_b,ax_d,cax):
    """
    Align visible right edge of d + colorbar + colorbar label
    with the right spine of panel b.
    """
    fig.canvas.draw()

    renderer=fig.canvas.get_renderer()

    target_right=ax_b.get_position().x1

    bbox_d=ax_d.get_tightbbox(
        renderer
    ).transformed(
        fig.transFigure.inverted()
    )

    bbox_cb=cax.get_tightbbox(
        renderer
    ).transformed(
        fig.transFigure.inverted()
    )

    group_bbox=Bbox.union([
        bbox_d,
        bbox_cb
    ])

    dx=target_right-group_bbox.x1

    shift_axis_x(ax_d,dx)
    shift_axis_x(cax,dx)

    fig.canvas.draw()

def add_rank_annotations(fig,ax_c,ax_d,ranking):
    """
    Two-line statistical annotations below panels c and d.
    Each annotation block is horizontally centered relative
    to the full visible extent of its corresponding subplot.
    """

    fig.canvas.draw()

    renderer=fig.canvas.get_renderer()

    # 获取包含变量名称、刻度等的完整可视范围
    bbox_c=ax_c.get_tightbbox(
        renderer
    ).transformed(
        fig.transFigure.inverted()
    )

    bbox_d=ax_d.get_tightbbox(
        renderer
    ).transformed(
        fig.transFigure.inverted()
    )

    # -----------------------------------------------------
    # 分别计算 c 和 d 整体的水平中心
    # -----------------------------------------------------
    center_c=(bbox_c.x0+bbox_c.x1)/2
    center_d=(bbox_d.x0+bbox_d.x1)/2

    # 两个文字块使用完全相同的纵向位置
    shared_y=min(
        bbox_c.y0,
        bbox_d.y0
    )-RANK_TEXT_GAP

    text_c=make_rank_text(
        ranking,
        PROCESS[0]
    )

    text_d=make_rank_text(
        ranking,
        PROCESS[1]
    )

    fig.text(
        center_c,
        shared_y,
        text_c,
        ha="center",
        va="top",
        multialignment="center",
        fontsize=RANK_TEXT_FONTSIZE,
        linespacing=RANK_TEXT_LINESPACING,
        fontname="Times New Roman"
    )

    fig.text(
        center_d,
        shared_y,
        text_d,
        ha="center",
        va="top",
        multialignment="center",
        fontsize=RANK_TEXT_FONTSIZE,
        linespacing=RANK_TEXT_LINESPACING,
        fontname="Times New Roman"
    )

def main():
    apply_style()

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    primary=pd.read_csv(
        PRIMARY_PERFORMANCE
    )

    alternative=pd.read_csv(
        ALTERNATIVE_PERFORMANCE
    )

    structure=pd.read_csv(
        STRUCTURE
    )

    ranking=pd.read_csv(
        RANKING
    )

    alternative.to_csv(
        OUTDIR/"source_data_FigS5_window_performance.csv",
        index=False
    )

    structure.to_csv(
        OUTDIR/"source_data_FigS5_window_treeshap.csv",
        index=False
    )

    ranking.to_csv(
        OUTDIR/"source_data_FigS5_window_ranking.csv",
        index=False
    )

    matrix_c=matrix_for_process(
        structure,
        PROCESS[0]
    )

    matrix_d=matrix_for_process(
        structure,
        PROCESS[1]
    )

    vmax=max(
        20,
        float(np.nanmax(matrix_c.to_numpy())),
        float(np.nanmax(matrix_d.to_numpy()))
    )

    fig=plt.figure(
        figsize=(180/25.4,160/25.4)
    )

    gs=fig.add_gridspec(
        2,
        4,
        width_ratios=[
            1.0,
            0.16,
            1.0,
            0.05
        ],
        height_ratios=[
            0.84,
            1.22
        ],
        left=0.075,
        right=0.955,
        bottom=0.125,
        top=0.955,
        wspace=0.08,
        hspace=0.40
    )

    ax_a=fig.add_subplot(
        gs[0,0]
    )

    ax_b=fig.add_subplot(
        gs[0,2]
    )

    ax_c=fig.add_subplot(
        gs[1,0]
    )

    ax_d=fig.add_subplot(
        gs[1,2]
    )

    cax=fig.add_subplot(
        gs[1,3]
    )

    plot_performance(
        ax_a,
        full_model_r2(
            primary,
            alternative,
            PROCESS[0]
        ),
        "a",
        PROCESS[0]
    )

    plot_performance(
        ax_b,
        full_model_r2(
            primary,
            alternative,
            PROCESS[1]
        ),
        "b",
        PROCESS[1]
    )

    image_c=plot_heatmap(
        ax_c,
        matrix_c,
        PROCESS[0],
        "c",
        vmax
    )

    plot_heatmap(
        ax_d,
        matrix_d,
        PROCESS[1],
        "d",
        vmax
    )

    for ax in [
        ax_a,
        ax_b,
        ax_c,
        ax_d
    ]:
        force_times_font(ax)

    fig.canvas.draw()

    pos_c=ax_c.get_position()
    pos_d=ax_d.get_position()

    y0=pos_c.y0
    h=pos_c.height

    w_c=pos_c.width*HEATMAP_WIDTH_SCALE
    w_d=pos_d.width*HEATMAP_WIDTH_SCALE

    left_c=pos_c.x0

    ax_c.set_position([
        left_c,
        y0,
        w_c,
        h
    ])

    provisional_d_right=(
        ax_b.get_position().x1
        -0.045
    )

    left_d=(
        provisional_d_right
        -w_d
    )

    left_cbar=(
        left_d
        +w_d
        +HEATMAP_CBAR_GAP
    )

    ax_d.set_position([
        left_d,
        y0,
        w_d,
        h
    ])

    cax.set_position([
        left_cbar,
        y0,
        HEATMAP_CBAR_WIDTH,
        h
    ])

    colorbar=fig.colorbar(
        image_c,
        cax=cax,
        orientation="vertical"
    )

    colorbar.set_label(
        "Relative TreeSHAP contribution (%)",
        fontsize=8.3,
        fontname="Times New Roman",
        labelpad=CBAR_LABEL_PAD
    )

    colorbar.ax.tick_params(
        labelsize=7.4,
        length=2.5,
        width=0.7,
        pad=2
    )

    colorbar.outline.set_linewidth(
        0.65
    )

    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontname(
            "Times New Roman"
        )

    colorbar.ax.yaxis.label.set_fontname(
        "Times New Roman"
    )

    # d + colorbar + colorbar文字整体
    # 与图b最右边界对齐
    align_d_colorbar_group_to_b(
        fig,
        ax_b,
        ax_d,
        cax
    )

    # =====================================================
    # c、d共同标题
    # =====================================================
    fig.canvas.draw()

    pos_c=ax_c.get_position()
    pos_d=ax_d.get_position()

    shared_x=(
        pos_c.x0
        +pos_d.x1
    )/2

    shared_y=max(
        pos_c.y1,
        pos_d.y1
    )+SHARED_TITLE_OFFSET

    fig.text(
        shared_x,
        shared_y,
        "Relative TreeSHAP contributions across time windows",
        ha="center",
        va="bottom",
        fontsize=9.4,
        fontname="Times New Roman"
    )

    # =====================================================
    # c、d底部统计文字：
    # 两行，并分别与c、d整体水平居中
    # =====================================================
    add_rank_annotations(
        fig,
        ax_c,
        ax_d,
        ranking
    )

    stem=OUTDIR/"FigS5_alternative_late_window_sensitivity"

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.02
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02
    )

    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={
            "compression":"tiff_lzw"
        }
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.close(fig)

if __name__=="__main__":
    main()