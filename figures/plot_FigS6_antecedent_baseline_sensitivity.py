"""Plot Fig. S6: sensitivity to antecedent rather than within-period baselines."""
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

ROOT=ROBUSTNESS_DATA_DIR
OUTDIR=FIGURE_OUTPUT_DIR / "supplementary"
PERFORMANCE=ROOT/"Table_S17_antecedent_baseline_performance_comparison.csv"
RANKING=ROOT/"Table_S17_antecedent_baseline_model_sensitivity.csv"

PROCESS_LABELS={"forest_fraction_change":"Forest-cover change","persistent_forest_agbd_change":"Persistent-forest AGBD change"}
PERIOD_LABELS={"early_1990_1999":"1990–1999","late_2000_2022":"2000–2022","full_1990_2022":"1990–2022"}

PROCESS_COLORS={"forest_fraction_change":"#8FAFC1","persistent_forest_agbd_change":"#9EB69A"}
MAIN_COLOR="#B9B2AA"
ANTECEDENT_COLOR="#A8BFA3"
EDGE_MAIN="#6E6761"
EDGE_ACCENT="#657A62"
EDGE_PROCESS="#5E6F79"

BAR_HEIGHT=0.53
PAIR_BAR_HEIGHT=0.40
PAIR_OFFSET=0.23

# 控制 0.276 和 0.284 两个黑色数值整体向右移动的距离
SPECIAL_LABEL_SHIFT=0.012

def apply_style():
    mpl.rcParams.update({
        "font.family":"Times New Roman",
        "font.size":8.2,
        "axes.titlesize":9.6,
        "axes.labelsize":8.8,
        "xtick.labelsize":7.8,
        "ytick.labelsize":7.8,
        "legend.fontsize":7.4,
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

def style_axis(ax,grid_axis="x"):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#444444")
    ax.tick_params(direction="out",pad=2,length=2.8,width=0.7)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis,color="#E6E8E9",linewidth=0.5)

def title(ax,letter,text):
    ax.set_title(f"({letter}) {text}",loc="left",pad=5,fontsize=9.6,fontweight="normal",fontname="Times New Roman")

def labels(frame):
    return [PERIOD_LABELS[row.period] for row in frame.itertuples()]

def apply_process_grouping(ax,frame):
    ax.axhspan(-0.5,2.5,color="#F4F7F8",zorder=0)
    ax.axhspan(2.5,5.5,color="#F4F7F3",zorder=0)
    ax.axhline(2.5,color="#AAB1B5",lw=0.65)
    for tick,row in zip(ax.get_yticklabels(),frame.itertuples()):
        tick.set_color(PROCESS_COLORS[row.process])
        tick.set_fontweight("bold")
        tick.set_fontname("Times New Roman")

def force_times_font(ax):
    ax.title.set_fontname("Times New Roman")
    ax.xaxis.label.set_fontname("Times New Roman")
    ax.yaxis.label.set_fontname("Times New Roman")
    for text in ax.get_xticklabels()+ax.get_yticklabels():
        text.set_fontname("Times New Roman")
    for text in ax.texts:
        text.set_fontname("Times New Roman")

def force_legend_font(legend):
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontname("Times New Roman")

def main():
    apply_style()
    OUTDIR.mkdir(parents=True,exist_ok=True)

    performance=pd.read_csv(PERFORMANCE)
    ranking=pd.read_csv(RANKING)

    full=performance.loc[performance["model"].eq("full")].copy()
    order=[
        ("forest_fraction_change","early_1990_1999"),
        ("forest_fraction_change","late_2000_2022"),
        ("forest_fraction_change","full_1990_2022"),
        ("persistent_forest_agbd_change","early_1990_1999"),
        ("persistent_forest_agbd_change","late_2000_2022"),
        ("persistent_forest_agbd_change","full_1990_2022")
    ]

    full["order"]=full.apply(lambda row:order.index((row["process"],row["period"])),axis=1)
    full=full.sort_values("order").reset_index(drop=True)
    ranking["order"]=ranking.apply(lambda row:order.index((row["process"],row["period"])),axis=1)
    ranking=ranking.sort_values("order").reset_index(drop=True)

    full.to_csv(OUTDIR/"source_data_FigS6_antecedent_baseline_performance.csv",index=False)
    ranking.to_csv(OUTDIR/"source_data_FigS6_antecedent_baseline_ranking.csv",index=False)

    fig,axes=plt.subplots(1,3,figsize=(180/25.4,68/25.4),constrained_layout=True)
    y=np.arange(len(full))

    # ======================== (a) Predictive performance ========================
    main_values=full["R2_main"].to_numpy()
    antecedent_values=full["R2_antecedent"].to_numpy()

    bars_main=axes[0].barh(
        y-PAIR_OFFSET,
        main_values,
        height=PAIR_BAR_HEIGHT,
        color=MAIN_COLOR,
        edgecolor=EDGE_MAIN,
        linewidth=0.55,
        label="Within-period baseline",
        zorder=2
    )

    bars_ant=axes[0].barh(
        y+PAIR_OFFSET,
        antecedent_values,
        height=PAIR_BAR_HEIGHT,
        color=ANTECEDENT_COLOR,
        edgecolor=EDGE_ACCENT,
        linewidth=0.55,
        label="Antecedent baseline",
        zorder=2
    )

    for bar,value in zip(bars_main,main_values):
        if abs(value-0.276)<0.001:
            axes[0].text(
                value+SPECIAL_LABEL_SHIFT,
                bar.get_y()+bar.get_height()/2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=6.8,
                color="black",
                fontname="Times New Roman"
            )
        else:
            axes[0].text(
                value-0.012,
                bar.get_y()+bar.get_height()/2,
                f"{value:.3f}",
                ha="right",
                va="center",
                fontsize=6.8,
                color="white",
                fontname="Times New Roman"
            )

    for bar,value in zip(bars_ant,antecedent_values):
        if abs(value-0.284)<0.001:
            axes[0].text(
                value+SPECIAL_LABEL_SHIFT,
                bar.get_y()+bar.get_height()/2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=6.8,
                color="black",
                fontname="Times New Roman"
            )
        else:
            axes[0].text(
                value-0.012,
                bar.get_y()+bar.get_height()/2,
                f"{value:.3f}",
                ha="right",
                va="center",
                fontsize=6.8,
                color="white",
                fontname="Times New Roman"
            )

    axes[0].set_yticks(y,labels(full))
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.20,0.86)
    axes[0].set_xlabel("Full-model OOF R²",fontname="Times New Roman")
    leg0=axes[0].legend(loc="lower right",frameon=False,ncol=1,handlelength=1.5,handletextpad=0.45,labelspacing=0.4)
    style_axis(axes[0])
    apply_process_grouping(axes[0],full)
    title(axes[0],"a","Predictive performance")
    force_legend_font(leg0)

    # ==================== (b) Predictor-ranking concordance ====================
    colors=[PROCESS_COLORS[row.process] for row in ranking.itertuples()]

    bars=axes[1].barh(
        y,
        ranking["rank_spearman_rho"],
        height=BAR_HEIGHT,
        color=colors,
        edgecolor=EDGE_PROCESS,
        linewidth=0.55
    )

    for bar,value in zip(bars,ranking["rank_spearman_rho"]):
        axes[1].text(
            value-0.008,
            bar.get_y()+bar.get_height()/2,
            f"{value:.3f}",
            ha="right",
            va="center",
            fontsize=6.8,
            color="#22313A",
            fontname="Times New Roman"
        )

    axes[1].set_yticks(y,labels(ranking))
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.80,1.005)
    axes[1].set_xlabel("Rank Spearman ρ vs. main baseline",fontname="Times New Roman")
    style_axis(axes[1])
    apply_process_grouping(axes[1],ranking)
    title(axes[1],"b","Predictor-ranking concordance")

    # ========================== (c) Top-5 consistency ==========================
    overlap=ranking["top5_overlap"].to_numpy()

    bars=axes[2].barh(
        y,
        overlap,
        height=BAR_HEIGHT,
        color=colors,
        edgecolor=EDGE_PROCESS,
        linewidth=0.55
    )

    for bar,value in zip(bars,overlap):
        axes[2].text(
            value-0.08,
            bar.get_y()+bar.get_height()/2,
            f"{int(value)}/5",
            ha="right",
            va="center",
            fontsize=6.9,
            color="white",
            fontname="Times New Roman"
        )

    axes[2].set_yticks(y,labels(ranking))
    axes[2].invert_yaxis()
    axes[2].set_xlim(0,5.2)
    axes[2].set_xticks(range(0,6))
    axes[2].set_xlabel("Top-5 predictor overlap",fontname="Times New Roman")
    style_axis(axes[2])
    apply_process_grouping(axes[2],ranking)
    title(axes[2],"c","Top-5 consistency")

    for ax in axes:
        force_times_font(ax)

    stem=OUTDIR/"FigS6_antecedent_baseline_sensitivity"

    for suffix,kwargs in {
        ".svg":{},
        ".pdf":{},
        ".tiff":{"dpi":600,"pil_kwargs":{"compression":"tiff_lzw"}},
        ".png":{"dpi":450}
    }.items():
        fig.savefig(
            stem.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.01,
            **kwargs
        )

    plt.close(fig)

if __name__=="__main__":
    main()