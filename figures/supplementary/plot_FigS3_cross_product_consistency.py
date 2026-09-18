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

ROOT=PROCESS_DATA_DIR
OUTDIR=FIGURE_OUTPUT_DIR / "supplementary"
THRESHOLD=ROOT/"persistent_forest_threshold_sensitivity_reference_comparison.csv"
COMMON=ROOT/"Table_S14_common_sample_performance_and_concordance.csv"
PRIMARY_PERFORMANCE=ROOT/"nested_spatial_xgboost_performance.csv"

PERIOD_ORDER=["early_1990_1999","late_2000_2022","full_1990_2022"]
PERIOD_LABELS={"early_1990_1999":"1990–1999","late_2000_2022":"2000–2022","full_1990_2022":"1990–2022"}
PERIOD_COLORS={"early_1990_1999":"#C98E86","late_2000_2022":"#7DA6BE","full_1990_2022":"#7EA28A"}

def apply_style():
    mpl.rcParams.update({
        "font.family":"Times New Roman",
        "font.size":8.2,
        "axes.titlesize":9.6,
        "axes.labelsize":8.8,
        "xtick.labelsize":8.0,
        "ytick.labelsize":8.0,
        "legend.fontsize":7.6,
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
        "savefig.facecolor":"white"
    })

def style_axis(ax,grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#444444")
    ax.tick_params(direction="out",pad=2,length=2.8,width=0.7)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis,color="#E3E6E8",linewidth=0.5)

def title(ax,letter,text):
    ax.set_title(f"({letter}) {text}",loc="left",pad=5,fontsize=9.6,fontweight="normal",fontname="Times New Roman")

def configuration_axis(ax,threshold,metric,ylabel,ylim,multiplier=1.0):
    configurations=[(0.05,0.80),(0.05,0.90),(0.05,1.00),(0.10,0.80),(0.10,0.90),(0.10,1.00),(0.20,0.80),(0.20,0.90),(0.20,1.00)]
    x=np.arange(len(configurations))
    for period in PERIOD_ORDER:
        subset=threshold.loc[threshold["period"].eq(period)].copy()
        values=[]
        for forest,persistence in configurations:
            value=subset.loc[subset["forest_presence_threshold"].eq(forest)&subset["temporal_persistence_threshold"].eq(persistence),metric].iloc[0]
            values.append(value*multiplier)
        ax.plot(x,values,marker="o",markersize=3.5,lw=1.1,color=PERIOD_COLORS[period],label=PERIOD_LABELS[period])
    for boundary in [2.5,5.5]:
        ax.axvline(boundary,color="#BFC4C7",lw=0.55,ls="--",zorder=0)
    ax.axvline(4,color="#555B60",lw=0.75,ls=":",zorder=0)
    ax.set_xticks(x,[f"{int(forest*100)}\n{int(persistence*100)}" for forest,persistence in configurations])
    ax.set_xlabel("Forest threshold (%) / temporal persistence (%)",fontname="Times New Roman")
    ax.set_ylabel(ylabel,fontname="Times New Roman")
    ax.set_ylim(*ylim)
    ax.text(4,ylim[0]+0.04*(ylim[1]-ylim[0]),"Primary",ha="center",va="bottom",fontsize=7.0,color="#40464B",fontname="Times New Roman")
    style_axis(ax)

def main():
    apply_style()
    OUTDIR.mkdir(parents=True,exist_ok=True)
    threshold=pd.read_csv(THRESHOLD)
    common=pd.read_csv(COMMON)
    threshold.to_csv(OUTDIR/"source_data_FigS4_threshold_definition.csv",index=False)
    common.to_csv(OUTDIR/"source_data_FigS4_common_sample.csv",index=False)

    fig,axes=plt.subplots(2,2,figsize=(180/25.4,124/25.4))
    fig.subplots_adjust(left=0.085,right=0.985,bottom=0.09,top=0.955,wspace=0.28,hspace=0.44)
    ax_a,ax_b,ax_c,ax_d=axes.ravel()

    configuration_axis(ax_a,threshold,"spearman_rho_vs_reference","Spearman ρ vs. primary",(0.975,1.001))
    title(ax_a,"a","Forest-definition sensitivity")
    ax_a.legend(loc="upper right",bbox_to_anchor=(0.985,0.985),ncol=1,frameon=False,columnspacing=0.8,handlelength=1.5,handletextpad=0.45,labelspacing=0.45,borderaxespad=0.0)

    configuration_axis(ax_b,threshold,"slope_sign_agreement_vs_reference","Slope-sign agreement (%)",(97.5,100.1),multiplier=100)
    title(ax_b,"b","Sign agreement under alternative definitions")

    common=common.loc[common["model"].eq("full")].copy()
    common["period_label"]=common["period"].map(PERIOD_LABELS)
    primary=pd.read_csv(PRIMARY_PERFORMANCE)
    primary_r2=primary.loc[(primary["process"].eq("persistent_forest_agbd_change"))&primary["model"].eq("full"),["period","R2"]].set_index("period")["R2"].to_dict()
    common=common.reset_index(drop=True)

    x_c=np.arange(len(common))
    width_c=0.25
    primary_values=np.array([primary_r2[p] for p in common["period"]])
    common_values=common["R2"].to_numpy()

    bars_primary=ax_c.bar(x_c-width_c/2,primary_values,width=width_c,color="#7A838A",edgecolor="#4B5358",linewidth=0.55,label="Primary sample")
    bars_common=ax_c.bar(x_c+width_c/2,common_values,width=width_c,color="#7DA6BE",edgecolor="#4B6575",linewidth=0.55,label="Common sample")

    for bar in bars_primary:
        ax_c.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.012,f"{bar.get_height():.3f}",ha="center",va="bottom",fontsize=7.2,fontname="Times New Roman")
    for bar in bars_common:
        ax_c.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.012,f"{bar.get_height():.3f}",ha="center",va="bottom",fontsize=7.2,fontname="Times New Roman")

    ax_c.set_xticks(x_c,common["period_label"])
    ymin=min(primary_values.min(),common_values.min())-0.06
    ymax=max(primary_values.max(),common_values.max())+0.09
    ax_c.set_ylim(max(0,ymin),ymax)
    ax_c.set_ylabel("OOF R²",fontname="Times New Roman")
    ax_c.legend(loc="upper center",ncol=2,frameon=False,columnspacing=1.2,handlelength=1.4,handletextpad=0.45)
    style_axis(ax_c)
    title(ax_c,"c","Primary versus common-sample performance")

    x_d=np.arange(len(common))
    width_d=0.25
    bars=ax_d.bar(x_d,common["ranking_spearman_rho_vs_primary"],width=width_d,color="#93B3A1",edgecolor="#44515A",linewidth=0.55)

    for bar,row in zip(bars,common.itertuples()):
        ax_d.text(bar.get_x()+bar.get_width()/2,bar.get_height()-0.010,f"{bar.get_height():.3f}",ha="center",va="top",fontsize=7.2,fontname="Times New Roman")
        ax_d.text(bar.get_x()+bar.get_width()/2,0.815,f"Top-5: {row.top5_overlap_count}/5",ha="center",va="bottom",fontsize=7.0,fontname="Times New Roman")

    ax_d.set_xticks(x_d,common["period_label"])
    ax_d.set_xlim(-0.55,len(common)-0.45)
    ax_d.set_ylim(0.80,1.01)
    ax_d.set_ylabel("Rank Spearman ρ vs. primary",fontname="Times New Roman")
    style_axis(ax_d)
    title(ax_d,"d","Predictor-ranking concordance")

    shift_down=0.025
    for ax in [ax_c,ax_d]:
        pos=ax.get_position()
        ax.set_position([pos.x0,pos.y0-shift_down,pos.width,pos.height])

    for ax in axes.ravel():
        ax.xaxis.label.set_fontname("Times New Roman")
        ax.yaxis.label.set_fontname("Times New Roman")
        for text in ax.get_xticklabels()+ax.get_yticklabels():
            text.set_fontname("Times New Roman")

    stem=OUTDIR/"FigS4_persistent_forest_definition_and_common_sample_robustness"
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight",pad_inches=0.01)
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",pad_inches=0.01)
    fig.savefig(stem.with_suffix(".tiff"),dpi=600,bbox_inches="tight",pad_inches=0.01,pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(stem.with_suffix(".png"),dpi=450,bbox_inches="tight",pad_inches=0.01)
    plt.close(fig)

if __name__=="__main__":
    main()