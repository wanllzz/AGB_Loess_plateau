"""Plot Fig. S2: sensitivity of forest-state classes to forest-fraction thresholds."""
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

DATA = TEMPORAL_DATA_DIR / "agb_forest_change_threshold_sensitivity.csv"
OUTDIR = FIGURE_OUTPUT_DIR / "supplementary"

CLASS_ORDER = ["persistent forest","forest expansion","forest contraction","sparse forest or other"]
CLASS_LABELS = {"persistent forest":"Persistent forest","forest expansion":"Forest expansion","forest contraction":"Forest contraction","sparse forest or other":"Sparse forest or other"}
COLORS = {"persistent forest":"#5B8E7D","forest expansion":"#82A9C4","forest contraction":"#C98572","sparse forest or other":"#D6D8D3"}

def configure_style():
    mpl.rcParams.update({
        "font.family":"Times New Roman",
        "font.size":8.2,
        "axes.labelsize":9.0,
        "xtick.labelsize":8.4,
        "ytick.labelsize":8.4,
        "legend.fontsize":8.0,
        "axes.linewidth":0.8,
        "xtick.major.width":0.7,
        "ytick.major.width":0.7,
        "xtick.major.size":3.0,
        "ytick.major.size":3.0,
        "svg.fonttype":"none",
        "pdf.fonttype":42,
        "ps.fonttype":42,
        "axes.unicode_minus":False,
        "figure.facecolor":"white",
        "axes.facecolor":"white",
        "savefig.facecolor":"white"
    })

def main():
    configure_style()
    OUTDIR.mkdir(parents=True,exist_ok=True)
    data = pd.read_csv(DATA)
    data = data.loc[data["zone"].eq("Loess Plateau")].copy()
    data["threshold_pct"] = (data["forest_fraction_threshold"] * 100).round().astype(int)

    plot_data = data.pivot(index="threshold_pct",columns="forest_change_class",values="percentage_of_zone").reindex(columns=CLASS_ORDER).sort_index()
    if not np.allclose(plot_data.sum(axis=1).to_numpy(),100.0,atol=0.05):
        raise ValueError("Threshold-class proportions do not sum to 100%.")
    plot_data.to_csv(OUTDIR / "source_data_FigS2_threshold_sensitivity.csv",index=True)

    fig, ax = plt.subplots(figsize=(89/25.4,63/25.4))
    fig.subplots_adjust(left=0.17,right=0.98,top=0.95,bottom=0.42)

    x = np.arange(len(plot_data))
    bottom = np.zeros(len(plot_data))
    bar_width = 0.34

    for forest_class in CLASS_ORDER:
        values = plot_data[forest_class].to_numpy()
        bars = ax.bar(x,values,bottom=bottom,width=bar_width,color=COLORS[forest_class],edgecolor="white",linewidth=0.7,label=CLASS_LABELS[forest_class])
        for bar, value, base in zip(bars, values, bottom):
            if value >= 7:
                txt_color = "#1F2933" if forest_class != "forest expansion" else "white"
                ax.text(bar.get_x()+bar.get_width()/2,base+value/2,f"{value:.1f}",ha="center",va="center",fontsize=7.3,color=txt_color,fontname="Times New Roman")
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([f"{threshold}%" for threshold in plot_data.index],fontname="Times New Roman")
    ax.set_xlabel("Forest-fraction threshold",labelpad=4,fontname="Times New Roman")
    ax.set_ylabel("Area proportion (%)",labelpad=4,fontname="Times New Roman")
    ax.set_ylim(0,100)
    ax.set_yticks(np.arange(0,101,20))
    ax.grid(axis="y",color="#D9DDDF",linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    leg = ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(0.5,-0.27),frameon=False,columnspacing=1.2,handlelength=1.3,handletextpad=0.45,labelspacing=0.55,borderaxespad=0.0)
    for text in leg.get_texts():
        text.set_fontname("Times New Roman")
    for tick in ax.get_xticklabels()+ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    stem = OUTDIR / "FigS2_forest_state_threshold_sensitivity"
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".tiff": {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}},
        ".png": {"dpi": 450}
    }.items():
        fig.savefig(stem.with_suffix(suffix),bbox_inches="tight",**kwargs)

    plt.close(fig)

if __name__ == "__main__":
    main()