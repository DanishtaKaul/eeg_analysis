

"""
Figure 6: the obstacle main effect, preparation phase.

TFR/topo block on top (panels A-E), one line plot per cluster (2x2, panels F-I)
below,redrawn from their saved *_inputs.pkl. The block comes from assemble_composites_obstacle.py.

Cluster display numbering (F-mass order): cluster02 -> Cluster 1,
cluster03 -> Cluster 2, cluster07 -> Cluster 3, cluster01 -> Cluster 4.
"""

import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

mpl.use("Agg")

# =========================================================================== #
# SECTION 1 -- block image + outputs  
# =========================================================================== #

# The TFR/topo block PNG (made by assemble_composites_obstacle.py,).
BLOCK_PNG = r"D:\Cluster_plots_last_prep\composites\obstacle_PREP_fig1_TFR_topo.png"

# Where to save the finished merged figure.
OUT_PNG = r"D:\Cluster_plots_last_prep\composites\obstacle_PREP_final_merged.png"
OUT_SVG = r"D:\Cluster_plots_last_prep\composites\obstacle_PREP_final_merged.svg"

# =========================================================================== #
# SECTION 2 -- pkl paths  
# =========================================================================== #

CLUSTERS = [
    ("Cluster 1", "F", [
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_present_vs_absent\PREP\cluster02_Present_vs_Absent_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_UP_vs_ALL_ABSENT\PREP\cluster02_UP_vs_ALL_ABSENT_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_EP_vs_ALL_ABSENT\PREP\cluster02_EP_vs_ALL_ABSENT_inputs.pkl",
    ]),
    ("Cluster 2", "G", [
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_present_vs_absent\PREP\cluster03_Present_vs_Absent_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_UP_vs_ALL_ABSENT\PREP\cluster03_UP_vs_ALL_ABSENT_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_EP_vs_ALL_ABSENT\PREP\cluster03_EP_vs_ALL_ABSENT_inputs.pkl",
    ]),
    ("Cluster 3", "H", [
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_UP_vs_EP\PREP\cluster07_UP_vs_EP_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_EP_vs_ALL_ABSENT\PREP\cluster07_EP_vs_ALL_ABSENT_inputs.pkl",
    ]),
    ("Cluster 4", "I", [
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_present_vs_absent\PREP\cluster01_Present_vs_Absent_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_UP_vs_ALL_ABSENT\PREP\cluster01_UP_vs_ALL_ABSENT_inputs.pkl",
    ]),
]



mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 20, "axes.labelsize": 22,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 16,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 6, "ytick.major.size": 6, "axes.linewidth": 1.1,
})

# locked style
LINE_W = 1.3
BAND_ALPHA = 0.18
BOX_LS = (0, (4, 1.2))
BOX_LW_WIDE = 1.8
BOX_LW_NARROW = 1.5
NARROW_MS = 20
BOX_PAD_FRAC = 0.08
LEG_DASH = (0, (4, 1.2))
Y_LIM = (-63.39, 26.73)   

LINE_COL = {
    "Present": "#5E2CA5", "Unexpected Present": "#1CA3C7",
    "Expected Present": "#E23B5A", "Absent": "#E6A100",
}
BOX_COL = {
    "Present vs Absent": "#000000", "UP vs ALL_ABSENT": "#F07E10",
    "EP vs ALL_ABSENT": "#00A86B", "UP vs EP": "#26418F",
}
BOX_LABEL = {
    "Present vs Absent": "Present vs Absent",
    "UP vs ALL_ABSENT": "Unexpected Present vs Absent",
    "EP vs ALL_ABSENT": "Expected Present vs Absent",
    "UP vs EP": "Unexpected Present vs Expected Present",
}
LINE_ORDER = ["Present", "Unexpected Present", "Expected Present", "Absent"]

FIG_W = 15.0
LINE_AREA_H = 11.5
GRID_LEFT, GRID_RIGHT = 0.075, 0.99
GRID_TOP, GRID_BOTTOM = 0.995, 0.05


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _cluster_data(pkl_paths):
    """Load a cluster's pkls; return (list of dicts, lines dict, times)."""
    dicts = [_load(p) for p in pkl_paths]
    lines = {}
    absent_src = None
    times = None
    for d in dicts:
        times = np.asarray(d["times"]) * 1000.0
        if d["labelA"] not in lines:
            lines[d["labelA"]] = (np.asarray(d["yA"]), np.asarray(d["ciA"]))
        if d["labelB"] == "Absent":
            if "Present vs Absent" in d["contrast"]:
                absent_src = (np.asarray(d["yB"]), np.asarray(d["ciB"]))
            if "Absent" not in lines:
                lines["Absent"] = (np.asarray(d["yB"]), np.asarray(d["ciB"]))
        else:
            if d["labelB"] not in lines:
                lines[d["labelB"]] = (np.asarray(
                    d["yB"]), np.asarray(d["ciB"]))
    if absent_src is not None:
        lines["Absent"] = absent_src
    return dicts, lines, times


def _draw_cluster(ax, title, letter, dicts, lines, times):
    present = [l for l in LINE_ORDER if l in lines]
    for lab in present:
        y, ci = lines[lab]
        c = LINE_COL[lab]
        ax.fill_between(times, y - ci, y + ci, color=c,
                        alpha=BAND_ALPHA, lw=0, zorder=1.5)
        ax.plot(times, y, color=c, lw=LINE_W, zorder=2)
    ax.axvline(0, ls="--", lw=1, color="0.4", zorder=2)

    def pair(d):
        y_a, ci_a = np.asarray(d["yA"]), np.asarray(d["ciA"])
        if d["labelB"] == "Absent":
            y_b, ci_b = lines["Absent"]
        else:
            y_b, ci_b = np.asarray(d["yB"]), np.asarray(d["ciB"])
        return y_a, ci_a, y_b, ci_b

    box_keys = []
    for d in dicts:
        key = d["contrast"]
        y_a, ci_a, y_b, ci_b = pair(d)
        drew = False
        for w in d["sig_windows"]:
            t0 = float(w["time_start"]) * 1000.0
            t1 = float(w["time_end"]) * 1000.0
            mask = (times >= t0) & (times <= t1)
            if not np.any(mask):
                continue
            drew = True
            lo = min(np.nanmin((y_a - ci_a)[mask]),
                     np.nanmin((y_b - ci_b)[mask]))
            hi = max(np.nanmax((y_a + ci_a)[mask]),
                     np.nanmax((y_b + ci_b)[mask]))
            pad = (hi - lo) * BOX_PAD_FRAC
            yb, yt = lo - pad, hi + pad
            lw = BOX_LW_NARROW if (t1 - t0) <= NARROW_MS else BOX_LW_WIDE
            for ex, ey in (([t0, t1], [yb, yb]), ([t0, t1], [yt, yt]),
                           ([t0, t0], [yb, yt]), ([t1, t1], [yb, yt])):
                ax.plot(ex, ey, color=BOX_COL[key], ls=BOX_LS, lw=lw,
                        alpha=1.0, zorder=2.8)
        if drew:
            box_keys.append(key)

    ax.set_xlim(times.min(), 0)
    ax.set_ylim(*Y_LIM)

    handles = [Line2D([0], [0], color=LINE_COL[l], lw=3.4, label=l)
               for l in present]
    handles += [Line2D([0], [0], color=BOX_COL[k], ls=LEG_DASH, lw=3.0,
                       label=BOX_LABEL[k]) for k in box_keys]
    ax.legend(handles=handles, loc="upper left", frameon=False,
              handlelength=2.6, fontsize=15.5, labelspacing=0.35)
    ax.set_title(title, fontsize=17)
    ax.text(-0.012, 1.06, letter, transform=ax.transAxes, fontsize=26,
            fontweight="bold", va="bottom", ha="right")


def main():
    # ---- print every box window  ----
    print("=" * 60)
    print("BOX WINDOWS THAT WILL BE DRAWN (check against manuscript):")
    print("=" * 60)
    for title, _letter, paths in CLUSTERS:
        print(f"\n{title}")
        for d in (_load(p) for p in paths):
            if len(d["sig_windows"]) == 0:
                print(f"   {d['contrast']:18s}: (no windows -> not drawn)")
            for w in d["sig_windows"]:
                t0 = round(w["time_start"] * 1000)
                t1 = round(w["time_end"] * 1000)
                print(f"   {d['contrast']:18s}: {t0} to {t1} ms")
    print("=" * 60 + "\n")

    block = Image.open(BLOCK_PNG).convert("RGB")
    bw, bh = block.size
    block_h_in = FIG_W / (bw / bh)
    fig_h = block_h_in + LINE_AREA_H + 0.9

    fig = plt.figure(figsize=(FIG_W, fig_h))
    outer = GridSpec(2, 1, figure=fig,
                     height_ratios=[block_h_in, LINE_AREA_H], hspace=0.05,
                     left=GRID_LEFT, right=GRID_RIGHT,
                     top=GRID_TOP, bottom=GRID_BOTTOM)
    ax_block = fig.add_subplot(outer[0, 0])
    ax_block.imshow(np.asarray(block))
    ax_block.axis("off")

    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1, 0],
                                    hspace=0.32, wspace=0.10)
    for (title, letter, paths), (r, c) in zip(
            CLUSTERS, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        dicts, lines, times = _cluster_data(paths)
        ax = fig.add_subplot(inner[r, c])
        _draw_cluster(ax, title, letter, dicts, lines, times)

    # Centre the shared labels on the actual 2x2 line-plot block.
    fig.canvas.draw()
    boxes = [fig.axes[i].get_position() for i in range(len(fig.axes))]
    # the four line-plot axes are the last four added (after the block + colourbars)
    lp = boxes[-4:]
    x_mid = (min(b.x0 for b in lp) + max(b.x1 for b in lp)) / 2
    y_mid = (min(b.y0 for b in lp) + max(b.y1 for b in lp)) / 2
    fig.text(x_mid, 0.010, "Time (ms)", ha="center", fontsize=22)
    fig.text(0.005, y_mid, "Power (% change from baseline)",
             va="center", rotation=90, fontsize=22)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
