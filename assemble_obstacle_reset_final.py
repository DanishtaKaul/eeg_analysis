

"""
Figure 8: the obstacle main effect, reset phase.

TFR/topo block on top (panels A-B), one line plot (panel C) below, redrawn from
its saved *_inputs.pkl. The block comes from assemble_composites_obstacle.py.

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

# The TFR/topo block PNG (A = TFR, B = topo), from assemble_composites_obstacle.py.
BLOCK_PNG = r"D:\Cluster_plots_last_prep\composites\obstacle_RESET_fig1_TFR_topo.png"

OUT_PNG = r"D:\Cluster_plots_last_prep\composites\obstacle_RESET_final_merged.png"
OUT_SVG = r"D:\Cluster_plots_last_prep\composites\obstacle_RESET_final_merged.svg"

# =========================================================================== #
# SECTION 2 -- pkl paths 
# =========================================================================== #

CLUSTERS = [
    ("Cluster 1", "C", [
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_present_vs_absent\RESET\cluster00_Present_vs_Absent_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_UA_vs_EA\RESET\cluster00_UA_vs_EA_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_UP_vs_ALL_ABSENT\RESET\cluster00_UP_vs_ALL_ABSENT_inputs.pkl",
        
        r"D:\Cluster_plots_last_prep\line_plots\obstacle_EP_vs_ALL_ABSENT\RESET\cluster00_EP_vs_ALL_ABSENT_inputs.pkl",
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

# RESET common y-range from the four pkls 
Y_LIM = (-57.25, 115.21)

LINE_COL = {
    "Present": "#5E2CA5", "Unexpected Present": "#1CA3C7",
    "Expected Present": "#E23B5A", "Absent": "#E6A100",
    "Unexpected Absent": "#9ACD32", "Expected Absent": "#6D4C41",
}
BOX_COL = {
    "Present vs Absent": "#000000", "UP vs ALL_ABSENT": "#F07E10",
    "EP vs ALL_ABSENT": "#00A86B", "UA vs EA": "#1F6FEB",
}
BOX_LABEL = {
    "Present vs Absent": "Present vs Absent",
    "UP vs ALL_ABSENT": "Unexpected Present vs Absent",
    "EP vs ALL_ABSENT": "Expected Present vs Absent",
    "UA vs EA": "Unexpected Absent vs Expected Absent",
}
LINE_ORDER = ["Present", "Unexpected Present", "Expected Present", "Absent",
              "Unexpected Absent", "Expected Absent"]

# Layout 
FIG_W = 15.0
LINE_AREA_H = 6.5
GRID_LEFT, GRID_RIGHT = 0.075, 0.99
GRID_TOP, GRID_BOTTOM = 0.995, 0.08


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

    ax.set_xlim(0, times.max())    # RESET: start-aligned, 0 -> +1000 ms
    ax.set_ylim(*Y_LIM)

    handles = [Line2D([0], [0], color=LINE_COL[l], lw=3.4, label=l)
               for l in present]
    handles += [Line2D([0], [0], color=BOX_COL[k], ls=LEG_DASH, lw=3.0,
                       label=BOX_LABEL[k]) for k in box_keys]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handlelength=2.6, fontsize=15.5, labelspacing=0.35)
    
    ax.text(-0.012, 1.03, letter, transform=ax.transAxes, fontsize=25,
            fontweight="bold", va="bottom", ha="right")


def main():
    # ---- print every box window ----
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
    ax_block.imshow(np.asarray(block), aspect="equal")
    ax_block.axis("off")

    # single line plot (one cluster)
    title, letter, paths = CLUSTERS[0]
    dicts, lines, times = _cluster_data(paths)
    ax = fig.add_subplot(outer[1, 0])
    _draw_cluster(ax, title, letter, dicts, lines, times)

    fig.canvas.draw()
    _p = ax.get_position()
    fig.text((_p.x0 + _p.x1) / 2, 0.015, "Time (ms)",
             ha="center", fontsize=22)
    fig.text(0.005, (_p.y0 + _p.y1) / 2,
             "Power (% change from baseline)",
             va="center", rotation=90, fontsize=22)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
