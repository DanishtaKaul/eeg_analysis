
# -*- coding: utf-8 -*-
"""
Figure 5: the light main effect, preparation phase.

Places the assembled TFR/topo block (left, panels A-C) beside the two post-hoc
line plots (right, panels D-E), redrawn from their saved *_inputs.pkl.

The block comes from assemble_composites.py; nothing on the left is
regenerated here. Only the line plots are drawn
"""

from pathlib import Path
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from PIL import Image

# =============================================================================
# PATHS  
# =============================================================================
# The assembled TFR/topo block from assemble_composites.py.
TFR_TOPO_BLOCK = Path(
    r"D:\Cluster_plots_last_prep\composites\light_PREP_fig1_TFR_topo.png")


# The two line-plot pkls, display order (Cluster 1 first).

PKL_CLUSTER1 = Path(
    r"D:\Cluster_plots_last_prep\line_plots\light_post_hoc_PREP\cluster0_ALL_THREE_inputs.pkl")
PKL_CLUSTER2 = Path(
    r"D:\Cluster_plots_last_prep\line_plots\light_post_hoc_PREP\cluster1_ALL_THREE_inputs.pkl")


OUT_PATH = Path(r"D:\Cluster_plots_last_prep\composites\light_PREP_FINAL.png")


# =============================================================================
# LAYOUT + STYLE  -- locked to match the TFR/topo block.
# =============================================================================
FIG_H_IN = 8.5           # whole-figure height
DPI = 300

RIGHT_WIDTH_MULT = 1.05  # right column width relative to the left block width
GUTTER_IN = 2.0          # extra horizontal space between the block and line plots
WSPACE_OUTER = 0.28      # gap between the block and the line-plot column
HSPACE_RIGHT = 0.42      # gap between the two stacked line plots
# shared y-label distance left of the plots (figure frac)
YLABEL_X_OFFSET = 0.05

# Panel letters (D, E) on the line plots — placed outside the top-left corner
# to match the A/B/C letters on the block.
PANEL_LETTER_X = -0.085  # axes-fraction x (negative = left of the plot)
PANEL_LETTER_Y = 1.02    # axes-fraction y (just above the plot)
PANEL_LETTER_SIZE = 18

# Fonts + tick marks, identical to the TFR/topo scripts.
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "axes.linewidth": 1.1,
})

# Condition line colours (Bright, Ambient, Dark) and contrast-box styling.
LINE_COLORS = ["#7758A3", "#D4763C", "#4D87A8"]
CONTRAST_COLORS = {
    "LIGHT_vs_AMBIENT": "#D11A2A",
    "LIGHT_vs_DARK":    "#6B3F12",
    "AMBIENT_vs_DARK":  "#1B8A3A",
}
CONTRAST_LINE_IDX = {
    "LIGHT_vs_AMBIENT": [0, 1],
    "LIGHT_vs_DARK":    [0, 2],
    "AMBIENT_vs_DARK":  [1, 2],
}
BOX_LS = (0, (4, 1.2))
LINE_W = 1.6             # plotted condition lines
BOX_W = 2.0              # plotted contrast boxes
LEG_SOLID_W = 3.5        # solid condition samples in the legend
# dashed contrast samples in the legend (matches BOX_W)
LEG_DASH_W = 2.6
# =============================================================================


def _pretty(label):
    parts = label.split("_vs_")
    left = parts[0].replace("_", " ").title()
    out = f"{left} vs {parts[1].replace('_', ' ').title()}" if len(
        parts) == 2 else left
    return out.replace("Light", "Bright")


def common_ylim(pkls):
    """Shared y-range across the line-plot pkls, padded 5% each side."""
    g_min, g_max = np.inf, -np.inf
    for p in pkls:
        with open(p, "rb") as fh:
            d = pickle.load(fh)
        m = np.asarray(d["means3"])
        if d["ci95_3"] is None:
            lo, hi = np.nanmin(m), np.nanmax(m)
        else:
            ci = np.asarray(d["ci95_3"])
            lo, hi = np.nanmin(m - ci), np.nanmax(m + ci)
        g_min, g_max = min(g_min, lo), max(g_max, hi)
    rng = g_max - g_min
    return g_min - 0.05 * rng, g_max + 0.05 * rng


def draw_lineplot(ax, pkl_path, ylim, show_xlabel, panel_letter=None, title=None):
    """Three condition lines + CI bands + one dashed box per significant window.
    Each plot carries its own legend (conditions + only its own contrasts).
    panel_letter, if given, is drawn outside the top-left corner to match the
    A/B/C letters on the TFR/topo block."""
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    times = np.asarray(d["times"])
    means3 = np.asarray(d["means3"])
    ci3 = None if d["ci95_3"] is None else np.asarray(d["ci95_3"])
    labels3 = d["labels3"]
    swm = d["sig_windows_map"]

    for i, lab in enumerate(labels3):
        ax.plot(times, means3[i], color=LINE_COLORS[i], lw=LINE_W, zorder=2)
        if ci3 is not None:
            ax.fill_between(times, means3[i] - ci3[i], means3[i] + ci3[i],
                            color=LINE_COLORS[i], alpha=0.18, lw=0, zorder=1.5)

    ax.axvline(0, ls="--", lw=1, zorder=2)
    ax.set_xlim(*d["xlim"])
    if ylim is not None:
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
    if show_xlabel:
        ax.set_xlabel("Time (ms)", labelpad=12)

    for key in ("LIGHT_vs_DARK", "AMBIENT_vs_DARK", "LIGHT_vs_AMBIENT"):
        color = CONTRAST_COLORS[key]
        for win in swm.get(key, []):
            t0, t1 = float(win[0]), float(win[1])
            mask = (times >= t0) & (times <= t1)
            if not np.any(mask):
                continue
            sel = CONTRAST_LINE_IDX[key]
            if ci3 is not None:
                lo = np.nanmin((means3 - ci3)[sel][:, mask])
                hi = np.nanmax((means3 + ci3)[sel][:, mask])
            else:
                lo = np.nanmin(means3[sel][:, mask])
                hi = np.nanmax(means3[sel][:, mask])
            pad = (hi - lo) * 0.02
            y0, h = lo - pad, (hi - lo) + 2 * pad
            xl, xr, yb, yt = t0, t1, y0, y0 + h
            for ex, ey in (([xl, xr], [yb, yb]), ([xl, xr], [yt, yt]),
                           ([xl, xl], [yb, yt]), ([xr, xr], [yb, yt])):
                ax.plot(ex, ey, color=color, ls=BOX_LS, lw=BOX_W, alpha=0.95,
                        zorder=2.8, clip_on=False, dash_capstyle="butt")

    # per-plot legend: conditions (thick solid) + only this plot's contrasts
    cond = [Line2D([0], [0], color=LINE_COLORS[i], lw=LEG_SOLID_W,
                   label=_pretty(lab)) for i, lab in enumerate(labels3)]
    leg1 = ax.legend(handles=cond, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     borderaxespad=0., frameon=False, handlelength=2.0)
    ax.add_artist(leg1)
    present = [k for k in ("LIGHT_vs_DARK", "AMBIENT_vs_DARK", "LIGHT_vs_AMBIENT")
               if len(swm.get(k, []))]
    box = [Line2D([0], [0], color=CONTRAST_COLORS[k], ls=BOX_LS, lw=LEG_DASH_W,
                  label=_pretty(k)) for k in present]
    ax.legend(handles=box, loc="upper left", bbox_to_anchor=(1.02, 0.52),
              borderaxespad=0., frameon=False, handlelength=2.0)

    # panel letter outside the top-left corner, matching the A/B/C letters on
    # the block 
    if title is not None:
        ax.set_title(title, fontsize=15)
    if panel_letter is not None:
        ax.text(PANEL_LETTER_X, PANEL_LETTER_Y, panel_letter,
                transform=ax.transAxes, fontsize=PANEL_LETTER_SIZE,
                fontweight="bold", va="bottom", ha="left")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    block = Image.open(TFR_TOPO_BLOCK).convert("RGB")
    bw, bh = block.size
    block_aspect = bw / bh

    ylim = common_ylim([PKL_CLUSTER1, PKL_CLUSTER2])

    left_col_w = FIG_H_IN * block_aspect
    right_col_w = left_col_w * RIGHT_WIDTH_MULT
    fig_w = left_col_w + right_col_w + GUTTER_IN

    fig = plt.figure(figsize=(fig_w, FIG_H_IN))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[left_col_w, right_col_w],
                     wspace=WSPACE_OUTER, left=0.02, right=0.82)

    ax_block = fig.add_subplot(outer[0, 0])
    ax_block.imshow(np.asarray(block))
    ax_block.axis("off")

    right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1],
                                    height_ratios=[1, 1], hspace=HSPACE_RIGHT)
    ax_top = fig.add_subplot(right[0, 0])
    ax_bot = fig.add_subplot(right[1, 0])
    draw_lineplot(ax_top, PKL_CLUSTER1, ylim,
                  show_xlabel=False, panel_letter="D", title="Cluster 1")
    draw_lineplot(ax_bot, PKL_CLUSTER2, ylim,
                  show_xlabel=True, panel_letter="E", title="Cluster 2")

    # one shared y-label centred on the two line-plot boxes
    fig.canvas.draw()
    pt = ax_top.get_position()
    pb = ax_bot.get_position()
    fig.text(pb.x0 - YLABEL_X_OFFSET, (pt.y1 + pb.y0) / 2,
             "Power (% change from baseline)", rotation=90,
             va="center", ha="center", fontsize=19)

    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PATH.with_suffix(".svg"),
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[SAVED] {OUT_PATH}")


if __name__ == "__main__":
    main()
