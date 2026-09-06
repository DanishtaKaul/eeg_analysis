# -*- coding: utf-8 -*-
"""
One time-frequency image per effect and segment, with every significant
omnibus cluster outlined on the same panel. Supplies panel A of Figures 5-8.
"""

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "axes.linewidth": 1.1,
})

# ----------------------------- CONFIG -----------------------------
# Keep clusters with p <= this value.
ALPHA_REPORT = 0.05

# NPZ files to plot 
FILES = [
    Path(r"D:\cluster_outputs\light_main_PREP_last_prep1000perm_seed42.npz"),
    Path(r"D:\cluster_outputs\light_main_RESET_last_prep1000perm_seed42.npz"),
    Path(r"D:\cluster_outputs\obstacle_main_effect_prep_1000perm_seed42.npz"),
    Path(r"D:\cluster_outputs\obstacle_main_effect_reset_1000perm_seed42.npz"),
]

# Where the combined TFRs are written.
OUTPUT_DIR = Path(r"D:\Cluster_plots_last_prep\all_clusters_tfr")

# Colour-bar limits for the F background. Match the existing TFR plots.
FIX_CBAR = True
F_CBAR_TFR = (0.0, 14.0)

# Outline colour per cluster, cycled in the order clusters are drawn.
CLUSTER_COLORS = ["#000000", "#0D47A1", "#2E7D32",
                  "#6A1B9A", "#5D4037", "#00695C"]
# Also print the cluster number inside each outline (off: colour + legend only).
LABEL_CLUSTERS = False
# ------------------------------------------------------------------


def _unpack_masks(masks_any, n_feat):
    """
    Return a list of 1D boolean masks (length n_feat), one per cluster.

    Accepts the same storage formats as the other scripts:
      - object array of packed items (uint8 arrays or bytes/bytearray)
      - 2D packed matrix (n_clusters, n_bytes)
      - 2D already-unpacked matrix (n_clusters, n_feat)
      - list-like of masks
    """
    if masks_any is None:
        return []

    # ceil number of bytes for n_feat bits
    n_bytes_expected = (n_feat + 7) // 8

    # Case A: object array of packed items (the common case).
    arr_obj = np.asarray(masks_any, dtype=object)
    if arr_obj.dtype == object and arr_obj.ndim == 1:
        out = []
        for pb in arr_obj:
            if isinstance(pb, (bytes, bytearray)):
                buf = np.frombuffer(pb, dtype=np.uint8)
            else:
                buf = np.asarray(pb)
                if buf.dtype != np.uint8:
                    buf = buf.astype(np.uint8, copy=False)
            bits = np.unpackbits(buf)
            out.append(bits[:n_feat].astype(bool))
        return out

    arr = np.asarray(masks_any)

    # Case B: 2D packed matrix (n_clusters, n_bytes).
    if arr.ndim == 2 and arr.shape[1] == n_bytes_expected:
        return [np.unpackbits(np.asarray(row, dtype=np.uint8))[:n_feat].astype(bool)
                for row in arr]

    # Case C: 2D already-unpacked matrix (n_clusters, n_feat).
    if arr.ndim == 2 and arr.shape[1] == n_feat:
        return [arr[i].astype(bool, copy=False) for i in range(arr.shape[0])]

    # Case D: list-like fallback.
    try:
        return [np.asarray(m, dtype=bool).ravel()[:n_feat] for m in masks_any]
    except Exception as e:
        raise TypeError(
            f"Cannot parse cluster masks of type {type(masks_any)}") from e


def _rezero_times_if_prep(times, segment):
    """Show PREP ending at 0 (end-aligned); RESET starting at 0 (start-aligned)."""
    times = np.asarray(times).copy()
    seg = segment.decode() if isinstance(segment, (bytes,)) else str(segment)
    seg_up = seg.upper()
    if seg_up == "PREP":
        return times - times[-1]
    if seg_up == "RESET":
        return times - times[0]
    return times


def _effect_label_from_name(npz_path):
    """Read the effect name off the filename for the plot title."""
    stem = Path(npz_path).stem.lower()
    if "obstacle" in stem:
        return "Obstacle main effect"
    if "light" in stem:
        return "Light main effect"
    return "Main effect"


def plot_all_clusters_on_one_tfr(F3d, sig_masks3d, sig_labels,
                                 freqs, times_ms, out_base, title,
                                 label_clusters=False):
    """
    Draw one TFR with every significant cluster outlined in its own colour.

    F3d         : (n_ch, n_freq, n_time) F values.
    sig_masks3d : list of (n_ch, n_freq, n_time) boolean masks, one per cluster.
    sig_labels  : list of cluster numbers matching sig_masks3d.
    freqs       : (n_freq,) frequency axis in Hz.
    times_ms    : (n_time,) time axis in milliseconds (already re-zeroed).
    out_base    : output path without extension (".png"/".svg" are appended).
    label_clusters : also print the cluster number inside each outline.
    """
    # Background image: mean F across all channels, giving (n_freq, n_time).
    # The outlines below carry the cluster-specific result; this background is
    # only context behind them.
    Fmean = F3d.mean(axis=0)

    # Colour limits.
    if FIX_CBAR:
        vmin, vmax = F_CBAR_TFR
    else:
        vmin = 0.0
        vmax = float(np.nanmax(Fmean)) if np.isfinite(Fmean).any() else 1.0

    # Obstacle TFR drawn shorter than light so it does not dominate its block.
    fig_w = 7.5 if "Obstacle" in title else 6.5
    if "Obstacle" in title:
        fig_h = 4.5 if "RESET" in title else 3.0
    else:
        fig_h = 4.3

    _obs = "Obstacle" in title
    _obs_reset = _obs and "RESET" in title
    if _obs_reset:
        LAB_FS, TICK_FS, CB_FS, LEG_FS = 20, 17, 18, 17
    elif _obs:
        LAB_FS, TICK_FS, CB_FS, LEG_FS = 13, 11, 12, 11
    else:
        LAB_FS, TICK_FS, CB_FS, LEG_FS = 19, 16, 16, 18
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # TFR heatmap. origin='lower' so low frequencies sit at the bottom; extent
    # maps the image to real time (ms) and frequency (Hz) coordinates.
    im = ax.imshow(
        Fmean, aspect='auto', origin='lower',
        extent=[times_ms[0], times_ms[-1], freqs[0], freqs[-1]],
        interpolation='nearest', vmin=vmin, vmax=vmax, cmap='Reds'
    )

    # Outline each significant cluster in its own colour.
    legend_handles = []
    for i, (cl_num, mask3d) in enumerate(zip(sig_labels, sig_masks3d)):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        # Collapse channels: True at any (freq, time) point where this cluster
        # exists on at least one channel. This is the cluster's TFR footprint.
        mask2d = mask3d.any(axis=0)  # (n_freq, n_time)

        # Coloured outline around that footprint.
        ax.contour(times_ms, freqs, mask2d.astype(float),
                   levels=[0.5], linewidths=2.2, colors=[color])

        # contour() makes no legend entry, so add a proxy line for the legend.
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=3.5,
                   label=f"Cluster {cl_num}"))

        if label_clusters:
            # Put the cluster number near the middle of its footprint.
            f_idx, t_idx = np.where(mask2d)
            f_centre = float(freqs[int(round(f_idx.mean()))])
            t_centre = float(times_ms[int(round(t_idx.mean()))])
            ax.text(t_centre, f_centre, str(cl_num),
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.15',
                              fc='white', ec=color, lw=0.8))

    ax.set_xlabel("Time (ms)", labelpad=10, fontsize=LAB_FS)
    ax.set_ylabel("Frequency (Hz)", labelpad=10, fontsize=LAB_FS)
    ax.set_yticks([4, 8, 13, 18, 23, 28])
    ax.tick_params(labelsize=TICK_FS)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("F-Value", fontsize=CB_FS)
    cb.ax.tick_params(labelsize=TICK_FS)
    cb.set_ticks([0, 2, 4, 6, 8, 10, 12, 14])

    # Legend mapping outline colour to cluster, placed below the plot so it
    # never sits over the colorbar or the data.
    if "Obstacle" in title:
        leg = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.30),
            ncol=min(len(legend_handles), 4),
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.5,
            handlelength=2.0,
            fontsize=LEG_FS,
        )
    else:
        leg = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=min(len(legend_handles), 4),
            frameon=False,
            handlelength=2.0,
            fontsize=LEG_FS,
        )

    fig.savefig(out_base + ".png", dpi=200, bbox_inches="tight",
                bbox_extra_artists=(leg,))
    fig.savefig(out_base + ".svg", bbox_inches="tight",
                bbox_extra_artists=(leg,))
    plt.close(fig)


def run_one(npz_path, out_dir):
    """Load one NPZ and write its combined all-clusters TFR."""
    d = np.load(str(npz_path), allow_pickle=True)

    if "F_obs" not in d.files:
        raise ValueError(f"{npz_path} missing 'F_obs' (not an ANOVA NPZ).")

    # Grid sizes and axes.
    seg = d["segment"].item() if d["segment"].ndim == 0 else str(
        d["segment"][()])
    n_ch, n_f, n_t = int(d["n_ch"]), int(d["n_freqs"]), int(d["n_times"])
    freqs = np.asarray(d["freqs"], dtype=float)
    times = _rezero_times_if_prep(np.asarray(d["times"], dtype=float), seg)
    times_ms = times * 1000.0

    # F values reshaped to (channel, frequency, time).
    F1d = np.asarray(d["F_obs"], dtype=float).ravel()
    F3d = F1d.reshape(n_ch, n_f, n_t)

    # Cluster masks and their p-values.
    p_vals = np.asarray(d["p_vals"], dtype=float)
    n_feat = n_ch * n_f * n_t
    if "masks_packed" in d.files:
        masks = _unpack_masks(d["masks_packed"], n_feat)
    elif "clusters" in d.files:
        masks = _unpack_masks(d["clusters"], n_feat)
    else:
        masks = []

    # Keep only significant clusters, reshape each mask to 3D, and renumber
    # them by F mass so Cluster 1 is the largest.
    sig = []
    for m1d, pv in zip(masks, p_vals):
        if not np.isfinite(pv) or pv > ALPHA_REPORT:
            continue
        m1 = np.asarray(m1d, bool)
        m3d = m1.reshape(n_ch, n_f, n_t)
        if m3d.any():
            sig.append((m3d, float(F1d[m1].sum())))   # mask and its F mass

    sig.sort(key=lambda s: -s[1])                      # biggest mass first
    sig_masks3d = [s[0] for s in sig]
    sig_labels = list(range(1, len(sig) + 1))          # Cluster 1 = biggest

    if not sig_masks3d:
        print(
            f"[INFO] {npz_path.name}: no significant clusters; nothing drawn.")
        return

    label = _effect_label_from_name(npz_path)
    seg_up = str(seg).upper()
    title = f"{label} \u2014 {seg_up}: all significant clusters"
    out_base = str(out_dir / f"{npz_path.stem}__{seg_up}__ALL_CLUSTERS_TFR")

    plot_all_clusters_on_one_tfr(
        F3d, sig_masks3d, sig_labels, freqs, times_ms,
        out_base, title, label_clusters=LABEL_CLUSTERS
    )
    print(f"[DONE] {npz_path.name}: outlined {len(sig_masks3d)} clusters "
          f"({sig_labels}) -> {out_base}.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not f.is_file():
            print(f"[WARN] NPZ not found, skipping: {f}")
            continue
        run_one(f, OUTPUT_DIR)
    print("All done.")


if __name__ == "__main__":
    main()
