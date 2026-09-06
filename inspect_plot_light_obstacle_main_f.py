# -*- coding: utf-8 -*-
"""
Inspect and plot significant clusters for the light and obstacle main-effect
NPZs (F-tests).

- Loads each NPZ, rebuilds cluster masks, keeps clusters with p <= ALPHA_REPORT
- For each significant cluster:
    * Topomap of mean F over the cluster window, with significant electrodes marked
    * Topomap at the peak F voxel (diagnostic only)
    * TFR heatmap with a significance contour
- Writes summary.txt and meta.json
Supplies the topomaps for Figures 5-8.
"""

import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

import matplotlib as mpl

from matplotlib.collections import PathCollection

# Set Arial and fixed text sizes for every figure this script makes,
# so the topomaps match the TFRs.
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

# -------- Config --------
ALPHA_REPORT = 0.05
SENSOR_DOT_SIZE = 12   # non-significant sensor marker size
FILES = [
    Path(r"D:\cluster_outputs\light_main_PREP_last_prep1000perm_seed42.npz"),
    Path(r"D:\cluster_outputs\light_main_RESET_last_prep1000perm_seed42.npz"),
    # OBSTACLE main-effect NPZs
    Path(r"D:\cluster_outputs\obstacle_main_effect_prep_1000perm_seed42.npz"),
    Path(r"D:\cluster_outputs\obstacle_main_effect_reset_1000perm_seed42.npz"),
]
OUTPUT_ROOT = Path(r"D:\Cluster_plots_last_prep")
# ------------------------

# ----- Fixed colorbar limits for F-plots -----
FIX_CBAR = True            # set False to go back to auto-scaling
F_CBAR_TFR = (0.0, 14.0)  # used by TFR heatmaps
F_CBAR_TOPO = (0.0, 10.0)  # used by topomaps (mean + peak)


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _unpack_masks(masks_any, n_feat):
    """
    Return list of 1D boolean masks (length n_feat), one per cluster.

    Accepts:
      - object array of packed items (uint8 arrays or bytes/bytearray), len == n_clusters
      - 2D packed matrix shape: (n_clusters, n_bytes) where n_bytes*8 >= n_feat
      - 2D already-unpacked boolean/numeric matrix: (n_clusters, n_feat)
      - list-like of masks
    """
    if masks_any is None:
        return []

    n_bytes_expected = (n_feat + 7) // 8  # ceil bytes needed

    # Case A: object array of packed items (common)
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

    # Normalize to ndarray for remaining cases
    arr = np.asarray(masks_any)

    # Case B: 2D **packed** matrix: (n_clusters, n_bytes)
    if arr.ndim == 2 and arr.shape[1] == n_bytes_expected:
        return [
            np.unpackbits(np.asarray(row, dtype=np.uint8))[
                :n_feat].astype(bool)
            for row in arr
        ]

    # Case C: 2D already-unpacked matrix: (n_clusters, n_feat)
    if arr.ndim == 2 and arr.shape[1] == n_feat:
        return [arr[i].astype(bool, copy=False) for i in range(arr.shape[0])]

    # Case D: list-like fallback
    try:
        return [np.asarray(m, dtype=bool).ravel()[:n_feat] for m in masks_any]
    except Exception as e:
        raise TypeError(
            f"Cannot parse cluster masks of type {type(masks_any)} with shape {getattr(masks_any, 'shape', None)}"
        ) from e


def _reshape_1d_to_3d(x1d, n_ch, n_f, n_t):
    return x1d.reshape(n_ch, n_f, n_t)


def _peak_index_from_mask(F1d, mask1d):
    # F is nonnegative; pick max F inside the cluster
    return int(np.argmax(np.where(mask1d, F1d, -np.inf)))


def _find_cluster_bounds(mask3d):
    """Return (ch_inds, f_min_idx, f_max_idx, t_min_idx, t_max_idx)."""
    ch_any = mask3d.any(axis=(1, 2))
    ch_inds = np.where(ch_any)[0].tolist()
    f_inds = np.where(mask3d.any(axis=(0, 2)))[0]
    t_inds = np.where(mask3d.any(axis=(0, 1)))[0]
    return ch_inds, int(f_inds.min()), int(f_inds.max()), int(t_inds.min()), int(t_inds.max())


def _plot_topomap_at_peak(F3d, mask3d, ch_names, info_like_epochs, freq_idx, time_idx, out_base):
    """Topomap at (freq_idx, time_idx); ring cluster channels active at that point. Saves PNG and SVG."""
    data = F3d[:, freq_idx, time_idx]
    in_cluster_now = mask3d[:, freq_idx, time_idx]
    ringed = [ch_names[i] for i, v in enumerate(in_cluster_now) if v]

    epochs = mne.read_epochs(info_like_epochs, preload=False)
    info = epochs.info

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im, _ = mne.viz.plot_topomap(
        data, info, axes=ax, show=False, outlines="head", sensors=True
    )
    ax.set_title(f"Topomap @ (f_idx={freq_idx}, t_idx={time_idx})")

    # Ring the cluster channels 
    picks = np.arange(len(info['ch_names']))
    try:
        pos = mne.channels.layout._auto_topomap_coords(
            info, picks=picks, to_sphere=True, sphere=None, ignore_overlap=True
        )
    except TypeError:
        # Older MNE signature
        pos = mne.channels.layout._auto_topomap_coords(
            info, picks=picks, ignore_overlap=True)

    name_to_idx = {name: i for i, name in enumerate(info['ch_names'])}
    for ch in ringed:
        i = name_to_idx.get(ch)
        if i is not None:
            ax.plot(pos[i, 0], pos[i, 1],
                    marker='o', markersize=12, fillstyle='none', linewidth=2)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=200)
    fig.savefig(out_base + ".svg")
    plt.close(fig)
    return ringed


def _plot_topomap_cluster_agg(
    F3d, mask3d, ch_names, sample_epochs_path, out_base,
    title, reduce="mean", scale_from="sig", cmap="Reds", debug=False,
):
    # 1) common TF window (union across channels)
    tf_window = mask3d.any(axis=0)

    # 2) aggregate per channel over the SAME window
    n_ch = F3d.shape[0]
    vals = np.full(n_ch, np.nan, float)
    for ch in range(n_ch):
        x = F3d[ch][tf_window]
        if x.size:
            vals[ch] = x.mean() if reduce == "mean" else x.sum()

    # which channels are ever significant in the cluster
    sig_ch = mask3d.any(axis=(1, 2))

    # align to epochs-info order
    epochs = mne.read_epochs(sample_epochs_path, preload=False)
    info = epochs.info
    if list(info['ch_names']) != list(ch_names):
        name_to_idx = {n: i for i, n in enumerate(ch_names)}
        order = [name_to_idx[n] for n in info['ch_names']]
        vals = vals[order]
        sig_ch = sig_ch[order]

    # color limits
    if FIX_CBAR:
        vmin, vmax = F_CBAR_TOPO
    else:
        src = vals[sig_ch] if sig_ch.any() else vals[np.isfinite(vals)]
        vmin = 0.0
        vmax = float(np.nanmax(src)) if src.size else 1.0

    # plot
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    try:
        im, _ = mne.viz.plot_topomap(
            vals, info, axes=ax, show=False, outlines="head", sensors=True, extrapolate="head",
            mask=sig_ch,
            mask_params=dict(marker='o', markerfacecolor='black',
                             markeredgecolor='black', markersize=6,
                             linewidth=0, clip_on=False),
            vmin=vmin, vmax=vmax, cmap=cmap
        )
    except TypeError:
        im, _ = mne.viz.plot_topomap(
            vals, info, axes=ax, show=False, outlines="head", sensors=True, extrapolate="head",
            mask=sig_ch,
            mask_params=dict(marker='o', markerfacecolor='black',
                             markeredgecolor='black', markersize=6,
                             linewidth=0, clip_on=False),
        )
        try:
            im.set_clim(vmin=vmin, vmax=vmax)
            im.set_cmap(cmap)
        except Exception:
            pass
        # Enlarge non-significant sensor markers
    for _coll in ax.collections:
        if isinstance(_coll, PathCollection):
            _coll.set_sizes([SENSOR_DOT_SIZE])
            _coll.set_clip_on(False)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _is_light = "light" in out_base.lower()
    if _is_light:
        cb.set_label("F-Value", fontsize=13)
        cb.ax.tick_params(labelsize=11)
    else:
        cb.set_label("F-Value")

    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=200)
    fig.savefig(out_base + ".svg")
    plt.close(fig)

    return vals, sig_ch, (vmin, vmax)


def _plot_tfr_with_contour(F3d, mask3d, freqs, times, ch_inds, out_base, title):
    # mean across cluster-channels 
    Fmean = F3d[ch_inds, :, :].mean(axis=0) if len(
        ch_inds) else F3d.mean(axis=0)
    mask2d = mask3d.any(axis=0)

    # fixed color limits (or auto)
    if FIX_CBAR:
        vmin, vmax = F_CBAR_TFR
    else:
        vmin = 0.0
        vmax = float(np.nanmax(Fmean)) if np.isfinite(Fmean).any() else 1.0

    # time in milliseconds
    times_ms = np.asarray(times, dtype=float) * 1000.0

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(
        Fmean, aspect='auto', origin='lower',
        extent=[times_ms[0], times_ms[-1], freqs[0], freqs[-1]],
        interpolation='nearest', vmin=vmin, vmax=vmax, cmap='Reds'
    )
    
    # a single crisp outer edge
    ax.contour(times_ms, freqs, mask2d.astype(float),
               levels=[0.5], linewidths=1.4, colors='k')

    ax.set_xlabel("Time (ms)", labelpad=10)
    ax.set_ylabel("Frequency (Hz)", labelpad=10)
    ax.set_yticks([4, 8, 13, 18, 23, 28])
    ax.set_title(title, pad=15)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("F value")

    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=200)
    fig.savefig(out_base + ".svg")
    plt.close(fig)


def _rezero_times_if_prep(times, segment):
    """Show PREP ending at 0; RESET starting at 0."""
    times = np.asarray(times).copy()
    seg = segment.decode() if isinstance(segment, (bytes,)) else str(segment)
    seg_up = seg.upper()
    if seg_up == "PREP":
        # Make the last PREP sample 0 (end-aligned)
        return times - times[-1]
    if seg_up == "RESET":
        # Make the first RESET sample 0 (start-aligned)
        return times - times[0]
    return times


def inspect_one(npz_path: Path, out_root: Path):
    d = np.load(npz_path, allow_pickle=True)

    # ---- meta ----
    seg = d["segment"].item() if d["segment"].ndim == 0 else str(
        d["segment"][()])
    n_ch, n_f, n_t = int(d["n_ch"]), int(d["n_freqs"]), int(d["n_times"])
    ch_names = [str(x) for x in d["ch_names"]]
    freqs = np.asarray(d["freqs"], dtype=float)
    times = np.asarray(d["times"], dtype=float)
    times = _rezero_times_if_prep(times, seg)

    # comparison label 
    # Determine label from filename so LIGHT & OBSTACLE both work
    stem_lower = Path(npz_path).stem.lower()
    if "obstacle" in stem_lower:
        label = "OBSTACLE_MAIN"   
    elif "light" in stem_lower:
        label = "LIGHT_MAIN"      
    else:
        label = "MAIN_EFFECT"

    sample_epochs_path = (
        str(d["sample_epochs_path"].item())
        if d["sample_epochs_path"].ndim == 0
        else str(d["sample_epochs_path"][()])
    )

    # ---- stats & clusters ----
    F1d = np.asarray(d["F_obs"], dtype=float).ravel()
    pvals = np.asarray(d["p_vals"], dtype=float)
    n_feat = n_ch * n_f * n_t

    if "masks_packed" in d.files:
        masks = _unpack_masks(d["masks_packed"], n_feat)
    elif "clusters" in d.files:
        masks = _unpack_masks(d["clusters"], n_feat)
    else:
        masks = []

    # --- checks ---
    if "F_obs" not in d.files:
        raise ValueError(
            f"{npz_path} does not contain 'F_obs' (is this an ANOVA NPZ?).")
    if len(masks) != len(pvals):
        raise ValueError(
            f"{npz_path}: #masks ({len(masks)}) != #pvals ({len(pvals)})")
    for i, m in enumerate(masks):
        if m.size != n_feat:
            raise ValueError(
                f"{npz_path}: mask[{i}] has {m.size} elements, expected n_feat={n_feat} "
                f"(n_ch={n_ch} * n_f={n_f} * n_t={n_t})"
            )

    # ---- output dir  ----
    parent = npz_path.parent.name
    seg_up = str(seg).upper()
    out_dir = _ensure_dir(out_root / parent /
                          f"{npz_path.stem}__{seg_up}__{label}")
    summary_path = Path(out_dir) / "summary.txt"

    # ---- reshape once ----
    F3d = _reshape_1d_to_3d(F1d, n_ch, n_f, n_t)

    # ---- gather clusters and sort (by p asc, then F_mass desc) ----
    clusters = []
    for k, (mask1d, pv) in enumerate(zip(masks, pvals)):
        if not np.isfinite(pv) or pv > ALPHA_REPORT:
            continue
        m1 = mask1d.astype(bool)
        n_points = int(m1.sum())
        if n_points == 0:
            continue

        # F stats inside the cluster
        F_in = F1d[m1]
        F_mass = float(F_in.sum())

        # Peak F inside the cluster
        peak_idx = _peak_index_from_mask(F1d, m1)
        peakF = float(F1d[peak_idx])

        # Store everything
        clusters.append((k, m1, pv, peakF, F_mass, n_points))

    clusters.sort(key=lambda x: (x[2], -x[4]))

    lines = []
    lines.append(f"File: {npz_path.name}")
    lines.append(f"Segment: {seg}")
    lines.append(f"Comparison: {label}")
    lines.append(f"n_ch={n_ch}, n_freqs={n_f}, n_times={n_t}")
    lines.append(f"alpha_report={ALPHA_REPORT}")
    lines.append("")

    n_sig = 0
    for k, mask1d, pv, peakF, F_mass, n_points in clusters:
        n_sig += 1
        mask3d = _reshape_1d_to_3d(mask1d, n_ch, n_f, n_t)

        # bounds & channels
        ch_inds, f_min, f_max, t_min, t_max = _find_cluster_bounds(mask3d)
        ch_list = [ch_names[i] for i in ch_inds]

        # decode peak voxel indices
        peak_idx = _peak_index_from_mask(F1d, mask1d)
        ch_i = peak_idx // (n_f * n_t)
        rem = peak_idx % (n_f * n_t)
        f_i = rem // n_t
        t_i = rem % n_t

        # plots 
        base_topo = os.path.join(
            out_dir, f"cluster{str(k).zfill(2)}_topomap_peak")
        base_tfr = os.path.join(out_dir, f"cluster{str(k).zfill(2)}_TFR")

        # aggregated topomap over the whole cluster (freq×time) per channel
        base_topo_agg = os.path.join(
            out_dir, f"cluster{str(k).zfill(2)}_topomap_agg")
        agg_title = (f"Topomap • mean F over cluster "
                     f"(f={freqs[f_min]:.2f}-{freqs[f_max]:.2f} Hz, "
                     f"t={times[t_min]*1000:.0f}–{times[t_max]*1000:.0f} ms)")
        _plot_topomap_cluster_agg(
            F3d, mask3d, ch_names, sample_epochs_path, base_topo_agg,
            title=agg_title, reduce="mean"
        )

        ringed = _plot_topomap_at_peak(
            F3d, mask3d, ch_names, sample_epochs_path, f_i, t_i, base_topo)
        title = f"Cluster {k} (p={pv:.4g}) • Peak F={peakF:.2f} at (f={freqs[f_i]:.2f} Hz, t={times[t_i]*1000:.0f} ms)"
        _plot_tfr_with_contour(F3d, mask3d, freqs, times,
                               ch_inds, base_tfr, title)

        # summary
        lines.append(f"[Cluster {k}] p={pv:.4g}")
        lines.append(f"  n_points: {n_points}")
        lines.append(f"  F_mass: {F_mass:.3f}")
        lines.append(f"  Channels ({len(ch_list)}): {', '.join(ch_list)}")
        lines.append(f"  topomap_agg: {base_topo_agg}.png")

        lines.append(
            f"  Freqs: {freqs[f_min]:.2f}–{freqs[f_max]:.2f} Hz  (idx {f_min}..{f_max})")
        lines.append(
            f"  Times: {times[t_min]*1000:.0f}–{times[t_max]*1000:.0f} ms   (idx {t_min}..{t_max})")
        lines.append(
            f"  Peak F: {peakF:.3f} at ch={ch_names[ch_i]}, f={freqs[f_i]:.2f} Hz, t={times[t_i]*1000:.0f} ms")
        if ringed:
            lines.append(f"  Ringed at peak (channels): {', '.join(ringed)}")
        lines.append(f"  topomap: {base_topo}.png")
        lines.append(f"  tfr:     {base_tfr}.png")
        lines.append("")

    if n_sig == 0:
        lines.append("No significant clusters at alpha_report.")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    meta = dict(
        file=npz_path.name,
        segment=str(seg),
        comparison=label,
        n_ch=n_ch, n_freqs=n_f, n_times=n_t,
        alpha_report=ALPHA_REPORT,
        n_sig=n_sig,
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] {npz_path.name}: wrote {n_sig} cluster plots to: {out_dir}")


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Assert files exist 
    for f in FILES:
        if not f.is_file():
            raise FileNotFoundError(f"Expected NPZ not found: {f}")
        with np.load(f, allow_pickle=True) as d:
            if "F_obs" not in d.files:
                raise ValueError(f"{f} missing 'F_obs' (not an ANOVA NPZ)")

    for f in FILES:
        inspect_one(f, OUTPUT_ROOT)
