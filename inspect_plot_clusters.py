# -*- coding: utf-8 -*-
"""
Inspector for the age main-effect cluster-permutation results (t-tests).

For each NPZ:
    * Load stats and metadata, rebuild the cluster masks.
    * Identify significant clusters 
    * Save a TFR with significance contours,
      a topomap of mean t over the cluster window, and a topomap at the peak.

Outputs go to OUTPUT_ROOT/<parent_folder>/<file_stem>__<segment>__<comparison>/.
Supplies panels A and B of Figure 4.
"""

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

import matplotlib as mpl

from matplotlib.collections import PathCollection

# Arial and fixed text sizes for every figure, matching the F-value
# light/obstacle script so topomaps and TFRs share one style.
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# -------- Config --------
ALPHA_REPORT = 0.05
SENSOR_DOT_SIZE = 40   # non-significant sensor marker size
INPUT_ROOT = r"D:\cluster_outputs"
OUTPUT_ROOT = r"D:\Cluster_plots_last_prep"
# ------------------------

# Plot-time time-axis controls (no stats rerun)
REZERO_TIME_AXES = True      # turn on re-zeroing
PREP_WINDOW_S = 2.7          # show last 2.7 s as -2.7..0 (0 = crossing START)
RESET_WINDOW_S = 1.0         # show first 1.0 s as 0..+1.0 (0 = crossing END)

# ---------- Helpers ----------


def slugify(s: str, max_len: int = 120) -> str:
    s = str(s)
    s = s.replace(" ", "_")
    s = re.sub(r"[^\w\.\-]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip(" ._-")
    if not s:
        s = "unnamed"
    return s[:max_len]


def nice_title(stem: str, comparison: str | None, segment: str, p: float) -> str:
    # prefer the NPZ's "comparison" label if present
    base = comparison or stem
    # strip perm/seed tokens like "...1000perm_seed42"
    base = re.sub(r'(?i)_?\d+perm(?:_seed\d+)?', '', base)
    base = re.sub(r'(?i)_?seed\d+', '', base)
    
    base = re.sub(r'(?i)_?ttest', ' t-test', base)
    base = re.sub(r'_+', '_', base).strip('_')
    return f"{base} | {segment} | p={p:.3g}"


def _as_str(x) -> str:
    if isinstance(x, np.ndarray):
        try:
            return str(x.item())
        except Exception:
            return str(x)
    return str(x)


def load_npz_robust(path):
    Z = np.load(path, allow_pickle=True)

    # t-values vector
    if "stat_obs" in Z:
        T_obs = np.asarray(Z["stat_obs"]).ravel()
    elif "T_obs" in Z:
        T_obs = np.asarray(Z["T_obs"]).ravel()
    else:
        raise KeyError(f"{path}: missing stat_obs/T_obs")

    # axes sizes
    n_ch = int(Z["n_ch"]) if "n_ch" in Z else int(Z["n_channels"])
    n_freq = int(Z["n_freqs"])
    n_time = int(Z["n_times"])

    # channel names
    ch_names = [str(x) for x in np.array(Z["ch_names"], dtype=object).tolist()]

    # descriptors
    segment = _as_str(Z["segment"]) if "segment" in Z else "UNKNOWN_SEGMENT"
    comparison = _as_str(
        Z["comparison"]) if "comparison" in Z else Path(path).stem

    # freq/time axes
    freqs = np.asarray(Z["freqs"], dtype=float)
    times = np.asarray(Z["times"], dtype=float)

    # epochs path for topomap layout
    sample_epochs_path = _as_str(
        Z["sample_epochs_path"]) if "sample_epochs_path" in Z else ""

    # cluster p-values
    if "p_vals" in Z:
        p_vals = np.asarray(Z["p_vals"], dtype=float)
    elif "pval" in Z:
        p_vals = np.asarray(Z["pval"], dtype=float)
    else:
        p_vals = np.array([])

    # rebuild masks
    n_feat = n_ch * n_freq * n_time
    masks = []
    if "masks_packed" in Z:
        mp = Z["masks_packed"]
        if mp.dtype == bool:
            arr = np.asarray(mp)
            if arr.ndim == 1 and arr.size == n_feat:
                masks = [arr.astype(bool).ravel()]
            elif arr.ndim == 2 and arr.shape[1] == n_feat:
                masks = [arr[i].astype(bool).ravel()
                         for i in range(arr.shape[0])]
            else:
                raise ValueError(
                    f"{path}: boolean masks_packed shape {arr.shape} unexpected (n_feat={n_feat})")
        elif mp.dtype == object:
            for obj in mp:
                packed = np.asarray(obj, dtype=np.uint8)
                unpacked = np.unpackbits(packed)
                if unpacked.size < n_feat:
                    raise ValueError(
                        f"{path}: unpacked mask too short {unpacked.size} < {n_feat}")
                masks.append(unpacked[:n_feat].astype(bool))
        elif np.asarray(mp).dtype == np.uint8 and np.asarray(mp).ndim == 2:
            arr = np.asarray(mp)
            for i in range(arr.shape[0]):
                unpacked = np.unpackbits(arr[i])
                masks.append(unpacked[:n_feat].astype(bool))
        else:
            raise ValueError(
                f"{path}: unsupported masks_packed dtype {np.asarray(mp).dtype}")
    elif "clusters" in Z:
        cl = Z["clusters"]
        try:
            if isinstance(cl, np.ndarray) and cl.dtype == object:
                masks = [np.asarray(c, dtype=bool).ravel() for c in cl]
            else:
                masks = [np.asarray(c, dtype=bool).ravel() for c in cl]
        except Exception:
            masks = []
    else:
        masks = []

    return dict(
        T_obs=T_obs,
        n_ch=n_ch, n_freq=n_freq, n_time=n_time,
        ch_names=ch_names,
        segment=segment,
        comparison=comparison,
        freqs=freqs, times=times,
        p_vals=p_vals,
        masks=masks,
        sample_epochs_path=sample_epochs_path,
    )


def unflatten(T_vec, n_ch, n_freq, n_time):
    return T_vec.reshape(n_ch, n_freq, n_time)


def cluster_summary(mask3d, ch_names, freqs, times):
    ch_idx = np.where(mask3d.any(axis=(1, 2)))[0]
    f_idx = np.where(mask3d.any(axis=(0, 2)))[0]
    t_idx = np.where(mask3d.any(axis=(0, 1)))[0]

    ch_list = [ch_names[i] for i in ch_idx]
    f_min, f_max = (np.nan, np.nan)
    t_min, t_max = (np.nan, np.nan)
    if f_idx.size:
        f_min, f_max = float(freqs[f_idx[0]]), float(freqs[f_idx[-1]])
    if t_idx.size:
        t_min, t_max = float(times[t_idx[0]]), float(times[t_idx[-1]])
    return ch_idx, f_idx, t_idx, ch_list, (f_min, f_max), (t_min, t_max)


# ---------- PLOTTING ----------

def plot_tfr(T3, mask3d, freqs, times, ch_idx, out_basename, title, xlabel="Time (s)", xlim=None):
    """
    Save TFR heatmap (mean across cluster channels) with significance contours.
    - X-axis shown in milliseconds.
    - Y-axis ticks at 4, 8, 13, 18, 23, 28 Hz.
    - Colorbar labeled 't value'.
    - NaNs rendered as light gray.
    - Symmetric color limits around 0 for t-values.
    """
    if len(ch_idx) == 0:
        return

    # Use milliseconds on the x-axis
    times_ms = np.asarray(times, dtype=float) * 1000.0
    xlim_ms = (xlim[0] * 1000.0, xlim[1] *
               1000.0) if xlim is not None else None

    # Average across cluster channels
    T2 = T3[ch_idx, :, :].mean(axis=0)  # (freq, time)

    # Aggregate significance mask across channels
    mask2 = mask3d[ch_idx, :, :].any(axis=0)  # (freq, time)

    # Symmetric vlims based on finite values 
    finite = np.isfinite(T2)
    if mask2.any() and np.any(finite & mask2):
        vmax = float(np.nanmax(np.abs(T2[finite & mask2])))
    else:
        vmax = float(np.nanmax(np.abs(T2[finite]))) if np.any(finite) else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin, vmax = -vmax, vmax

    # Mask invalids so they render as a neutral color
    T2m = np.ma.masked_invalid(T2)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color='0.90')  # light gray for NaNs

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    pcm = ax.pcolormesh(
        times_ms, freqs, T2m, shading='nearest',
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    # Labels/limits/ticks
    ax.set_xlabel("Time (ms)", labelpad=10)
    if xlim_ms is not None:
        ax.set_xlim(*xlim_ms)
    ax.set_ylabel("Frequency (Hz)", labelpad=10)
    ax.set_yticks([4, 8, 13, 18, 23, 28])
    ax.set_title(title, pad=15)

    # Significance contours (use ms on x-axis)
    if mask2.any():
        tt, ff = np.meshgrid(times_ms, freqs)
        ax.contour(tt, ff, mask2.astype(float), levels=[
                   0.5], linewidths=1, colors='k')

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("t-value")

    fig.tight_layout()
    fig.savefig(f"{out_basename}.png", dpi=300)
    fig.savefig(f"{out_basename}.svg")
    plt.close(fig)


def plot_topomap_at_peak(T3, mask3d, info, freqs, times, ch_names, out_basename, title):
    """
    Find peak |t| inside mask3d, save a topomap at that (f*, t*), and return
    (t_peak, f_peak, t_peak_s, ch_peak). Writes PNG + SVG.
    """
    if not mask3d.any() or info is None:
        return None

    # Peak indices within the cluster
    Tmasked = np.where(mask3d, np.abs(T3), -np.inf)
    ch_i, f_i, t_i = np.unravel_index(np.nanargmax(Tmasked), T3.shape)

    have = [ch for ch in ch_names if ch in info["ch_names"]]
    if len(have) == 0:
        return None

    picks = mne.pick_channels(info["ch_names"], include=have)
    info_use = mne.pick_info(info, picks)
    idx_map = [ch_names.index(name) for name in info_use["ch_names"]]
    data = T3[np.array(idx_map), f_i, t_i]

    vmax = float(np.nanmax(np.abs(data))) or 1.0
    vlim = (-vmax, vmax)

    # Mark which channels are significant at this exact (f, t)
    ch_mask_slice = mask3d[:, f_i, t_i]
    ch_mask_reordered = np.array(
        [ch_mask_slice[ch_names.index(nm)] for nm in info_use["ch_names"]], dtype=bool)

    # Names of ringed (peak-slice) channels in the same order as the topomap
    ringed_channels = [nm for nm, flag in zip(
        info_use["ch_names"], ch_mask_reordered) if flag]

    ch_peak_name = ch_names[ch_i]
    t_peak_val = float(T3[ch_i, f_i, t_i])   # signed t at the peak
    f_peak = float(freqs[f_i])
    t_peak = float(times[t_i])

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im, _ = mne.viz.plot_topomap(
        data, info_use, axes=ax, show=False, cmap='RdBu_r',
        vlim=vlim, sensors=True, contours=6, outlines='head',
        mask=ch_mask_reordered,
        mask_params=dict(marker='o', markerfacecolor='none',
                         markeredgecolor='k', linewidth=1.2)
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("t")
    ax.set_title(
        f"{title}\nPeak |t| @ {f_peak:.2f} Hz, {t_peak:.3f} s ({ch_peak_name})")
    fig.tight_layout()
    fig.savefig(f"{out_basename}.png", dpi=300)
    fig.savefig(f"{out_basename}.svg")
    plt.close(fig)

    return dict(t_peak=t_peak_val, f_peak=f_peak, t_peak_s=t_peak, ch_peak=ch_peak_name, ringed_channels=ringed_channels, )


def plot_topomap_cluster_mean_t(T3, mask3d, info, ch_names, out_basename, title):
    """
    Aggregated topomap: for each channel, compute the MEAN t-value over the cluster's
    significant freq×time window (union across channels). Black dots mark channels
    that are significant anywhere in the cluster.
    """
    if info is None or not mask3d.any():
        return

    tf_window = mask3d.any(axis=0)                     # (n_freq, n_time)
    n_ch = T3.shape[0]
    vals = np.full(n_ch, np.nan, float)
    for ch in range(n_ch):
        x = T3[ch][tf_window]
        if x.size:
            vals[ch] = np.nanmean(x)                   # signed mean t

    # reorder to match topomap info order
    have = [ch for ch in ch_names if ch in info["ch_names"]]
    if not have:
        return
    picks = mne.pick_channels(info["ch_names"], include=have)
    info_use = mne.pick_info(info, picks)
    order = [ch_names.index(nm) for nm in info_use["ch_names"]]
    vals_plot = vals[order]

    # significant electrodes = any (f,t) in cluster at that channel
    sig_ch = mask3d.any(axis=(1, 2))
    sig_plot = sig_ch[[ch_names.index(nm) for nm in info_use["ch_names"]]]

    # symmetric color scaling around 0 (prefer scaling from significant chans)
    finite = np.isfinite(vals_plot)
    src = vals_plot[finite & sig_plot] if sig_plot.any() else vals_plot[finite]
    vmax = float(np.nanmax(np.abs(src))) if src.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im, _ = mne.viz.plot_topomap(
        vals_plot, info_use, axes=ax, show=False, cmap='RdBu_r',
        vlim=(vmin, vmax), outlines="head", sensors=True,
        mask=sig_plot,
        mask_params=dict(marker='o', markerfacecolor='black',
                         markeredgecolor='black', markersize=14, linewidth=0)
    )
    # Enlarge non-significant sensor markers
    for _coll in ax.collections:
        if isinstance(_coll, PathCollection):
            _coll.set_sizes([SENSOR_DOT_SIZE])
            _coll.set_clip_on(False)

    ax.set_title(title, pad=15)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("t-value")
    fig.tight_layout()
    fig.savefig(f"{out_basename}.png", dpi=300)
    fig.savefig(f"{out_basename}.svg")
    plt.close(fig)


def load_info_from_epochs(sample_epochs_path, want_ch_names):
    if not sample_epochs_path or not os.path.exists(sample_epochs_path):
        return None
    try:
        ep = mne.read_epochs(sample_epochs_path, preload=False, verbose=False)
        picks = mne.pick_channels(ep.info["ch_names"], include=want_ch_names)
        if len(picks) == 0:
            return ep.info
        return mne.pick_info(ep.info, picks)
    except Exception:
        return None


# ---------- Main ----------

def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    index_rows = []

    # files = [p for p in input_root.rglob("*.npz")]
    files = [
        input_root / "age_main_PREP_ttest_last_prep1000perm_seed42.npz",
        input_root / "age_main_RESET_ttest_last_prep1000perm_seed42.npz",
    ]
    if not files:
        print(f"No NPZ files found under {INPUT_ROOT}")
        return

    print(f"Inspecting {len(files)} NPZ files.")

    for fpath in files:
        print(f"\n=== Inspecting: {fpath} ===")
        try:
            info = load_npz_robust(str(fpath))
        except Exception as e:
            print(f"[SKIP] {fpath}: {e}")
            continue

        T_obs = info["T_obs"]
        n_ch = info["n_ch"]
        n_freq = info["n_freq"]
        n_time = info["n_time"]
        ch_names = info["ch_names"]
        freqs = info["freqs"]
        times = info["times"]
        segment = info["segment"]
        comparison = info["comparison"]
        p_vals = info["p_vals"]
        masks = info["masks"]
        sample_epochs_path = info["sample_epochs_path"]

        # --- Re-zero time axis for plotting ---
        # turn times axis to numpy array
        times_raw = np.asarray(times, dtype=float)
        times_plot = times_raw.copy()
        xlabel = "Time (s)"
        xlim = None

        if REZERO_TIME_AXES:
            seg_upper = str(segment).upper()
            if "PREP" in seg_upper:
                # 0 at PREP end (= crossing START); show -2.7..0
                times_plot = times_raw - times_raw[-1]

                xlabel = "Time (s, rel. to crossing START)"
                xlim = (-PREP_WINDOW_S, 0.0)
            elif "RESET" in seg_upper:
                # 0 at RESET start (= crossing END); show 0..+1.0
                times_plot = times_raw - times_raw[0]

                xlabel = "Time (s, rel. to crossing END)"
                xlim = (0.0, RESET_WINDOW_S)

        times_use = times_plot
        # -------------------------------------------------------------

        parent_label = slugify(fpath.parent.name) or "root"
        stem_safe = slugify(fpath.stem)
        seg_safe = slugify(segment)
        comp_safe = slugify(comparison)
        test_label = f"{stem_safe}__{seg_safe}__{comp_safe}"

        test_dir = output_root / parent_label / test_label
        test_dir.mkdir(parents=True, exist_ok=True)

        with open(test_dir / "meta.json", "w", encoding="utf-8") as fh:
            meta = {
                "file": str(fpath),
                "parent_folder": str(fpath.parent),
                "segment": segment,
                "comparison": comparison,
                "n_ch": n_ch,
                "n_freq": n_freq,
                "n_time": n_time,
                "ch_names": ch_names,
                "freqs_min": float(freqs[0]) if len(freqs) else None,
                "freqs_max": float(freqs[-1]) if len(freqs) else None,
                "times_min": (xlim[0] if xlim is not None
                              else (float(times_use[0]) if len(times_use) else None)),
                "times_max": (xlim[1] if xlim is not None
                              else (float(times_use[-1]) if len(times_use) else None)),


            }
            json.dump(meta, fh, indent=2)

        T3 = unflatten(T_obs, n_ch, n_freq, n_time)

        T3_use = T3

        info_obj = load_info_from_epochs(sample_epochs_path, ch_names)

        any_sig = False
        summary_lines = [
            "Notes: Diverging colormap centered at 0 for t-values; black contours show cluster significance at α=0.05.",
            ""
        ]

        for k, m in enumerate(masks):
            m = np.asarray(m, dtype=bool).ravel()
            if m.size != (n_ch * n_freq * n_time):
                print(
                    f"[WARN]   cluster {k}: mask size {m.size} != n_feat; skipping")
                continue

            pv = float(p_vals[k]) if k < len(p_vals) else np.nan
            sig = (pv <= ALPHA_REPORT) if np.isfinite(pv) else False

            if not sig:
                index_rows.append(dict(
                    file=str(fpath), parent=parent_label, segment=segment, comparison=comparison,
                    cluster=k, p_value=pv, significant=False,
                    n_points=int(np.sum(m)), n_channels=np.nan,
                    f_min=np.nan, f_max=np.nan, t_min=np.nan, t_max=np.nan,
                    peak_t=np.nan, peak_abs_t=np.nan, peak_freq_hz=np.nan,
                    peak_time_s=np.nan, peak_channel=""
                ))
                continue

            any_sig = True
            m3 = m.reshape(n_ch, n_freq, n_time)
            m3_use = m3
            ch_idx, f_idx, t_idx, ch_list, (fmin, fmax), (tmin, tmax) = cluster_summary(
                m3_use, ch_names, freqs, times_use
            )

            # aggregated mean-t topomap over the cluster window
            agg_title = (f"Topomap • mean t over cluster "
                         f"(f={fmin:.2f}-{fmax:.2f} Hz, t={tmin:.3f}–{tmax:.3f} s)")
            plot_topomap_cluster_mean_t(
                T3_use, m3_use, info_obj, ch_names,
                out_basename=str(test_dir / f"cluster{k:02d}_topomap_agg"),
                title=agg_title
            )

            # --- Cluster statistics to report in summary.txt ---
            # Sum of t-values inside the cluster mask = "cluster tmax (Σt)"
            cluster_tmass = float(T_obs[m].sum())
            cluster_sign = "positive" if cluster_tmass > 0 else "negative"

            # Build a clean figure title (removes "...1000perm", "seed42", etc.)
            title_str = nice_title(fpath.stem, comparison, segment, pv)

            plot_tfr(
                T3_use, m3_use, freqs, times_use, ch_idx,
                out_basename=str(test_dir / f"cluster{k:02d}_TFR"),
                title=title_str,  xlabel=xlabel, xlim=xlim
            )
            # Diagnostic only, not used in the paper figures.
            peak_info = None
            try:
                peak_info = plot_topomap_at_peak(
                    T3_use, m3_use, info_obj, freqs, times_use, ch_names,
                    out_basename=str(
                        test_dir / f"cluster{k:02d}_topomap_peak"),
                    title=title_str,
                )

            except Exception as e:
                print(f"[WARN]   cluster {k}: topomap failed ({e})")

            peak_line = ""
            if peak_info is not None:
                peak_line = (f"peak |t|      : {abs(peak_info['t_peak']):.3f} "
                             f"(t={peak_info['t_peak']:+.3f}) @ "
                             f"{peak_info['f_peak']:.2f} Hz, {peak_info['t_peak_s']:.3f} s, "
                             f"electrode {peak_info['ch_peak']}")
            ring_line = ""
            if peak_info is not None:
                rc = peak_info.get("ringed_channels", [])
                if rc:
                    ring_line = f"peak-slice channels (n={len(rc)}): {', '.join(rc)}"

            summary_lines += [
                f"Cluster {k}",
                f"p-value       : {pv:.6g}",
                f"tmax (Σt)     : {cluster_tmass:.3f} ({cluster_sign} cluster)",
                f"n points      : {int(np.sum(m))}",
                f"channels (n={len(ch_list)}): {', '.join(ch_list)}",
                f"freqs range   : {fmin:.3f}–{fmax:.3f} Hz",
                f"times range   : {tmin*1000:.0f}–{tmax*1000:.0f} ms",
                peak_line,
                ring_line,
                ""
            ]

            row = dict(
                file=str(fpath), parent=parent_label, segment=segment, comparison=comparison,
                cluster=k, p_value=pv, significant=True,
                n_points=int(np.sum(m)), n_channels=len(ch_list),
                f_min=fmin, f_max=fmax, t_min=tmin, t_max=tmax,
                peak_t=np.nan, peak_abs_t=np.nan, peak_freq_hz=np.nan,
                peak_time_s=np.nan, peak_channel=""
            )
            if peak_info is not None:
                row.update(dict(
                    peak_t=peak_info['t_peak'],
                    peak_abs_t=abs(peak_info['t_peak']),
                    peak_freq_hz=peak_info['f_peak'],
                    peak_time_s=peak_info['t_peak_s'],
                    peak_channel=peak_info['ch_peak'],
                ))
            index_rows.append(row)

        with open(test_dir / "summary.txt", "w", encoding="utf-8") as fh:
            if any_sig:
                fh.write("\n".join(summary_lines))
            else:
                fh.write("No significant clusters at alpha = 0.05\n")

        print(f"[DONE] {fpath.name} → {test_dir}")

    # df = pd.DataFrame(index_rows)
    # df.to_csv(Path(OUTPUT_ROOT) / "master_summary.csv", index=False)
    # print(f"\nMaster summary: {Path(OUTPUT_ROOT) / 'master_summary.csv'}")


if __name__ == "__main__":
    main()
