# -*- coding: utf-8 -*-

"""
Post-hoc line plots for the OBSTACLE main effect:
  Contrast = Present (EP+UP)/2  vs  Absent (EA+UA)/2
Collapsed across LIGHT (LIGHT, AMBIENT, DARK), for PREP and RESET segments.

Per significant omnibus cluster (p < 0.05):
  1) Build ROI from cluster mask (OR over time -> ch×freq)
  2) Extract per-time ROI series for Present & Absent (per subject)
  3) Run paired time-only cluster permutation on the Present-Absent difference
  4) For each significant time window (p < 0.05):
       - Plot mean lines with within-subject 95% CI (Cousineau-Morey)
       - Shade significant windows with a band + dashed boundaries
       - Append one row per window to CSV 
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from mne.stats import permutation_cluster_1samp_test
from matplotlib.ticker import FuncFormatter


from cluster_config import participants, base_dir
from extract_prep_reset import load_and_extract_prep_reset

import matplotlib as mpl

# Arial and fixed text sizes
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# ---------- CONFIG ----------
NPZ_PREP = r"D:\cluster_outputs\obstacle_main_effect_prep_1000perm_seed42.npz"
NPZ_RESET = r"D:\cluster_outputs\obstacle_main_effect_reset_1000perm_seed42.npz"


PLOT_ROOT = Path(
    r"D:\Cluster_plots_last_prep\line_plots\obstacle_present_vs_absent")
CSV_ROOT = Path(r"D:\cluster_outputs\line_posthocs\obstacle_present_vs_absent")

ALPHA_REPORT = 0.05   # keep omnibus clusters with p <= 0.05
# cluster-forming alpha for time-only test (two-sided)
ALPHA_CF = 0.001
N_PERM = 1000
SEED = 42

LIGHT_LEVELS = ["LIGHT", "AMBIENT", "DARK"]
# ---------------------------


# -------------------- helpers --------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rezero_times_if_prep(times, segment):
    """Show PREP ending at 0; RESET starting at 0."""
    times = np.asarray(times).copy()
    seg = segment.decode() if isinstance(segment, (bytes,)) else str(segment)
    seg_up = seg.upper()
    if seg_up == "PREP":
        return times - times[-1]   # end-aligned
    if seg_up == "RESET":
        return times - times[0]    # start-aligned
    return times


def _unpack_masks(masks_any, n_feat):
    """
    Return list of 1D boolean masks (length n_feat), one per cluster.
    Accepts packed object arrays, 2D packed rows, 2D unpacked, or list-like.
    """
    if masks_any is None:
        return []
    n_bytes_expected = (n_feat + 7) // 8

    arr_obj = np.asarray(masks_any, dtype=object)
    if arr_obj.dtype == object and arr_obj.ndim == 1:
        out = []
        for pb in arr_obj:
            if isinstance(pb, (bytes, bytearray)):
                buf = np.frombuffer(pb, dtype=np.uint8)
            else:
                buf = np.asarray(pb, dtype=np.uint8)
            bits = np.unpackbits(buf)
            out.append(bits[:n_feat].astype(bool))
        return out

    arr = np.asarray(masks_any)
    if arr.ndim == 2 and arr.shape[1] == n_bytes_expected:
        return [np.unpackbits(np.asarray(row, dtype=np.uint8))[:n_feat].astype(bool) for row in arr]
    if arr.ndim == 2 and arr.shape[1] == n_feat:
        return [arr[i].astype(bool, copy=False) for i in range(arr.shape[0])]

    try:
        return [np.asarray(m, dtype=bool).ravel()[:n_feat] for m in masks_any]
    except Exception as e:
        raise TypeError(
            f"Cannot parse cluster masks of type {type(masks_any)}") from e


def _collect_present_absent(segment: str, prep_window: str = "last"):
    """
    Returns:
      present  : (n_subj, n_ch, n_f, n_t)
      absent   : (n_subj, n_ch, n_f, n_t)
      times    : (n_t,)
      freqs    : (n_f,)
    Collapse within each LIGHT first (Present=(EP+UP)/2, Absent=(EA+UA)/2),
    then average across the 3 LIGHTS.
    """
    assert segment in ("PREP", "RESET")

    per_subj_present, per_subj_absent = [], []
    freqs = None
    times = None

    for pid in participants:
        per_light_present, per_light_absent = [], []

        for light in LIGHT_LEVELS:
            cond_EP = f"GLOBAL_{light}_EXPECTED_PRESENT"
            cond_UP = f"GLOBAL_{light}_UNEXPECTED_PRESENT"
            cond_EA = f"GLOBAL_{light}_EXPECTED_ABSENT"
            cond_UA = f"GLOBAL_{light}_UNEXPECTED_ABSENT"

            EP_prep, EP_reset, t_prep, t_reset, fr = load_and_extract_prep_reset(
                pid, cond_EP, base_dir, prep_window=prep_window)
            UP_prep, UP_reset, _,      _,      _ = load_and_extract_prep_reset(
                pid, cond_UP, base_dir, prep_window=prep_window)
            EA_prep, EA_reset, _,      _,      _ = load_and_extract_prep_reset(
                pid, cond_EA, base_dir, prep_window=prep_window)
            UA_prep, UA_reset, _,      _,      _ = load_and_extract_prep_reset(
                pid, cond_UA, base_dir, prep_window=prep_window)

            arr_EP = EP_prep if segment == "PREP" else EP_reset
            arr_UP = UP_prep if segment == "PREP" else UP_reset
            arr_EA = EA_prep if segment == "PREP" else EA_reset
            arr_UA = UA_prep if segment == "PREP" else UA_reset

            if freqs is None:
                freqs = np.asarray(fr)
            if times is None:
                times = np.asarray(t_prep if segment == "PREP" else t_reset)

            per_light_present.append(0.5 * (arr_EP + arr_UP))
            per_light_absent.append(0.5 * (arr_EA + arr_UA))

        # collapse across LIGHT (mean of 3 lights)
        present_pid = np.mean(np.stack(per_light_present, axis=0), axis=0)
        absent_pid = np.mean(np.stack(per_light_absent,  axis=0), axis=0)

        per_subj_present.append(present_pid)
        per_subj_absent.append(absent_pid)

    present = np.stack(per_subj_present, axis=0)
    absent = np.stack(per_subj_absent,  axis=0)
    return present, absent, times, freqs


def _roi_timeseries_from_mask(arr4, mask1d, n_ch, n_f, n_t):
    """
    arr4   : (n_subj, n_ch, n_f, n_t)
    mask1d : (n_ch*n_f*n_t,) boolean
    return : (n_subj, n_t) averaged across ROI (ch×freq), per subject
    """
    mask3d = np.asarray(mask1d, bool).reshape(n_ch, n_f, n_t)
    roi_chf = mask3d.any(axis=2)  # collapse over time -> (ch, f)
    arr_flat = arr4.reshape(arr4.shape[0], n_ch*n_f, n_t)
    roi_idx = np.where(roi_chf.ravel())[0]
    if roi_idx.size == 0:
        raise RuntimeError(
            "Empty ch×freq ROI from significant omnibus cluster — check NPZ masks and shapes.")

    return arr_flat[:, roi_idx, :].mean(axis=1)


def _within_subject_ci_two_cond(yA, yB, alpha=0.05):
    """
    Cousineau–Morey within-subject CI for two conditions, per timepoint.
    yA,yB: (n_subj, n_time)   returns (meanA, ciA, meanB, ciB)
    """
    yA = np.asarray(yA, float)
    yB = np.asarray(yB, float)
    n, T = yA.shape
    grand = np.mean(np.stack([yA, yB], axis=0), axis=0)  # (n, T)
    overall_grand = grand.mean(axis=0, keepdims=True)    # (1, T)
    yA_norm = yA - grand + overall_grand
    yB_norm = yB - grand + overall_grand
    cm = np.sqrt(2/(2-1))  # Morey correction for m=2
    seA = yA_norm.std(axis=0, ddof=1) / np.sqrt(n)
    seB = yB_norm.std(axis=0, ddof=1) / np.sqrt(n)
    tcrit = t.ppf(1 - alpha/2.0, df=n-1)
    ciA = cm * tcrit * seA
    ciB = cm * tcrit * seB
    return yA.mean(axis=0), ciA, yB.mean(axis=0), ciB


def _time_cluster_test(diff, alpha_cf=ALPHA_CF, n_perm=N_PERM, seed=SEED):
    """
    diff: (n_subj, n_time) Present - Absent
    Paired time-only cluster test (two-sided).
    Returns: T_obs, [list of 1D boolean time-masks], p_vals, threshold_t
    """
    n = diff.shape[0]
    thr_t = t.ppf(1 - alpha_cf/2.0, df=n-1)  # two-sided
    T_obs, cl_inds, p_vals, _ = permutation_cluster_1samp_test(
        diff,
        n_permutations=n_perm,
        threshold=thr_t,
        tail=0,                 # 0 = two-sided
        out_type="indices",     # returns time indices per cluster
        seed=seed,
    )
    # turn indices into 1-D boolean masks (n_time,)
    n_time = diff.shape[1]
    masks = []
    for ind in cl_inds:
        m = np.zeros(n_time, dtype=bool)
        if len(ind):
            m[np.asarray(ind).ravel()] = True
        masks.append(m)
    return T_obs, masks, p_vals, float(thr_t)


def _plot_lines_with_shading(times, yA, yB, ciA, ciB, sig_windows,
                             seg, out_png, title_suffix="Present vs Absent"):
    from matplotlib.patches import Rectangle

    
    line_colors = ["#0D5C9D", "#1C7F46"]
    box_color = "black"  

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    # Present
    ax.plot(times, yA, label="Present", color=line_colors[0])
    ax.fill_between(times, yA - ciA, yA + ciA, alpha=0.25,
                    color=line_colors[0], linewidth=0)

    # Absent
    ax.plot(times, yB, label="Absent", color=line_colors[1])
    ax.fill_between(times, yB - ciB, yB + ciB, alpha=0.25,
                    color=line_colors[1], linewidth=0)

    # Rectangles for significant windows
    for w in sig_windows:
        start = float(w["time_start"])
        end = float(w["time_end"])
        left, right = start, end

        mask = (times >= left) & (times <= right)
        if not np.any(mask):
            continue
        win_min = min(np.nanmin((yA - ciA)[mask]), np.nanmin((yB - ciB)[mask]))
        win_max = max(np.nanmax((yA + ciA)[mask]), np.nanmax((yB + ciB)[mask]))
        pad = (win_max - win_min) * 0.08
        rect = Rectangle((left, win_min - pad), right - left,
                         (win_max - win_min) + 2 * pad,
                         fill=False, edgecolor=box_color,
                         linestyle=(0, (2, 1)), linewidth=1.2,
                         alpha=0.95, zorder=2.8, clip_on=False)
        ax.add_patch(rect)

    ax.set_xlabel("Time (ms)", labelpad=10)
    ax.xaxis.set_label_coords(0.5, -0.09)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*1000:.0f}"))
    ax.set_ylabel("Power (% change from baseline)", labelpad=10)
    ax.set_title(f"{title_suffix} • {seg}", pad=15)

    if str(seg).upper() == "PREP":
        ax.set_xlim(times[0], 0.0)
    elif str(seg).upper() == "RESET":
        ax.set_xlim(0.0, times[-1])
    ax.autoscale(axis='y', tight=False)
    ax.margins(y=0.05)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels,
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0., frameon=False, ncol=1)

    fig.savefig(out_png, dpi=200, bbox_inches='tight',
                bbox_extra_artists=(leg,))
    fig.savefig(out_png.with_suffix(".svg"),
                bbox_inches='tight', bbox_extra_artists=(leg,))
    plt.close(fig)


def _summarize_sig_windows(times, T_obs, clusters_1d, p_vals):
    """
    Produce rows only for significant time windows (p<0.05),
    with t-mass, bounds, and duration.
    """
    rows = []
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
    for k, c1d in enumerate(clusters_1d):
        pv = p_vals[k] if k < len(p_vals) else np.nan
        if not (np.isfinite(pv) and pv < 0.05):
            continue
        idx = np.where(c1d)[0]
        if idx.size == 0:
            continue
        dur_s = idx.size * dt
        rows.append(dict(
            # kept internally (not written to CSV)
            time_cluster=int(k),
            p_value=float(pv),
            n_timepoints=int(idx.size),
            time_start=float(times[idx[0]]),
            time_end=float(times[idx[-1]]),
            duration_s=float(dur_s),
            t_mass=float(np.sum(T_obs[idx])),
        ))
    return rows


def _cluster_roi_info(mask1d, n_ch, n_f, n_t, freqs, ch_names=None):
    """
    Return ROI summary with channel NAMES (from NPZ 'ch_names'),
    plus freq span and counts. 'roi_channels_list' matches LIGHT CSV style.
    """
    m3 = np.asarray(mask1d, bool).reshape(n_ch, n_f, n_t)
    ch_any = m3.any(axis=(1, 2))                # (ch,)
    f_any = m3.any(axis=(0, 2))                # (f,)
    ch_idx = np.where(ch_any)[0].tolist()
    f_idx = np.where(f_any)[0]

    if f_idx.size:
        fmin, fmax = float(freqs[f_idx.min()]), float(freqs[f_idx.max()])
    else:
        fmin, fmax = float(freqs[0]), float(freqs[-1])

    # Map indices - names when provided; else fall back to indices as strings
    if ch_names is not None and len(ch_names) == n_ch:
        ch_list = [str(ch_names[i]) for i in ch_idx]
    else:
        ch_list = [str(i) for i in ch_idx]

    return {
        "roi_n_channels": int(ch_any.sum()),
        "roi_n_freqs": int(f_any.sum()),
        "freq_min_hz": fmin,
        "freq_max_hz": fmax,
        "roi_channels_list": ";".join(ch_list),  # semicolon-separated names
    }


def _window_effect_summary(yP, yA, win_mask):
    """
    Aggregate stats within a significant time window (paired across subjects).
    Returns mean Present/Absent, mean diff, and 95% CI for the paired diff.
    """
    idx = np.where(win_mask)[0]
    subj_P = yP[:, idx].mean(axis=1)
    subj_A = yA[:, idx].mean(axis=1)
    subj_diff = subj_P - subj_A
    n = subj_diff.size
    diff_mean = float(subj_diff.mean())
    sd = float(subj_diff.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = t.ppf(0.975, df=n-1) if n > 1 else 0.0
    return {
        "mean_present": float(subj_P.mean()),
        "mean_absent": float(subj_A.mean()),
        "diff_mean": diff_mean,
        "diff_ci95_lo": float(diff_mean - tcrit*se),
        "diff_ci95_hi": float(diff_mean + tcrit*se),
        "n_subjects": int(n),
    }


# -------------------- main worker --------------------

def run_for_npz(npz_path, plots_dir: Path, csv_dir: Path):
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as d:
        if "F_obs" not in d.files:
            raise ValueError(f"{npz_path} missing 'F_obs' (not an ANOVA NPZ)")

        seg = d["segment"].item() if d["segment"].ndim == 0 else str(
            d["segment"][()])
        n_ch = int(d["n_ch"])
        n_f = int(d["n_freqs"])
        n_t = int(d["n_times"])
        freqs = np.asarray(d["freqs"], dtype=float)
        times = _rezero_times_if_prep(np.asarray(d["times"], dtype=float), seg)

        # channel names from NPZ
        ch_names = None
        if "ch_names" in d.files:
            arr = d["ch_names"]
            try:
                ch_names = [str(x)
                            for x in arr.tolist()]  # handles object arrays
            except Exception:
                ch_names = [str(x) for x in np.asarray(arr).ravel()]

        pvals = np.asarray(d["p_vals"], dtype=float)
        n_feat = n_ch * n_f * n_t

        if "masks_packed" in d.files:
            masks = _unpack_masks(d["masks_packed"], n_feat)
        elif "clusters" in d.files:
            masks = _unpack_masks(d["clusters"], n_feat)
        else:
            masks = []

        # checks
        if len(masks) != len(pvals):
            raise ValueError(
                f"{npz_path}: #masks ({len(masks)}) != #pvals ({len(pvals)})")
        for i, m in enumerate(masks):
            if np.asarray(m).size != n_feat:
                raise ValueError(
                    f"{npz_path}: mask[{i}] has {np.asarray(m).size} elems, expected {n_feat}")

    # Gather Present/Absent arrays (mirrors omnibus collapse)
    present, absent, _, _ = _collect_present_absent(
        segment=str(seg).upper(), prep_window="last")
    expected_shape = (len(participants), n_ch, n_f, n_t)
    if present.shape != expected_shape or absent.shape != expected_shape:
        raise RuntimeError(
            f"Data shapes {present.shape} / {absent.shape} != expected {expected_shape}")

    # Make dirs per segment
    seg_upper = str(seg).upper()
    seg_plot_dir = _ensure_dir(plots_dir / seg_upper)
    seg_csv_dir = _ensure_dir(csv_dir / seg_upper)
    csv_path = seg_csv_dir / f"present_vs_absent_{seg_upper}.csv"

    # ensure fresh CSV header (new columns included)
    if csv_path.exists():
        csv_path.unlink()

    # Loop significant omnibus clusters
    n_sig_used = 0  # number of omnibus clusters that produced >=1 significant time window
    for k, (mask1d, pv) in enumerate(zip(masks, pvals)):
        if not (np.isfinite(pv) and pv < ALPHA_REPORT):
            continue
        # Omnibus cluster time window (relative ms), from the omnibus mask.
        # Collapse channel and frequency axes to get the cluster's time extent.
        m3d_omni = np.asarray(mask1d, bool).reshape(n_ch, n_f, n_t)
        omni_time_mask = m3d_omni.any(axis=(0, 1))
        omni_t_rel = times[omni_time_mask]
        omni_t0_ms = float(omni_t_rel.min()) * 1000.0   # window start (ms)
        omni_t1_ms = float(omni_t_rel.max()) * 1000.0   # window end   (ms)

        # ROI time series (mean over ROI channels×freqs), per subject
        y_present = _roi_timeseries_from_mask(
            present, mask1d, n_ch, n_f, n_t)  # (n_subj, n_t)
        y_absent = _roi_timeseries_from_mask(
            absent,  mask1d, n_ch, n_f, n_t)  # (n_subj, n_t)
        diff = y_present - y_absent

        # Time-only paired cluster permutation across the full segment (two-sided)
        T_obs, t_masks_1d, t_pvals, thr_t = _time_cluster_test(
            diff, alpha_cf=ALPHA_CF)

        # --- diagnostics ---
        supra = (np.abs(T_obs) >= thr_t)
        min_p = float(np.nanmin(t_pvals)) if len(t_pvals) else None
        print(f"[{seg_upper} c{k:02d}] thr_t={thr_t:.3f}, max|T|={np.nanmax(np.abs(T_obs)):.3f}, "
              f"n_supra={int(supra.sum())}, min_p={min_p}")
        for j, c1d in enumerate(t_masks_1d):
            if j < len(t_pvals) and np.isfinite(t_pvals[j]) and t_pvals[j] < 0.05:
                idx = np.where(c1d)[0]
                if idx.size:
                    print(
                        f"    win{j}: idx=({idx[0]}..{idx[-1]})  time=({times[idx[0]]:.3f}..{times[idx[-1]]:.3f})  p={t_pvals[j]:.4g}")
        # ------------------------------

        # Significant windows only
        rows = _summarize_sig_windows(times, T_obs, t_masks_1d, t_pvals)
        if not rows:
            # No significant time windows skip plotting and CSV for this cluster
            continue

        n_sig_used += 1

        # CI and plot only significant windows
        meanP, ciP, meanA, ciA = _within_subject_ci_two_cond(
            y_present, y_absent, alpha=0.05)
        # --- compare CSV-style window mean vs plot peak ---
        for r in rows:
            m = (times >= r["time_start"]) & (times <= r["time_end"])

            # CSV-style: per-subject window mean, then group mean
            csv_present = y_present[:, m].mean(axis=1).mean()
            csv_absent = y_absent[:,  m].mean(axis=1).mean()

            # Plot can reach these instantaneous peaks (group mean curve)
            peak_present = meanP[m].max()
            peak_absent = meanA[m].max()

            print(
                f"[{seg_upper} c{str(k).zfill(2)}] "
                f"{r['time_start']:.3f}–{r['time_end']:.3f}s  "
                f"CSV Present={csv_present:.2f}, CSV Absent={csv_absent:.2f}  "
                f"PlotPeak Present={peak_present:.2f}, PlotPeak Absent={peak_absent:.2f}"
            )

        out_png = seg_plot_dir / \
            f"cluster{str(k).zfill(2)}_Present_vs_Absent.png"
        # Only windows fully inside the omnibus window get shaded on the plot.
        rows_in_window = [
            r for r in rows
            if (float(r["time_start"]) * 1000.0 >= omni_t0_ms)
            and (float(r["time_end"]) * 1000.0 <= omni_t1_ms)
        ]
        _plot_lines_with_shading(
            times, meanP, meanA, ciP, ciA, rows_in_window, seg_upper, out_png)

        # Save the plot inputs so the figure can be re-rendered from the pkl.
        import pickle
        pkl_path = out_png.with_name(out_png.stem + "_inputs.pkl")
        with open(pkl_path, "wb") as fh:
            pickle.dump({
                "times": times,
                "yA": meanP, "yB": meanA,
                "ciA": ciP, "ciB": ciA,
                "labelA": "Present", "labelB": "Absent",
                "line_colors": ["#0D5C9D", "#1C7F46"],
                "box_color": "black",
                "sig_windows": rows_in_window,
                "seg": seg_upper,
                "contrast": "Present vs Absent",
            }, fh)

        # ROI summary (channels + freq range) for this omnibus cluster 
        roi_info = _cluster_roi_info(
            mask1d, n_ch, n_f, n_t, freqs, ch_names=ch_names)

        # Build LIGHT-style rows for CSV
        out_rows = []
        for r in rows:
            win_mask = t_masks_1d[r["time_cluster"]]  # internal index
            eff = _window_effect_summary(y_present, y_absent, win_mask)
            # In-window only if both edges sit inside the omnibus window.
            t0_ms = float(r["time_start"]) * 1000.0
            t1_ms = float(r["time_end"]) * 1000.0
            in_window = (t0_ms >= omni_t0_ms) and (t1_ms <= omni_t1_ms)
            out_rows.append({
                "segment": seg_upper,
                "omnibus_cluster": int(k),
                # per-window p-value
                "cluster_p_value": float(r["p_value"]),
                "contrast": "Present vs Absent",
                "t_mass": float(r["t_mass"]),
                "n_timepoints": int(r["n_timepoints"]),
                "t0_rel": float(r["time_start"]) * 1000,
                "t1_rel": float(r["time_end"]) * 1000,
                "roi_n_channels": int(roi_info["roi_n_channels"]),
                "roi_n_freqs": int(roi_info["roi_n_freqs"]),
                "roi_fmin_hz": float(roi_info["freq_min_hz"]),
                "roi_fmax_hz": float(roi_info["freq_max_hz"]),
                # names; fallback indices
                "roi_channels_list": roi_info["roi_channels_list"],
                "mean_A": float(eff["mean_present"]),
                "mean_B": float(eff["mean_absent"]),
                "diff_mean": float(eff["diff_mean"]),
                "diff_ci95_lo": float(eff["diff_ci95_lo"]),
                "diff_ci95_hi": float(eff["diff_ci95_hi"]),
                # main cluster window start (ms)
                "omnibus_t0": omni_t0_ms,
                # main cluster window end   (ms)
                "omnibus_t1": omni_t1_ms,
                # True = main text, False = supplementary
                "in_window": bool(in_window),
            })

        
        COLS = [
            "segment", "omnibus_cluster", "cluster_p_value", "contrast",
            "t_mass", "n_timepoints", "t0_rel", "t1_rel",
            "roi_n_channels", "roi_n_freqs", "roi_fmin_hz", "roi_fmax_hz", "roi_channels_list",
            "mean_A", "mean_B", "diff_mean", "diff_ci95_lo", "diff_ci95_hi", "omnibus_t0", "omnibus_t1", "in_window",
        ]
        df_out = pd.DataFrame(out_rows)[COLS]

        mode = 'a' if csv_path.exists() else 'w'
        header = not csv_path.exists()
        df_out.to_csv(csv_path, index=False, mode=mode, header=header)

    # Meta per segment
    meta = dict(
        npz=str(npz_path),
        segment=seg_upper,
        alpha_report=ALPHA_REPORT,
        alpha_cf=ALPHA_CF,
        n_permutations=N_PERM,
        seed=SEED,
        # clusters that produced sig windows (and plots/rows)
        n_sig_clusters_used=n_sig_used,
    )
    with open(seg_plot_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[DONE] {npz_path.name}: wrote plots/rows for {n_sig_used} clusters → {seg_plot_dir}")


# -------------------- entrypoint --------------------

if __name__ == "__main__":
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_ROOT.mkdir(parents=True, exist_ok=True)

    # PREP
    if not Path(NPZ_PREP).is_file():
        raise FileNotFoundError(f"Missing NPZ: {NPZ_PREP}")
    run_for_npz(NPZ_PREP, PLOT_ROOT, CSV_ROOT)

    # RESET
    if not Path(NPZ_RESET).is_file():
        raise FileNotFoundError(f"Missing NPZ: {NPZ_RESET}")
    run_for_npz(NPZ_RESET, PLOT_ROOT, CSV_ROOT)

    print("All done.")
