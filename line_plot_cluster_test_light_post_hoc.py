# -*- coding: utf-8 -*-

"""
Per-cluster line-plot post-hoc tests for the LIGHT main effect.

- Loads the omnibus (3D) cluster result for a segment (PREP/RESET) and keeps
  only clusters significant at p < 0.05.
- Builds a channel × frequency ROI from each significant omnibus cluster by
  collapsing its time dimension.
- Extracts ROI time series for LIGHT/AMBIENT/DARK (collapsed across the 4 obstacle levels).
- Runs time-only paired cluster-permutation tests (two-sided) for:
    LIGHT vs AMBIENT, LIGHT vs DARK, AMBIENT vs DARK.
- For each significant temporal cluster, writes one CSV row per segment (contrast,
  cluster p-value, t-mass, timing, ROI channels/freqs, condition means and CI) and
  saves a line plot with shaded significant windows. 

Note: the condition level "Light" is displayed as "Bright" in the plots.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from mne.stats import permutation_cluster_1samp_test
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from scripts.cluster_config import (
    participants, base_dir
)
from scripts.extract_prep_reset import (
    load_and_extract_prep_reset, MEDIAN_PREP_DUR
)

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


# ------------------ GLOBAL CONFIG ------------------
ALPHA_CF = 0.001      # cluster-forming alpha (for t-threshold)
ALPHA_CLUST = 0.05    # cluster p-value threshold
N_PERM = 1000
SEED = 42

DEFAULT_SAVE_DIR = Path(r"D:\cluster_outputs")
DEFAULT_PLOT_DIR = Path(r"D:\Cluster_plots_last_prep")

DEFAULT_SEGMENTS = ["PREP", "RESET"]
DEFAULT_OMNIBUS_TPL = "light_main_{segment}_last_prep1000perm_seed42.npz"

LIGHT_LEVELS = ["LIGHT", "AMBIENT", "DARK"]
OBSTACLE_LEVELS = [
    "EXPECTED_PRESENT", "EXPECTED_ABSENT",
    "UNEXPECTED_PRESENT", "UNEXPECTED_ABSENT"
]
# ---------------------------------------------------


# ================== HELPERS ==================
def plot_lines_with_ci_and_sig(times, means, ci95, labels,
                               sig_windows, title, out_path, xlim=None):
    """
    times: (n_time,)
    means, ci95: (n_lines, n_time)
    labels: list[str] for those lines
    sig_windows: list[(t0, t1, npts)]
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for i, lab in enumerate(labels):
        ax.plot(times, means[i], label=lab)
        ax.fill_between(times, means[i] - ci95[i],
                        means[i] + ci95[i], alpha=0.25, linewidth=0)

    ax.axvline(0, linestyle="--", linewidth=1)  # crossing
    for (t0, t1, _) in sig_windows:
        ax.axvspan(t0, t1, alpha=0.15)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Power (% change from baseline)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title)

    # Legend to the RIGHT, without shrinking the axes
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0., frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                bbox_extra_artists=(leg,))
    plt.close(fig)


def _within_subject_ci_cousineau_morey(data_3cond: np.ndarray):
    """data_3cond: (n_subj, 3, n_time) -> (means, ci95) each (3, n_time)"""
    n_subj, k, n_t = data_3cond.shape
    corrected = data_3cond.copy()
    subj_mean = corrected.mean(axis=1, keepdims=True)
    grand_mean = corrected.mean(axis=(0, 1), keepdims=True)
    corrected = corrected - subj_mean + grand_mean
    means = corrected.mean(axis=0)
    sds = corrected.std(axis=0, ddof=1)
    se = sds / np.sqrt(n_subj)
    se *= np.sqrt(k/(k-1))  # Morey correction
    tcrit = t_dist.ppf(0.975, df=n_subj-1)
    ci95 = tcrit * se
    return means, ci95


def _summarize_sig_clusters(time_mask: np.ndarray, times: np.ndarray):
    idx = np.where(time_mask)[0]
    if idx.size == 0:
        return []
    blocks, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        blocks.append((start, prev))
        start = i
        prev = i
    blocks.append((start, prev))
    return [(float(times[a]), float(times[b]), int(b - a + 1)) for a, b in blocks]


def collect_arrays_for_segment(segment="PREP", prep_window="last"):
    """
    Returns (LIGHT, AMBIENT, DARK, freqs, times):
      each array shape: (n_subj, n_ch, n_f, n_t)
      each LIGHT level is the mean across the 4 obstacle variants.
    """
    assert segment in ("PREP", "RESET")
    X_by_light = {k: [] for k in LIGHT_LEVELS}
    n_t_expected = None
    ref_cf = None  # (n_ch, n_freqs)
    freqs = None
    times = None

    for pid in participants:
        per_light = {}
        for L in LIGHT_LEVELS:
            per_obst = []
            for obst in OBSTACLE_LEVELS:
                key = f"GLOBAL_{L}_{obst}"
                if key not in MEDIAN_PREP_DUR:
                    raise RuntimeError(f"{pid} {key}: missing MEDIAN_PREP_DUR")

                prep, reset, t_prep, t_reset, freqs_ax = load_and_extract_prep_reset(
                    pid, key, base_dir=base_dir, prep_window=prep_window
                )
                arr = prep if segment == "PREP" else reset
                t_vec = t_prep if segment == "PREP" else t_reset

                if n_t_expected is None:
                    n_t_expected = arr.shape[-1]
                elif arr.shape[-1] != n_t_expected:
                    raise RuntimeError(
                        f"{pid} {key} {segment} length mismatch {arr.shape[-1]} vs {n_t_expected}"
                    )

                if ref_cf is None:
                    ref_cf = arr.shape[:2]  # (n_ch, n_freqs)
                elif arr.shape[:2] != ref_cf:
                    raise RuntimeError(
                        f"{pid} {key} ch/f shape mismatch {arr.shape[:2]} vs {ref_cf}"
                    )

                if times is None:
                    times = t_vec.copy()
                else:
                    assert len(t_vec) == len(
                        times), f"time length mismatch in {segment}"

                if freqs is None:
                    freqs = freqs_ax.copy()
                else:
                    assert len(freqs_ax) == len(
                        freqs), f"freq length mismatch in {segment}"

                per_obst.append(arr)

            if len(per_obst) != 4:
                raise RuntimeError(
                    f"{pid} {L}: expected 4 obstacle variants, got {len(per_obst)}")
            per_light[L] = np.mean(np.stack(per_obst, axis=0), axis=0)

        missing = [L for L in LIGHT_LEVELS if L not in per_light]
        if missing:
            raise RuntimeError(
                f"{pid}: missing light levels {missing}; aborting.")

        for L in LIGHT_LEVELS:
            X_by_light[L].append(per_light[L])

    X_light = np.stack(X_by_light["LIGHT"],   axis=0)
    X_ambient = np.stack(X_by_light["AMBIENT"], axis=0)
    X_dark = np.stack(X_by_light["DARK"],    axis=0)
    return X_light, X_ambient, X_dark, freqs, times


def _unpack_masks(masks_any, n_feat):
    """
    Return list of 1D boolean masks (length n_feat), one per cluster.
    Accepts:
      - object array of packed items (uint8 arrays or bytes/bytearray)
      - 2D packed matrix (n_clusters, n_bytes)
      - 2D already-unpacked boolean/numeric matrix (n_clusters, n_feat)
      - list-like of masks
    """
    if masks_any is None:
        return []

    n_bytes_expected = (n_feat + 7) // 8  # ceil bytes needed
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
    if arr.ndim == 2 and arr.shape[1] == n_bytes_expected:
        return [
            np.unpackbits(np.asarray(arr[i], dtype=np.uint8))[
                :n_feat].astype(bool)
            for i in range(arr.shape[0])
        ]
    if arr.ndim == 2 and arr.shape[1] == n_feat:
        return [arr[i].astype(bool, copy=False) for i in range(arr.shape[0])]
    try:
        return [np.asarray(m, dtype=bool).ravel()[:n_feat] for m in masks_any]
    except Exception as e:
        raise TypeError(
            f"Cannot parse cluster masks of type {type(masks_any)} "
            f"with shape {getattr(masks_any, 'shape', None)}"
        ) from e


def load_omnibus_npz(npz_path: Path):
    d = np.load(str(npz_path), allow_pickle=True)

    meta = {
        "alpha": float(d["alpha"]),
        "threshold": float(d["threshold"]),
        "n_permutations": int(d["n_permutations"]),
        "seed": int(d["seed"]),
        "n_ch": int(d["n_ch"]),
        "n_freqs": int(d["n_freqs"]),
        "n_times": int(d["n_times"]),
        "ch_names": [str(x) for x in d["ch_names"]],
        "segment": str(d["segment"]),
        "sample_epochs_path": str(d["sample_epochs_path"]),
        "saved_at": str(d["saved_at"]),
    }
    n_feat = meta["n_ch"] * meta["n_freqs"] * meta["n_times"]

    # Rebuild masks robustly
    if "masks_packed" in d.files:
        clusters = _unpack_masks(d["masks_packed"], n_feat)
    elif "masks_packed_uint8" in d.files:
        clusters = _unpack_masks(d["masks_packed_uint8"], n_feat)
    elif "clusters" in d.files:
        clusters = _unpack_masks(d["clusters"], n_feat)
    else:
        clusters = []

    # masks must match header grid
    if clusters:
        m0 = np.asarray(clusters[0], bool)
        if m0.size != n_feat:
            raise RuntimeError(
                f"Cluster mask length {m0.size} != n_ch*n_freqs*n_times ({n_feat})."
            )

    freqs = np.asarray(d["freqs"], dtype=float)
    times = np.asarray(d["times"], dtype=float)
    F_obs = np.asarray(d["F_obs"])
    p_vals = np.asarray(d["p_vals"], dtype=float)
    return F_obs, clusters, p_vals, meta, freqs, times


def run_temporal_cluster_test(D, t_thr, n_perm, seed):
    """
    D: (n_subj, n_time) paired differences
    Returns: T_obs (n_time,), masks_1d (list of (n_time,) bool), p_raw, H0
    """
    T_obs, cl_inds, p_raw, H0 = permutation_cluster_1samp_test(
        D,
        threshold=t_thr,
        tail=0,                 # two-sided
        adjacency=None,         # cluster along time only
        n_permutations=n_perm,
        out_type="indices",     # robust across MNE versions
        n_jobs=7,
        seed=seed,
        verbose=False
    )

    n_time = D.shape[-1]
    masks = []
    for idx in cl_inds:
        if isinstance(idx, (list, tuple)):
            parts = []
            for part in idx:
                a = np.asarray(part)
                if np.issubdtype(a.dtype, np.integer):
                    parts.append(a.ravel())
            idx_flat = np.unique(np.concatenate(parts)) if parts else np.asarray(
                idx, dtype=int).ravel()
        else:
            idx_flat = np.asarray(idx, dtype=int).ravel()
        m = np.zeros(n_time, dtype=bool)
        if idx_flat.size:
            m[idx_flat] = True
        masks.append(m)
    return T_obs, masks, p_raw, H0


# ================== CORE WORK ==================
def run_line_posthocs_for_segment(save_dir: Path, segment: str, omnibus_npz_path: Path):
    """Create ONE lean CSV per segment with time-only cluster results (t-mass), and plots."""
    print(f"\n=== {segment}: loading omnibus results ===")
    F_obs, clusters, p_vals, meta, freqs, times = load_omnibus_npz(
        omnibus_npz_path)

    p_vals = np.asarray(p_vals, float)
    sig_inds = [i for i, p in enumerate(
        p_vals) if np.isfinite(p) and p < ALPHA_CLUST]
    if not sig_inds:
        print(f"[INFO] No significant omnibus clusters in {segment}.")
        # still write an empty CSV to keep outputs predictable
        out_dir = save_dir / f"light_posthoc"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_csv = out_dir / f"light_posthoc_report_{segment}.csv"
        pd.DataFrame([]).to_csv(report_csv, index=False)
        print(f"[SAVED] {report_csv} (empty)")
        return

    # Load data collapsed across obstacle for each LIGHT level
    X_light, X_ambient, X_dark, freqs2, times2 = collect_arrays_for_segment(
        segment=segment, prep_window="last"
    )
    n_subj, n_ch, n_f_cur, n_t_cur = X_light.shape
    n_f_omni, n_t_omni = len(freqs), len(times)

    # time axis and center (0 = crossing)
    times_use = times if len(times) == n_t_cur else times2
    anchor = times_use[-1] if segment == "PREP" else times_use[0]
    t_plot = times_use - anchor
    xlim = (-2.7, 0.0) if segment == "PREP" else (0.0, 1.0)
    # after compute t_plot and xlim
    t_plot = t_plot * 1000.0
    xlim = (xlim[0] * 1000.0, xlim[1] * 1000.0)

    # two-sided t threshold for time-only post-hocs
    df = n_subj - 1
    t_thr = float(t_dist.ppf(1 - ALPHA_CF / 2.0, df=df))
    seed = int(meta.get("seed", SEED))

    out_dir = save_dir / 'line_posthocs' / f"light_posthocs_{segment}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = DEFAULT_PLOT_DIR / "line_plots" / f'light_post_hoc_{segment}'
    plot_dir.mkdir(parents=True, exist_ok=True)

    master_rows = []

    contrasts = [
        ("LIGHT_vs_AMBIENT", "LIGHT", "AMBIENT"),
        ("LIGHT_vs_DARK",    "LIGHT", "DARK"),
        ("AMBIENT_vs_DARK",  "AMBIENT", "DARK"),
    ]
    label_to_array = {"LIGHT": X_light, "AMBIENT": X_ambient, "DARK": X_dark}

    for k in sig_inds:
        print(f"\n-- Omnibus Cluster {k} (p={p_vals[k]:.4g}) --")

        # Build ch×freq ROI from the omnibus cluster (collapse time)
        n_feat = n_ch * n_f_omni * n_t_omni
        m1d = np.asarray(clusters[k], dtype=bool)
        if m1d.size != n_feat:
            raise RuntimeError(
                f"Omnibus mask size {m1d.size} != expected {n_feat}")
        m3d = m1d.reshape(n_ch, n_f_omni, n_t_omni)

        # m3d is the main-effect cluster mask, shaped (channels, frequencies, time).
        # Collapsing the channel and frequency axes leaves one True/False per time
        # point: True wherever this main cluster exists in time.
        omni_time_mask = m3d.any(axis=(0, 1))

        # The main cluster's window runs from its earliest to its latest time point.
        omni_t_abs = times[omni_time_mask]

        # Express those two edges in the same milliseconds scale the post-hocs use.
        omni_t0_ms = (float(omni_t_abs.min()) - anchor) * \
            1000.0   # window start (ms)
        omni_t1_ms = (float(omni_t_abs.max()) - anchor) * \
            1000.0   # window end   (ms)

        roi_chxf = m3d.any(axis=2)   # (n_ch, n_freq)

        roi_ch_idx = np.flatnonzero(roi_chxf.any(axis=1))
        roi_f_idx = np.flatnonzero(roi_chxf.any(axis=0))
        if roi_chxf.sum() == 0:
            print(
                f"[WARN] Cluster {k}: empty ch×freq ROI after time collapse; skipping.")
            continue

        roi_freqs_hz = np.asarray(freqs)[roi_f_idx]
        fmin, fmax = float(roi_freqs_hz.min()), float(roi_freqs_hz.max())
        roi_ch_names = [meta["ch_names"][i] for i in roi_ch_idx]
        roi_channels_list = ";".join(roi_ch_names)

        # ROI time series per condition: (n_subj, n_time)
        series = {
            "LIGHT":   (label_to_array["LIGHT"] * roi_chxf[None, :, :, None]).sum(axis=(1, 2)) / roi_chxf.sum(),
            "AMBIENT": (label_to_array["AMBIENT"] * roi_chxf[None, :, :, None]).sum(axis=(1, 2)) / roi_chxf.sum(),
            "DARK":    (label_to_array["DARK"] * roi_chxf[None, :, :, None]).sum(axis=(1, 2)) / roi_chxf.sum(),
        }

        # Means & within-subject 95% CI (for plotting)
        Y3 = np.stack([series["LIGHT"], series["AMBIENT"],
                      series["DARK"]], axis=1)  # (n_subj, 3, n_time)
        means, ci95 = _within_subject_ci_cousineau_morey(Y3)

        # Collect windows for combined plot (per-contrast, with p-values)
        wins_map = {}

        # Time-only paired cluster tests for each contrast
        for cname, Aname, Bname in contrasts:
            A = series[Aname]  # (n_subj, n_time)
            B = series[Bname]
            D = A - B

            # check last feature is time
            print(f"{segment} | Cluster {k} | {cname}")
            print("A shape:", A.shape)
            print("B shape:", B.shape)
            print("D shape:", D.shape)

            T_obs, cl_masks, p_raw, _ = run_temporal_cluster_test(
                D, t_thr, N_PERM, seed)

            # collect shaded windows for the 2-line plot of THIS contrast
            wins_rel_all = []

            for m, p in zip(cl_masks, p_raw):
                if np.isfinite(p) and p < ALPHA_CLUST:
                    wins_abs = _summarize_sig_clusters(m, times_use)
                    if not wins_abs:
                        continue

                    for (t0_abs, t1_abs, npts_win) in wins_abs:
                        # window-specific mask
                        m_win = (times_use >= t0_abs) & (
                            times_use <= t1_abs) & m

                        # relative times (PREP anchored at 0 = end of PREP; RESET at 0 = start of RESET)
                        t0_rel = float(t0_abs - anchor)
                        t1_rel = float(t1_abs - anchor)

                        # cluster mass and n_timepoints for THIS window only
                        t_mass = float(T_obs[m_win].sum())
                        n_pts = int(m_win.sum())

                        # per-subject window means and paired diffs
                        A_win = A[:, m_win].mean(axis=1)    # (n_subj,)
                        B_win = B[:, m_win].mean(axis=1)    # (n_subj,)
                        D_win = A_win - B_win               # (n_subj,)

                        # group stats
                        diff_mean = float(D_win.mean())
                        se = float(D_win.std(ddof=1) / np.sqrt(n_subj))
                        tcrit = float(t_dist.ppf(0.975, df=n_subj-1))
                        diff_ci95_lo = diff_mean - tcrit * se
                        diff_ci95_hi = diff_mean + tcrit * se
                        mean_A = float(A_win.mean())
                        mean_B = float(B_win.mean())

                        # Post-hoc window edges, same millisecond scale.
                        t0_ms = t0_rel * 1000.0
                        t1_ms = t1_rel * 1000.0

                        # In-window only if BOTH edges sit inside the main cluster window.
                        
                        in_window = (t0_ms >= omni_t0_ms) and (
                            t1_ms <= omni_t1_ms)
                        # Only in-window post-hocs get a dashed rectangle on the plots.
                        # Out-of-window ones are still written to the CSV, just not drawn.
                        if in_window:
                            wins_rel_all.append((t0_ms, t1_ms, npts_win))
                            wins_map.setdefault(cname, []).append(
                                (t0_ms, t1_ms, npts_win, float(p))
                            )

                        master_rows.append({
                            "segment": segment,
                            "omnibus_cluster": int(k),
                            "cluster_p_value": float(p),
                            "contrast": cname,             # e.g., LIGHT_vs_AMBIENT
                            "t_mass": t_mass,              # signed sum of t’s within THIS window
                            "n_timepoints": int(n_pts),
                            # PREP: [-2.7..0], RESET: [0..1]
                            "t0_rel": t0_rel * 1000,
                            "t1_rel": t1_rel * 1000,
                            "roi_n_channels": int(len(roi_ch_idx)),
                            "roi_n_freqs": int(len(roi_f_idx)),
                            "roi_fmin_hz": fmin,
                            "roi_fmax_hz": fmax,
                            "roi_channels_list": roi_channels_list,
                            "mean_A": mean_A,
                            "mean_B": mean_B,
                            "diff_mean": diff_mean,
                            "diff_ci95_lo": diff_ci95_lo,
                            "diff_ci95_hi": diff_ci95_hi,
                            # main cluster window start (ms)
                            "omnibus_t0": omni_t0_ms,
                            # main cluster window end   (ms)
                            "omnibus_t1": omni_t1_ms,
                            # True = main text, False = supplementary
                            "in_window": bool(in_window),
                        })

            # Make the plot for this contrast (two lines + shaded wins)
            idx_map = {"LIGHT": 0, "AMBIENT": 1, "DARK": 2}
            sel = [idx_map[Aname], idx_map[Bname]]
            means_sel = means[sel]
            ci_sel = ci95[sel]
            labels_sel = [Aname, Bname]

            fig_path = plot_dir / f"cluster{k}_{cname.replace(' ', '_')}.png"
            plot_title = (
                f"{segment} — Cluster {k}: {Aname} vs {Bname}\n"
                f"ROI: {len(roi_ch_idx)} ch × {len(roi_f_idx)} freq "
                f"({fmin:.1f}–{fmax:.1f} Hz); α={ALPHA_CLUST}, perms={N_PERM}"
            )
            if wins_rel_all:
                plot_lines_with_ci_and_sig(
                    times=t_plot,
                    means=means_sel,
                    ci95=ci_sel,
                    labels=labels_sel,
                    sig_windows=wins_rel_all,
                    title=plot_title,
                    out_path=fig_path,
                    xlim=xlim,
                )
                print(f"[SAVED] {fig_path}")
            else:
                print(
                    f"[SKIP] Cluster {k} {cname}: no significant time windows — plot not generated.")

        sig_map = {
            "LIGHT_vs_AMBIENT": wins_map.get("LIGHT_vs_AMBIENT", []),
            "LIGHT_vs_DARK":    wins_map.get("LIGHT_vs_DARK", []),
            "AMBIENT_vs_DARK":  wins_map.get("AMBIENT_vs_DARK", []),
        }
        combined_path = plot_dir / f"cluster{k}_ALL_THREE.png"
        # Save the plot inputs so the figure can be adjusted without re-running the analysis.
        import pickle
        with open(plot_dir / f"cluster{k}_ALL_THREE_inputs.pkl", "wb") as fh:
            pickle.dump({
                "times": t_plot, "means3": means, "ci95_3": ci95,
                "labels3": ["LIGHT", "AMBIENT", "DARK"],
                "sig_windows_map": sig_map, "xlim": xlim,
            }, fh)

        plot_three_with_contrast_boxes(
            times=t_plot, means3=means, ci95_3=ci95,
            labels3=["LIGHT", "AMBIENT", "DARK"],
            sig_windows_map=sig_map, title=(f"{segment} — Cluster {k}: All LIGHT levels\n"
                                            f"ROI: {len(roi_ch_idx)} ch × {len(roi_f_idx)} freq "
                                            f"({fmin:.1f}–{fmax:.1f} Hz); α={ALPHA_CLUST}, perms={N_PERM}"),
            out_path=combined_path, xlim=xlim, annotate_p=False
        )
    # write exactly ONE CSV for this segment (even if empty)
    report_csv = out_dir / f"light_posthoc_report_{segment}.csv"
    pd.DataFrame(master_rows).to_csv(report_csv, index=False)
    print(f"[SAVED] {report_csv}")


def _pretty(s: str) -> str:
    # "LIGHT" -> "Bright", "AMBIENT" -> "Ambient", "LIGHT_vs_DARK" -> "Bright vs Dark"
    parts = s.split("_vs_")
    left = parts[0].replace("_", " ").title()
    if len(parts) == 2:
        right = parts[1].replace("_", " ").title()
        out = f"{left} vs {right}"
    else:
        out = left
    # Display only: the condition level "Light" is shown as "Bright"
    return out.replace("Light", "Bright")


def plot_three_with_contrast_boxes(times, means3, ci95_3, labels3,
                                   sig_windows_map, title, out_path,
                                   xlim=None, box_linestyle=(0, (2, 1)),
                                   box_linewidth=1.2, contrast_colors=None,
                                   annotate_p=False, box_pad_frac=0.02,
                                   fill_alpha=0.10,  # compat only; no fill used
                                   inset_frac_by_contrast=None):
    """
    Three-line plot with CI and dashed-outline boxes for significant windows.
    
    """
    if contrast_colors is None:
        contrast_colors = {
            "LIGHT_vs_AMBIENT": "#D11A2A",
            "LIGHT_vs_DARK":    "#6B3F12",   # avoid green/orange/blue
            "AMBIENT_vs_DARK":  "#1B8A3A",
        }
    if inset_frac_by_contrast is None:
        inset_frac_by_contrast = {
            "LIGHT_vs_DARK":    0.00,   # largest box
            "AMBIENT_vs_DARK":  0.020,
            "LIGHT_vs_AMBIENT": 0.040,  # smallest so borders don’t overlap
        }
    contrast_line_idx = {
        "LIGHT_vs_AMBIENT": [0, 1],
        "LIGHT_vs_DARK":    [0, 2],
        "AMBIENT_vs_DARK":  [1, 2],
    }

    
    # half-width tile, taller to fit the legend below
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.0))

    # Lines + CI
    line_handles = []
    line_colors = ["#7758A3", "#D4763C", "#4D87A8"]
    for i, lab in enumerate(labels3):
        ln, = ax.plot(times, means3[i], label=_pretty(
            lab), color=line_colors[i], zorder=2)
        line_handles.append(ln)
        if ci95_3 is not None:
            ax.fill_between(times, means3[i] - ci95_3[i],
                            means3[i] + ci95_3[i], alpha=0.25,
                            color=line_colors[i],
                            linewidth=0, zorder=1.5)

    ax.axvline(0, linestyle="--", linewidth=1, zorder=2)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel("Time (ms)", labelpad=10)

    ax.set_ylabel("Power (% change from baseline)", labelpad=10)

    # Dashed-edge rectangles per contrast/window (NO FILL)
    draw_order = ("LIGHT_vs_DARK", "AMBIENT_vs_DARK", "LIGHT_vs_AMBIENT")
    for key in draw_order:
        color = contrast_colors[key]
        inset_frac = float(inset_frac_by_contrast.get(key, 0.0))
        for win in sig_windows_map.get(key, []):
            # accept (t0,t1,...) or dict with keys t0/t1
            if isinstance(win, dict):
                t0 = float(win.get("t0", win.get("time_start")))
                t1 = float(win.get("t1", win.get("time_end")))
                p = win.get("p", win.get("p_value"))
            else:
                t0, t1 = float(win[0]), float(win[1])
                p = float(win[3]) if (isinstance(
                    win, (list, tuple)) and len(win) >= 4) else None

            mask = (times >= t0) & (times <= t1)
            if not np.any(mask):
                continue

            # if ci95_3 is not None:
            #     win_min = np.nanmin((means3 - ci95_3)[:, mask])
            #     win_max = np.nanmax((means3 + ci95_3)[:, mask])
            # else:
            #     win_min = np.nanmin(means3[:, mask])
            #     win_max = np.nanmax(means3[:, mask])
            sel = contrast_line_idx.get(key, list(range(len(labels3))))
            if ci95_3 is not None:
                win_min = np.nanmin((means3 - ci95_3)[sel][:, mask])
                win_max = np.nanmax((means3 + ci95_3)[sel][:, mask])
            else:
                win_min = np.nanmin(means3[sel][:, mask])
                win_max = np.nanmax(means3[sel][:, mask])

            pad = (win_max - win_min) * box_pad_frac
            y0 = win_min - pad
            height = (win_max - win_min) + 2 * pad
            width = t1 - t0

            x_in = width * inset_frac
            y_in = height * inset_frac
            t0i, width_i = t0 + x_in, max(width - 2 * x_in, 1e-9)
            y0i, height_i = y0 + y_in, max(height - 2 * y_in, 1e-9)

           
            x_left, x_right = t0i, t0i + width_i
            y_bot, y_top = y0i, y0i + height_i
            box_edges = [
                ([x_left, x_right], [y_bot, y_bot]),   # bottom
                ([x_left, x_right], [y_top, y_top]),   # top
                ([x_left, x_left], [y_bot, y_top]),    # left
                ([x_right, x_right], [y_bot, y_top]),  # right
            ]
            for ex, ey in box_edges:
                ax.plot(ex, ey, color=color, linestyle=box_linestyle,
                        linewidth=box_linewidth, alpha=0.95, zorder=2.8,
                        clip_on=False, dash_capstyle="butt")

            if annotate_p and (p is not None):
                ax.text(t0i + width_i/2.0, y0i + height_i,
                        f"p={p:.3f}", ha="center", va="bottom",
                        fontsize=8, color=color, zorder=3.0)

    
    # Legends on the RIGHT (stacked, no titles)
    cond_leg = ax.legend(
        handles=line_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),   # top legend
        borderaxespad=0.,
        frameon=False,
        title=None,                   
        ncol=1,
        handlelength=2.0,

    )
    ax.add_artist(cond_leg)

    non_empty = [k for k in draw_order if len(sig_windows_map.get(k, [])) > 0]
    contr_leg = None
    if non_empty:
        # Legend labels without the "Sig window:" prefix
        # "Light vs Ambient", etc.
        label_map = {k: _pretty(k) for k in non_empty}

        proxies = [
            Line2D([0], [0],
                   color=contrast_colors[k],
                   linestyle=box_linestyle, linewidth=box_linewidth,
                   label=label_map.get(k, k.replace("_vs_", " vs ").replace("_", " ")))
            for k in non_empty
        ]

        contr_leg = ax.legend(
            handles=proxies,
            loc="upper left",
            # stacked below the conditions legend
            bbox_to_anchor=(1.02, 0.62),
            borderaxespad=0.,
            frameon=False,
            title=None,                    # no "Contrasts" title
            ncol=1,
            handlelength=2.0,

        )

    fig.tight_layout()
    extra = (cond_leg,) if contr_leg is None else (cond_leg, contr_leg)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                bbox_extra_artists=extra)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Per-cluster line-plot post-hoc tests for LIGHT main effect."
    )
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--segments", nargs="+", default=DEFAULT_SEGMENTS,
                        help="Segments to process (e.g., PREP RESET)")
    parser.add_argument("--omnibus_tpl", type=str, default=DEFAULT_OMNIBUS_TPL,
                        help="Filename template for omnibus NPZ; {segment} will be substituted.")
    args = parser.parse_args()

    for seg in args.segments:
        npz_path = args.save_dir / args.omnibus_tpl.format(segment=seg)
        if not npz_path.exists():
            print(f"[WARN] Omnibus NPZ not found: {npz_path}")
            # still drop an empty CSV to keep artifacts consistent
            out_dir = args.save_dir / f"line_posthocs_{seg}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"light_posthoc_report_{seg}.csv").write_text("")
            continue
        run_line_posthocs_for_segment(
            args.save_dir, seg, npz_path)


if __name__ == "__main__":
    main()
