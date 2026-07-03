"""Age main-effect cluster permutation test (Young vs Old) on ERSP for the PREP and RESET segments"""

import os
import time
import numpy as np
import mne
from scipy.stats import t
import matplotlib.pyplot as plt
import pandas as pd  

from mne.stats import combine_adjacency, permutation_cluster_test, ttest_ind_no_p
from functools import partial
from extract_prep_reset import MEDIAN_PREP_DUR, load_and_extract_prep_reset
from extract_prep_reset import load_baseline_bands_from_rawdb
from cluster_config import (
    participants, Young, Old,
    base_dir, aligned_epochs_dir,
    sample_pid, sample_condition,
    raw_tfr_dir,

)

SAVE_DIR = r"D:\cluster_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)


# --- Alpha levels ---
ALPHA_CF = 0.001   # cluster-forming alpha 
ALPHA_CLUST = 0.05  # cluster-level/reporting alpha


def save_cluster_result_npz_age(
    path, *, segment, T_obs, clusters, p_vals,
    alpha_cf, alpha_clust, threshold, n_permutations, seed,
    ch_names, n_ch, n_freqs, n_times,
    freqs, times, sample_epochs_path,
):
    """
    Save cluster-permutation result (independent t-test) with axes.
    - T_obs: 1D array of t-values over flattened features (ch×freq×time)
    - clusters: list of boolean masks (from out_type='mask'), each length n_feat
    """
    n_feat = int(n_ch) * int(n_freqs) * int(n_times)

    T_obs = np.asarray(T_obs).ravel()
    assert T_obs.size == n_feat, f"T map has {T_obs.size} features, expected {n_feat}"

    # pack cluster masks to a (n_clusters, n_feat) boolean array
    if clusters:
        masks_packed = np.vstack(
            [np.asarray(m, dtype=bool).ravel() for m in clusters])
    else:
        masks_packed = np.zeros((0, n_feat), dtype=bool)

    np.savez_compressed(
        path,
        # primary stats
        segment=np.array(segment),
        T_obs=T_obs,
        p_vals=np.asarray(p_vals),
        masks_packed=masks_packed,
        # thresholds & meta
        threshold=np.float64(threshold),
        alpha_cf=np.float64(alpha_cf),
        alpha_clust=np.float64(alpha_clust),
        n_permutations=np.int32(n_permutations),
        seed=np.int32(seed),
        # shape & labels
        n_ch=np.int32(n_ch), n_freqs=np.int32(n_freqs), n_times=np.int32(n_times),
        ch_names=np.array(ch_names, dtype=object),
        freqs=np.asarray(freqs),
        times=np.asarray(times),
        sample_epochs_path=np.array(sample_epochs_path, dtype=object),
        flatten_axes=np.array(["ch", "freq", "time"], dtype=object),
        saved_at=np.array(time.strftime("%Y-%m-%d %H:%M:%S")),
    )


def run_age(participants, young_set, old_set, base_dir, aligned_epochs_dir, sample_pid, sample_condition,
            n_permutations=1000, alpha=0.05, seed=42, n_jobs=7):

    # --- preflight: every RAW dB file must exist ---
    for pid in participants:
        for cond_key in MEDIAN_PREP_DUR.keys():
            base = f"TFR_{cond_key}_ALL"
            pdir = os.path.join(raw_tfr_dir, pid)
            assert os.path.exists(os.path.join(
                pdir, f"{base}-RAWDB.npy")),  f"Missing RAWDB for {pid} {cond_key}"
            assert os.path.exists(os.path.join(
                pdir, f"{base}-times.npy")),  f"Missing times for {pid} {cond_key}"
            assert os.path.exists(os.path.join(
                pdir, f"{base}-freqs.npy")),  f"Missing freqs for {pid} {cond_key}"

    # --- averaged PREP/RESET per subject for Young and Old ---
    prep_all_y, prep_all_o = [], []
    reset_all_y, reset_all_o = [], []
    baseline_rows = []

    N_PREP = None
    N_RESET = None
    ref_shape = None  # (n_channels, n_freqs)

    # axes holders (set once)
    freqs_global = None
    times_prep_global = None
    times_reset_global = None

    def load_epochs(pid, condition):
        cond_safe = condition.replace(" ", "_").replace("/", "_")
        filepath = os.path.join(aligned_epochs_dir, pid,
                                cond_safe, f"aligned_epochs_{cond_safe}.fif")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Epochs file not found: {filepath}")
        return mne.read_epochs(filepath, preload=False)

    # Build channel adjacency from sample PID epochs
    sample_epochs = load_epochs(sample_pid, sample_condition)
    ch_adjacency, ch_names = mne.channels.find_ch_adjacency(
        sample_epochs.info, ch_type='eeg')

    # ---- Collect and average within each participant across ALL conditions ----
    for pid in participants:
        try:
            per_pid_prep = []   # PREP segments for this PID across all conditions
            per_pid_reset = []  # RESET segments for this PID across all conditions
            per_pid_baseline_alpha = []
            per_pid_baseline_theta = []
            per_pid_baseline_beta = []

            for cond_key in MEDIAN_PREP_DUR.keys():

                # KEEP the times and freqs
                prep, reset, t_prep, t_reset, freqs = load_and_extract_prep_reset(
                    pid, cond_key, base_dir, prep_window="last"  # last prep window
                )

                # --- baseline (uncorrected) from RAW dB for the same pid/cond ---
                bl_bands, freqs_raw, times_raw = load_baseline_bands_from_rawdb(
                    pid, cond_key, dir_raw=raw_tfr_dir, baseline=(0.0, 2.0)
                )
                # assert ERSP vs RAW frequency axes match
                # # Align RAW-dB and ERSP frequency axes: pick RAW bins in [FREQ_MIN, FREQ_MAX]
                # and verify they match the ERSP freq array (same bins & order).

                
                assert np.allclose(
                    freqs_raw, freqs), f"{pid} {cond_key}: ERSP/RAW freqs mismatch"
                # After: assert np.allclose(freqs_raw, freqs), ...
                per_pid_baseline_alpha.append(float(bl_bands["alpha"].mean()))
                per_pid_baseline_theta.append(float(bl_bands["theta"].mean()))
                per_pid_baseline_beta.append(float(bl_bands["beta"].mean()))

                # set canonical axes once to makse sure everyone used same freq axis and segment length
                if freqs_global is None:
                    freqs_global = freqs.copy()
                    times_prep_global = t_prep.copy()
                    times_reset_global = t_reset.copy()
                    print(
                        f"[FREQ] Analysis freq axis: {freqs_global[0]:.2f}–{freqs_global[-1]:.2f} Hz (Nfreq={len(freqs_global)})")

                else:
                    # same freqs; same segment lengths must match
                    assert np.allclose(
                        freqs_global, freqs), f"{pid} {cond_key}: different freqs axis"
                    assert len(t_prep) == len(
                        times_prep_global),  f"{pid} {cond_key}: PREP time length differs"
                    assert len(t_reset) == len(
                        times_reset_global), f"{pid} {cond_key}: RESET time length differs"

                # set lengths ONCE from the first loaded condition
                if N_PREP is None:
                    N_PREP = prep.shape[-1]
                if N_RESET is None:
                    N_RESET = reset.shape[-1]

                # per-condition checks
                assert prep.shape[-1] == N_PREP, f"{pid} {cond_key}: unexpected PREP length {prep.shape[-1]}"
                if ref_shape is None:
                    ref_shape = prep.shape[:2]  # (n_channels, n_freqs)
                else:
                    assert prep.shape[:2] == ref_shape, f"{pid} {cond_key}: shape {prep.shape[:2]} != {ref_shape}"

                per_pid_prep.append(prep)
                per_pid_reset.append(reset)

            # fail if no conditions loaded
            assert len(per_pid_prep) > 0, f"No conditions loaded for {pid}"

            # average across all conditions for this participant
            pid_mean_prep = np.mean(
                np.stack(per_pid_prep,  axis=0), axis=0)  # (ch, f, t_prep)
            pid_mean_reset = np.mean(
                np.stack(per_pid_reset, axis=0), axis=0)  # (ch, f, t_reset)

            # ---per-participant baseline summary (mean across conditions)
            bl_alpha_pid = np.mean(
                per_pid_baseline_alpha) if per_pid_baseline_alpha else np.nan
            bl_theta_pid = np.mean(
                per_pid_baseline_theta) if per_pid_baseline_theta else np.nan
            bl_beta_pid = np.mean(
                per_pid_baseline_beta) if per_pid_baseline_beta else np.nan
            # convert dB (reference = V²) - dB (reference = µV²)
            alpha_db_uv = bl_alpha_pid + 120
            theta_db_uv = bl_theta_pid + 120
            beta_db_uv = bl_beta_pid + 120
            # --- append one baseline row per participant ---
            group = "Young" if pid in young_set else (
                "Old" if pid in old_set else "Other")
            baseline_rows.append(dict(
                PID=pid,
                group=group,
                baseline_start_s=0.0,
                baseline_end_s=2.0,
                alpha_db=float(bl_alpha_pid),
                theta_db=float(bl_theta_pid),
                beta_db=float(bl_beta_pid),
                alpha_db_uv=float(alpha_db_uv),
                theta_db_uv=float(theta_db_uv),
                beta_db_uv=float(beta_db_uv),
            ))

            # assign to Young / Old bins
            if pid in young_set:
                prep_all_y.append(pid_mean_prep)
                reset_all_y.append(pid_mean_reset)
            elif pid in old_set:
                prep_all_o.append(pid_mean_prep)
                reset_all_o.append(pid_mean_reset)
            else:
                raise ValueError(f"[ERROR] {pid} not assigned to Young or Old")

        except Exception as e:
            print(f"[ERROR] {pid}: {e}")
            raise

    # stack to arrays: (n_subjects, n_channels, n_freqs, n_times)
    prep_all_y = np.stack(prep_all_y,  axis=0)
    prep_all_o = np.stack(prep_all_o,  axis=0)
    reset_all_y = np.stack(reset_all_y, axis=0)
    reset_all_o = np.stack(reset_all_o, axis=0)

    print("PREP Young:",  prep_all_y.shape)
    print("PREP Old:  ",  prep_all_o.shape)
    print("RESET Young:", reset_all_y.shape)
    print("RESET Old:  ",  reset_all_o.shape)

    # sync checks
    assert reset_all_y.shape[-1] == N_RESET and reset_all_o.shape[-1] == N_RESET, "Reset time length mismatch"

    # PREP combined adjacency (channels × freqs × times)
    _, n_channels_p, n_freqs_p, n_times_prep = prep_all_y.shape
    assert prep_all_o.shape[1:] == (n_channels_p, n_freqs_p, n_times_prep)
    adjacency_prep = combine_adjacency(ch_adjacency, n_freqs_p, n_times_prep)
    print("adjacency_prep shape:", adjacency_prep.shape)

    # RESET combined adjacency
    _, n_channels_r, n_freqs_r, n_times_reset = reset_all_y.shape
    assert (n_channels_r, n_freqs_r) == (n_channels_p, n_freqs_p)
    assert reset_all_o.shape[1:] == (n_channels_r, n_freqs_r, n_times_reset)
    adjacency_reset = combine_adjacency(ch_adjacency, n_freqs_r, n_times_reset)
    print("adjacency_reset shape:", adjacency_reset.shape)

    # ---- Flatten for cluster test ----
    Xy_prep = prep_all_y.reshape(prep_all_y.shape[0], -1)
    Xo_prep = prep_all_o.reshape(prep_all_o.shape[0], -1)
    Xy_reset = reset_all_y.reshape(reset_all_y.shape[0], -1)
    Xo_reset = reset_all_o.reshape(reset_all_o.shape[0], -1)

    # essential adjacency vs feature checks
    assert adjacency_prep.shape[0] == Xy_prep.shape[1] == Xo_prep.shape[1]
    assert adjacency_reset.shape[0] == Xy_reset.shape[1] == Xo_reset.shape[1]

    #  NaN/Inf check
    assert np.isfinite(Xy_prep).all() and np.isfinite(
        Xo_prep).all(), "PREP: NaNs/Infs in data"
    assert np.isfinite(Xy_reset).all() and np.isfinite(
        Xo_reset).all(), "RESET: NaNs/Infs in data"

    # ----------  visualize adjacency once ----------
    mne.viz.plot_ch_adjacency(
        sample_epochs.info, ch_adjacency, ch_names, kind='2d')

    # independent-samples t-test with two-sided t-thresholds
   

    student_stat_fun = partial(ttest_ind_no_p, equal_var=True)

    df_prep = Xy_prep.shape[0] + Xo_prep.shape[0] - 2
    df_reset = Xy_reset.shape[0] + Xo_reset.shape[0] - 2
    t_thr_prep = float(t.ppf(1 - ALPHA_CF/2, df_prep))    # two-sided CF alpha
    t_thr_reset = float(t.ppf(1 - ALPHA_CF/2, df_reset))  # two-sided CF alpha
    print(
        f"t-thresholds (two-sided, CF alpha={ALPHA_CF}): PREP={t_thr_prep:.3f} (df={df_prep}), RESET={t_thr_reset:.3f} (df={df_reset})")

    # ---- PREP: cluster permutation (Young vs Old) ----
    T_prep, clusters_prep, p_prep, H0_prep = permutation_cluster_test(
        [Xy_prep, Xo_prep],
        stat_fun=student_stat_fun,
        threshold=t_thr_prep,
        tail=0,  # two-sided test
        adjacency=adjacency_prep,
        n_permutations=n_permutations,
        out_type='mask',  # boolean masks
        n_jobs=n_jobs,
        seed=seed,
        verbose=True,
    )
    print(
        f"PREP: {len(clusters_prep)} clusters, min p = {p_prep.min() if len(p_prep) else None}")

    # ---- Save PREP result ----
    cond_safe = sample_condition.replace(" ", "_").replace("/", "_")
    sample_epochs_path = os.path.join(
        aligned_epochs_dir, sample_pid, cond_safe, f"aligned_epochs_{cond_safe}.fif")
    prep_out = os.path.join(
        SAVE_DIR, f"age_main_PREP_ttest_last_prep{n_permutations}perm_seed{seed}.npz")
    save_cluster_result_npz_age(
        path=prep_out,
        segment="PREP",
        T_obs=T_prep,
        clusters=clusters_prep,
        p_vals=p_prep,
        alpha_cf=ALPHA_CF,
        alpha_clust=ALPHA_CLUST,
        threshold=t_thr_prep,
        n_permutations=n_permutations,
        seed=seed,
        ch_names=sample_epochs.info["ch_names"],
        n_ch=n_channels_p, n_freqs=n_freqs_p, n_times=n_times_prep,
        freqs=freqs_global, times=times_prep_global,
        sample_epochs_path=sample_epochs_path,
    )

    # ---- RESET: cluster permutation (Young vs Old) ----
    T_reset, clusters_reset, p_reset, H0_reset = permutation_cluster_test(
        [Xy_reset, Xo_reset],
        stat_fun=student_stat_fun,
        threshold=t_thr_reset,
        tail=0,  # two-sided
        adjacency=adjacency_reset,
        n_permutations=n_permutations,
        out_type='mask',
        n_jobs=n_jobs,
        seed=seed,
        verbose=True,
    )
    print(
        f"RESET: {len(clusters_reset)} clusters, min p = {p_reset.min() if len(p_reset) else None}")

    # ---- Save RESET result ----
    reset_out = os.path.join(
        SAVE_DIR, f"age_main_RESET_ttest_last_prep{n_permutations}perm_seed{seed}.npz")
    save_cluster_result_npz_age(
        path=reset_out,
        segment="RESET",
        T_obs=T_reset,
        clusters=clusters_reset,
        p_vals=p_reset,
        alpha_cf=ALPHA_CF,
        alpha_clust=ALPHA_CLUST,
        threshold=t_thr_reset,
        n_permutations=n_permutations,
        seed=seed,
        ch_names=sample_epochs.info["ch_names"],
        n_ch=n_channels_r, n_freqs=n_freqs_r, n_times=n_times_reset,
        freqs=freqs_global, times=times_reset_global,
        sample_epochs_path=sample_epochs_path,
    )

    baseline_csv = os.path.join(
        SAVE_DIR, f"baseline_band_power_age_main_seed{seed}.csv"
    )

    # Build DF from per-participant rows
    df_base = pd.DataFrame(baseline_rows)

    # Compute group means for Young and Old only
    df_groups = df_base[df_base["group"].isin(["Young", "Old"])].copy()
    group_means = (
        df_groups.groupby("group")[
            ["theta_db", "alpha_db", "beta_db", "theta_db_uv", "alpha_db_uv", "beta_db_uv"]]
        .mean()
        .reset_index()
    )

    # Append special rows for group means
    append_rows = []
    for _, r in group_means.iterrows():
        append_rows.append({
            "PID": f"GROUP_MEAN_{r['group']}",
            "group": r["group"],
            "baseline_start_s": 0.0,
            "baseline_end_s": 2.0,
            "alpha_db": round(float(r["alpha_db"]), 3),
            "theta_db": round(float(r["theta_db"]), 3),
            "beta_db":  round(float(r["beta_db"]), 3),
            "alpha_db_uv": round(float(r["alpha_db_uv"]), 3),
            "theta_db_uv": round(float(r["theta_db_uv"]), 3),
            "beta_db_uv":  round(float(r["beta_db_uv"]), 3),
        })

    df_out = pd.concat([df_base, pd.DataFrame(append_rows)], ignore_index=True)

    # Keep consistent column order
    col_order = [
        "PID", "group", "baseline_start_s", "baseline_end_s",
        "alpha_db", "theta_db", "beta_db", "alpha_db_uv", "theta_db_uv", "beta_db_uv"
    ]
    df_out = df_out[col_order]

    df_out.to_csv(baseline_csv, index=False)
    print(f"[SAVED] baseline band power (with group means) - {baseline_csv}")

    # === Save group means inside significant clusters (AFTER permutation) ===

    def _cluster_means_in_mask(Y_all, O_all, mask3d):
        # Y_all/O_all: (n_subj, ch, f, t); mask3d: (ch, f, t) boolean
        Y_vals = Y_all.reshape(
            Y_all.shape[0], -1)[:, mask3d.ravel()].mean(axis=1)  # Y_all.reshape(Y_all.shape[0], -1) flattens each subject’s (ch × f × t) into a 1-D vector.
        O_vals = O_all.reshape(
            O_all.shape[0], -1)[:, mask3d.ravel()].mean(axis=1)  # mask3d.ravel() is a boolean vector of the same length (marking voxels inside the cluster).:, mask3d.ravel()] selects only those voxels.
        return Y_vals, O_vals

    # --- standardized effect size (Cohen's d) ---
    def _cohens_d(y, o):
        # count how many participants are in each group (sample sizes).
        ny, no = len(y), len(o)
        # pooled SD with ddof=1 for sample SDs
        pooled_var = (((ny - 1) * np.var(y, ddof=1)) +  # ddof (delta degree of freedom) 1 divides by sample variance
                      ((no - 1) * np.var(o, ddof=1))) / (ny + no - 2)
        pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else np.nan
        # compute mean difference between groups divided by pooled SD = Cohen’s d
        return (np.mean(y) - np.mean(o)) / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan

    # --- RECTANGLE effect size HELPERS  ---

    def _rectangular_box_indices(mask3):
        """
        Given a 3D cluster mask (ch, f, t), return the tightest
        rectangular box that contains it:
          - ch_idx: array of channel indices in rectangle
          - f_idx:  array of freq indices in rectangle
          - t0, t1: first and last time index (inclusive)
        """
        # any channel that participates at any freq×time
        ch_any = mask3.any(axis=(1, 2))      # (n_ch,)
        # any freq that participates at any ch×time
        f_any = mask3.any(axis=(0, 2))      # (n_freq,)
        # any time that participates at any ch×freq
        t_any = mask3.any(axis=(0, 1))      # (n_time,)

        ch_idx = np.where(ch_any)[0]
        f_idx = np.where(f_any)[0]
        t_idx = np.where(t_any)[0]

        if ch_idx.size == 0 or f_idx.size == 0 or t_idx.size == 0:
            return None  # empty / degenerate cluster

        t0, t1 = int(t_idx[0]), int(t_idx[-1])
        return ch_idx, f_idx, t0, t1

    def _rectangular_means(Y_all, O_all, mask3):
        """
        Compute Young/Old means inside the *rectangular box*
        that tightly circumscribes the cluster.

        Y_all, O_all: (n_subj, ch, f, t)
        mask3:        (ch, f, t) boolean cluster mask

        Returns:
          Y_rect, O_rect: (n_y,), (n_o,) subject-level means in the rectangle
          ch_idx, f_idx, t0, t1: rectangle indices
        """
        box = _rectangular_box_indices(mask3)
        if box is None:
            return None, None, None, None, None, None
        ch_idx, f_idx, t0, t1 = box

        # slice box: (n_subj, n_ch_rect, n_f_rect, n_t_rect)
        Y_box = Y_all[:, ch_idx][:, :, f_idx][:, :, :, t0:t1+1]
        O_box = O_all[:, ch_idx][:, :, f_idx][:, :, :, t0:t1+1]

        # average over ch×f×t - one value per subject
        Y_rect = Y_box.mean(axis=(1, 2, 3))
        O_rect = O_box.mean(axis=(1, 2, 3))

        return Y_rect, O_rect, ch_idx, f_idx, t0, t1

    rows = []

    # PREP clusters
    # k=cluster index, m=cluster’s boolean mask over the flattened feature space (length = n_channels * n_freqs * n_times_prep).
    for k, m in enumerate(clusters_prep):
        # Get the p-value for this cluster.
        pv = float(p_prep[k]) if k < len(p_prep) else np.nan
        # Filter to significant clusters only at the cluster level (e.g., p ≤ 0.05).
        if np.isfinite(pv) and pv < ALPHA_CLUST:
            mask3 = np.asarray(m, bool).reshape(
                n_channels_p, n_freqs_p, n_times_prep)  # Convert the flattened boolean mask m back into a 3-D mask with axes channels x fres, times prep
            # prep_all_y is passed in as Y_all, and prep_all_o as O_all
            # Yv: array of shape (n_young,), each element = that young participant’s mean % change inside the cluster
            Yv, Ov = _cluster_means_in_mask(prep_all_y, prep_all_o, mask3)
            

            # 2) Effect size based on the *rectangular box* around the cluster
            Y_rect, O_rect, ch_idx, f_idx, t0, t1 = _rectangular_means(
                prep_all_y, prep_all_o, mask3
            )

            if Y_rect is not None:
                d_rect = float(_cohens_d(Y_rect, O_rect))
                rect_n_ch = int(len(ch_idx))
                rect_n_f = int(len(f_idx))
                rect_n_t = int(t1 - t0 + 1)
                # absolute times (sec)
                rect_t0_s = float(times_prep_global[t0])
                rect_t1_s = float(times_prep_global[t1])
                rect_fmin = float(freqs_global[f_idx[0]])
                rect_fmax = float(freqs_global[f_idx[-1]])
            else:
                d_rect = np.nan
                rect_n_ch = rect_n_f = rect_n_t = 0
                rect_t0_s = rect_t1_s = np.nan
                rect_fmin = rect_fmax = np.nan

            rows.append(dict(
                segment="PREP", cluster=k, p_value=pv,
                # average % change over young participants, within the cluster voxels.
                young_mean=float(Yv.mean()), young_sd=float(Yv.std(ddof=1)),
                old_mean=float(Ov.mean()),  old_sd=float(Ov.std(ddof=1)),
                diff_y_minus_o=float(Yv.mean() - Ov.mean()),
                
                n_subjects_y=int(Yv.size), n_subjects_o=int(Ov.size),
                n_voxels=int(mask3.sum()),

                # rectangle-based summary
                rect_cohens_d=d_rect,
                rect_n_channels=rect_n_ch,
                rect_n_freqs=rect_n_f,
                rect_n_times=rect_n_t,
                rect_t0_s=rect_t0_s,
                rect_t1_s=rect_t1_s,
                rect_fmin_hz=rect_fmin,
                rect_fmax_hz=rect_fmax,
            ))

    # RESET clusters
    for k, m in enumerate(clusters_reset):
        pv = float(p_reset[k]) if k < len(p_reset) else np.nan
        if np.isfinite(pv) and pv < ALPHA_CLUST:
            mask3 = np.asarray(m, bool).reshape(
                n_channels_r, n_freqs_r, n_times_reset)
            Yv, Ov = _cluster_means_in_mask(reset_all_y, reset_all_o, mask3)
            rows.append(dict(
                segment="RESET", cluster=k, p_value=pv,
                young_mean=float(Yv.mean()), young_sd=float(Yv.std(ddof=1)),
                old_mean=float(Ov.mean()),  old_sd=float(Ov.std(ddof=1)),
                diff_y_minus_o=float(Yv.mean() - Ov.mean()),
                
                n_subjects_y=int(Yv.size), n_subjects_o=int(Ov.size),
                n_voxels=int(mask3.sum()),
            ))

    out_csv = os.path.join(SAVE_DIR, f"age_cluster_group_means_seed{seed}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[SAVED] cluster group means - {out_csv}")

    # summary to return
    return dict(
        prep=dict(n_clusters=len(clusters_prep), min_p=float(
            p_prep.min()) if len(p_prep) else None),
        reset=dict(n_clusters=len(clusters_reset), min_p=float(
            p_reset.min()) if len(p_reset) else None),
        t_thresholds=dict(prep=t_thr_prep, reset=t_thr_reset, alpha=ALPHA_CF)
    )


if __name__ == "__main__":
    results = run_age(
        participants=participants,
        young_set=Young,
        old_set=Old,
        base_dir=base_dir,
        aligned_epochs_dir=aligned_epochs_dir,
        sample_pid=sample_pid,
        sample_condition=sample_condition,
        n_permutations=1000,
        alpha=ALPHA_CLUST,  # keep 0.05 as the cluster-level/reporting alpha
        seed=42,
        n_jobs=4,
    )
    print("\nSummary:", results)
