"""Light main-effect cluster permutation test (RM-ANOVA: Light/Ambient/Dark) on ERSP for the PREP and RESET segments."""

import os
import numpy as np
import mne
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd  


from mne.stats import (
    permutation_cluster_test,
    f_mway_rm,
    f_threshold_mway_rm,
    combine_adjacency,
)

from extract_prep_reset import load_and_extract_prep_reset, MEDIAN_PREP_DUR, load_baseline_bands_from_rawdb
from cluster_config import (
    participants,
    base_dir, aligned_epochs_dir,
    sample_pid, sample_condition,
    raw_tfr_dir
)

ALPHA_CF = 0.001       # cluster-forming alpha
ALPHA_CLUST = 0.05  # cluster-level alpha (reporting) sig group differences


def _pack_masks(mask_list):
    """Compress list of 1D boolean masks to bytes (8× smaller)."""
    return np.array([np.packbits(m.astype(np.uint8)) for m in mask_list],
                    dtype=object)


def _unpack_masks(packed, n_feat):
    """Decompress packed masks back to 1D boolean arrays."""
    return [np.unpackbits(p).astype(bool)[:n_feat] for p in packed]


def _safe_cond_key(light: str, obst: str) -> str:
    # Matches MEDIAN_PREP_DUR keys like "GLOBAL_LIGHT_EXPECTED_PRESENT"
    return f"GLOBAL_{light}_{obst}"


def collect_light_baseline_df(baseline=(0.0, 2.0)) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per PID × LIGHT level, where the baseline
    band powers are averaged across the four obstacle variants.
    
    """
    rows = []
    for pid in participants:
        for light in LIGHT_LEVELS:  # ["LIGHT","AMBIENT","DARK"]
            theta_vals, alpha_vals, beta_vals = [], [], []
            for obst in OBSTACLE_LEVELS:  # 4 obstacle variants
                cond_key = _safe_cond_key(light, obst)
                if cond_key not in MEDIAN_PREP_DUR:
                    raise KeyError(
                        f"{pid} {cond_key} not found in MEDIAN_PREP_DUR")
                try:
                    bl_bands, freqs_raw, times_raw = load_baseline_bands_from_rawdb(
                        pid, cond_key, dir_raw=raw_tfr_dir,
                        baseline=baseline
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Missing RAW baseline for {pid} {cond_key}"
                    ) from e
                # store band means for this (pid, light, obstacle)
                theta_vals.append(float(bl_bands["theta"].mean()))
                alpha_vals.append(float(bl_bands["alpha"].mean()))
                beta_vals.append(float(bl_bands["beta"].mean()))

            # require all 4 obstacle variants to call it "collapsed"
            if len(theta_vals) == 4:
                rows.append(dict(
                    PID=pid,
                    light=light,
                    baseline_start_s=float(baseline[0]),
                    baseline_end_s=float(baseline[1]),
                    theta_db=float(np.mean(theta_vals)),
                    alpha_db=float(np.mean(alpha_vals)),
                    beta_db=float(np.mean(beta_vals)),
                    theta_db_uv=float(np.mean(theta_vals) + 120.0),
                    alpha_db_uv=float(np.mean(alpha_vals) + 120.0),
                    beta_db_uv=float(np.mean(beta_vals) + 120.0),
                ))
            else:
                raise RuntimeError(
                    f"{pid} {light}: expected 4 obstacle variants, got {len(theta_vals)}"
                )

    return pd.DataFrame(rows)


def save_cluster_result_npz(
    path, segment, F_obs, clusters, p_vals, alpha, threshold,
    n_permutations, seed, ch_names, n_ch, n_freqs, n_times,
    sample_epochs_path="", df1=None, df2=None, alpha_clust=None,
    freqs=None, times=None,
    # keep args for backwards-compat
    subj_means_LIGHT=None, subj_means_AMBIENT=None, subj_means_DARK=None,
    subject_ids=None, cond_labels=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_feat = int(n_ch) * int(n_freqs) * int(n_times)
    masks_packed = _pack_masks(clusters)

    # freqs/times length must match header
    if freqs is not None:
        freqs = np.asarray(freqs)
        assert len(freqs) == int(
            n_freqs), f"len(freqs) {len(freqs)} != n_freqs {n_freqs}"
    if times is not None:
        times = np.asarray(times)
        assert len(times) == int(
            n_times), f"len(times) {len(times)} != n_times {n_times}"

    np.savez_compressed(
        path,
        F_obs=F_obs,
        p_vals=p_vals,
        masks_packed=masks_packed,
        threshold=np.float64(threshold),
        n_permutations=np.int32(n_permutations),
        seed=np.int32(seed),
        n_ch=np.int32(n_ch),
        n_freqs=np.int32(n_freqs),
        n_times=np.int32(n_times),
        n_feat=np.int64(n_feat),
        ch_names=np.array(ch_names, dtype=object),
        segment=np.array(segment),
        sample_epochs_path=np.array(sample_epochs_path, dtype=object),
        saved_at=np.array(time.strftime("%Y-%m-%d %H:%M:%S")),
        df1=(np.int32(df1) if df1 is not None else np.int32(-1)),
        df2=(np.int32(df2) if df2 is not None else np.int32(-1)),
        alpha=np.float64(alpha),
        alpha_clust=(np.float64(alpha_clust)
                     if alpha_clust is not None else np.float64(np.nan)),
        freqs=(np.asarray(freqs) if freqs is not None else np.array([])),
        times=(np.asarray(times) if times is not None else np.array([])),

    )
    print(f"[SAVE] {segment} results - {path}")


def load_cluster_result_npz(path):
    d = np.load(path, allow_pickle=True)

    meta = {
        "alpha": float(d["alpha"]),
        "threshold": float(d["threshold"]),
        "n_permutations": int(d["n_permutations"]),
        "seed": int(d["seed"]),
        "n_ch": int(d["n_ch"]),
        "n_freqs": int(d["n_freqs"]),
        "n_times": int(d["n_times"]),
        "n_feat": int(d["n_feat"]),
        "ch_names": list(d["ch_names"]),
        "segment": str(d["segment"]),
        "sample_epochs_path": str(d["sample_epochs_path"]),
        "saved_at": str(d["saved_at"]),

        "df1": int(d["df1"]) if "df1" in d.files else None,
        "df2": int(d["df2"]) if "df2" in d.files else None,
        "alpha_clust": float(d["alpha_clust"]) if "alpha_clust" in d.files else None,
        "freqs": np.asarray(d["freqs"]) if "freqs" in d.files else None,
        "times": np.asarray(d["times"]) if "times" in d.files else None,
    }

    F_obs = d["F_obs"]
    p_vals = d["p_vals"]
    masks = _unpack_masks(d["masks_packed"], meta["n_feat"])

    # ---- per-cluster, per-condition subject means & labels ----
    subj_means = {}
    if "subj_means_LIGHT" in d.files:
        subj_means["LIGHT"] = [np.asarray(v, dtype=float)
                               for v in d["subj_means_LIGHT"]]
    if "subj_means_AMBIENT" in d.files:
        subj_means["AMBIENT"] = [np.asarray(
            v, dtype=float) for v in d["subj_means_AMBIENT"]]
    if "subj_means_DARK" in d.files:
        subj_means["DARK"] = [np.asarray(v, dtype=float)
                              for v in d["subj_means_DARK"]]

    subject_ids = list(d["subject_ids"]) if "subject_ids" in d.files else None
    cond_labels = list(d["cond_labels"]) if "cond_labels" in d.files else None

    return F_obs, masks, p_vals, meta, subj_means, subject_ids, cond_labels


# -----------------------------
# Configuration
# -----------------------------
LIGHT_LEVELS = ["LIGHT", "AMBIENT", "DARK"]
OBSTACLE_LEVELS = [
    "EXPECTED_PRESENT",
    "EXPECTED_ABSENT",
    "UNEXPECTED_PRESENT",
    "UNEXPECTED_ABSENT",
]

# Where to save NPZ outputs
SAVE_DIR = r"D:\cluster_outputs"   # raw string avoids backslash-escape issues


# -----------------------------
# Helpers
# -----------------------------

def _load_epochs(pid: str, condition: str):
    """Load any valid Epochs file just for channel layout / adjacency."""
    cond_safe = condition.replace(" ", "_").replace("/", "_")
    filepath = os.path.join(aligned_epochs_dir, pid,
                            cond_safe, f"aligned_epochs_{cond_safe}.fif")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Epochs file not found: {filepath}")
    return mne.read_epochs(filepath, preload=False)


def _assert_finite(name: str, arr: np.ndarray):
    if not np.isfinite(arr).all():
        where = np.argwhere(~np.isfinite(arr))
        raise ValueError(
            f"{name}: found non-finite values at indices like {where[:5]}")


def _stack_mean(items):
    """Stack a list of arrays and mean over the new axis 0."""
    return np.mean(np.stack(items, axis=0), axis=0)


def _flatten4(arr4):
    """(n_subj, ch, f, t) -> (n_subj, ch*f*t)."""
    return arr4.reshape(arr4.shape[0], -1)


def _summarize_F_clusters(F_obs, clusters, pvals, alpha, X_list, label_list):
    sig = np.flatnonzero(pvals < alpha)
    print(
        f"  Found {len(clusters)} clusters; {len(sig)} significant at p<{alpha}.")
    if len(sig) == 0:
        return
    for i in sig:
        # clusters are boolean masks over flattened features (out_type='mask')
        mask = clusters[i]
        if mask.dtype != bool:
            mask = mask.astype(bool)
        means = []
        for X in X_list:  # each: (n_subj, n_feat)
            vals = X[:, mask].mean(axis=1)
            means.append((vals.mean(), vals.std(ddof=1)))
        means_str = ", ".join(
            [f"{lab}={m:.3f}±{s:.3f}" for lab, (m, s) in zip(label_list, means)])
        print(
            f"   - Cluster {i}: p={pvals[i]:.4f}, size={int(mask.sum())}, {means_str}")

# -----------------------------
# build per-subject arrays averaged across OBSTACLE for each LIGHT
# -----------------------------


def collect_light_arrays_for_segment(segment: str = "PREP", prep_window: str = "last"):
    """
    segment: 'PREP' or 'RESET'
    Returns
    -------
    X_light, X_ambient, X_dark : np.ndarray
        Shape for each: (n_subjects, n_channels, n_freqs, n_times_segment)
    sample_epochs : mne.Epochs (for channel adjacency)
    """
    assert segment in ("PREP", "RESET"), "segment must be 'PREP' or 'RESET'"

    # Grab a sample Epochs for channel info/adjacency
    sample_epochs = _load_epochs(sample_pid, sample_condition)

    X_by_light = {lvl: [] for lvl in LIGHT_LEVELS}
    N_seg = None
    ref_shape = None  # (n_channels, n_freqs)

    # Build data per PID
    for pid in participants:
        try:
            # For each LIGHT, average across the 4 obstacle levels
            per_pid_segment_by_light = {}

            for light in LIGHT_LEVELS:
                per_obstacle = []
                shape_ref = None

                for obst in OBSTACLE_LEVELS:
                    key = f"GLOBAL_{light}_{obst}"
                    if key not in MEDIAN_PREP_DUR:
                        raise KeyError(
                            f"{pid} {key} not found in MEDIAN_PREP_DUR")
                    try:
                        prep, reset, t_prep, t_reset, freqs = load_and_extract_prep_reset(
                            pid, key, base_dir, prep_window=prep_window
                        )
                    except FileNotFoundError as e:
                        raise FileNotFoundError(
                            f"Missing TFR for {pid} {key}"
                        ) from e

                    # choose the segment-specific array (already fixed-length in  pipeline)
                    # shape: (ch, freq, time_seg)
                    arr = prep if segment == "PREP" else reset

                    # only enforce same shape across obstacles
                    if shape_ref is None:
                        shape_ref = arr.shape
                    else:
                        assert arr.shape == shape_ref, (
                            f"{pid} {key}: shape differs {arr.shape} vs {shape_ref}"
                        )

                    per_obstacle.append(arr)

                # must have all 4 obstacle variants to average
                assert len(
                    per_obstacle) == 4, f"{pid} {light}: got {len(per_obstacle)} obstacles (need 4)"
                per_pid_segment_by_light[light] = _stack_mean(
                    per_obstacle)  # (ch, freq, time_seg)
                # only average within lights
                print(
                    f"[{pid}] {light}: averaged {len(per_obstacle)} obstacles into 1 array {per_pid_segment_by_light[light].shape}")
                # input("..")
            # Only keep PIDs that have all 3 LIGHT levels available
            if all(per_pid_segment_by_light[lv] is not None for lv in LIGHT_LEVELS):
                for lv in LIGHT_LEVELS:
                    if lv not in per_pid_segment_by_light:
                        raise RuntimeError(
                            f"{pid}: missing LIGHT level {lv} for {segment}"
                        )

                for lv in LIGHT_LEVELS:
                    X_by_light[lv].append(per_pid_segment_by_light[lv])
        except Exception as e:
            raise RuntimeError(f"{pid} ({segment}): {e}") from e

    # Stack into arrays (n_subj, ch, f, t)
    X_light = np.stack(X_by_light["LIGHT"],   axis=0)
    X_ambient = np.stack(X_by_light["AMBIENT"], axis=0)
    X_dark = np.stack(X_by_light["DARK"],    axis=0)

    # Basic checks
    assert X_light.shape == X_ambient.shape == X_dark.shape, "Shapes differ among light levels"
    _assert_finite("X_light", X_light)
    _assert_finite("X_ambient", X_ambient)
    _assert_finite("X_dark", X_dark)

    # print(f"Final stacks: LIGHT={len(X_by_light['LIGHT'])}, "
    #       f"AMBIENT={len(X_by_light['AMBIENT'])}, "
    #       f"DARK={len(X_by_light['DARK'])}")

    return X_light, X_ambient, X_dark, sample_epochs

# -----------------------------
# Run cluster permutation for main effect of LIGHT
# -----------------------------


def run_light_main_effect(prep_window: str = "last", n_permutations: int = 1000,
                          seed: int = 42, n_jobs=7):
    results = {}

    # ===== PREP =====
    print("\n=== PREP segment (main effect of LIGHT) ===")
    Xl_prep, Xa_prep, Xd_prep, sample_epochs = collect_light_arrays_for_segment(
        segment="PREP", prep_window=prep_window
    )
    n_subj, n_ch, n_f, n_tp = Xl_prep.shape
    print(f"  Data shape: (n_subj={n_subj}, ch={n_ch}, f={n_f}, t={n_tp})")

    # Build adjacency (channels × freqs × times)
    ch_adj, ch_names = mne.channels.find_ch_adjacency(
        sample_epochs.info, ch_type='eeg')

    # --- Plot channel graph to check montage neighbors ---
    mne.viz.plot_ch_adjacency(
        sample_epochs.info, ch_adj, ch_names, kind="2d", edit=False)

    adj_prep = combine_adjacency(ch_adj, n_f, n_tp)
    print(f"  Adjacency PREP: {adj_prep.shape}")

    # --- get axes for saving (use any valid key; same freqs across segments, times differ per segment) ---
    any_key = next(iter(MEDIAN_PREP_DUR.keys()))
    try:
        _, _, t_prep_ax, t_reset_ax, freqs_ax = load_and_extract_prep_reset(
            sample_pid, any_key, base_dir, prep_window=prep_window
        )
    except FileNotFoundError:
        # fallback: find first PID that has this key
        t_prep_ax = t_reset_ax = freqs_ax = None
        for _pid in participants:
            try:
                _, _, t_prep_ax, t_reset_ax, freqs_ax = load_and_extract_prep_reset(
                    _pid, any_key, base_dir, prep_window=prep_window
                )
                break
            except FileNotFoundError:
                continue
        if freqs_ax is None:
            raise FileNotFoundError(
                "Could not load any axes for saving freqs/times.")

    # Flatten – keep the three conditions SEPARATE here (each: n_subj × n_feat)
    X_prep = [_flatten4(Xl_prep), _flatten4(Xa_prep), _flatten4(Xd_prep)]
    n_feat_prep = X_prep[0].shape[1]

    # [(n_subj, n_feat), ...]
    print("[PREP] X_prep shapes:", [x.shape for x in X_prep])
    print(f"[PREP] n_subj={X_prep[0].shape[0]}, n_feat={n_feat_prep}")

    assert X_prep[0].shape[0] == len(participants), \
        f"Expected {len(participants)} subjects, got {X_prep[0].shape[0]}"

    # -- ASSERTS (catch mismatches early) --
    assert X_prep[0].shape[0] == X_prep[1].shape[0] == X_prep[
        2].shape[0], "Different n_subj across Light levels (PREP)"
    assert X_prep[1].shape[1] == n_feat_prep and X_prep[2].shape[
        1] == n_feat_prep, "Different n_feat across levels (PREP)"
    assert adj_prep.shape == (
        n_feat_prep, n_feat_prep), f"Adjacency mismatch: {adj_prep.shape} vs features {n_feat_prep}"
    assert np.isfinite(X_prep[0]).all() and np.isfinite(X_prep[1]).all(
    ) and np.isfinite(X_prep[2]).all(), "Non-finite values (PREP)"

    # One-way RM ANOVA stat: stack inside the stat_fun  (n_subj, n_feat, 3)
    factor_levels = [3]

    def stat_fun(*conds):
        """
        Accept either:
          - 3 arrays: each (n_subj, n_feat)  - stack to (n_subj, 3, n_feat)
          - 1 array:  pre-stacked; move the 'levels' axis to axis=1 so shape is (n_subj, 3, n_feat)
        """
        if len(conds) == 3:
            # Preferred path: three arrays (n_subj, n_feat)
            n_subj0, n_feat0 = conds[0].shape
            assert all(arr.shape == (n_subj0, n_feat0)
                       for arr in conds), "Shapes differ across Light levels"
            data = np.stack(conds, axis=1)  # -> (n_subj, 3, n_feat)
        elif len(conds) == 1:
            # Defensive path: if something upstream passed a single pre-stacked array
            data = np.asarray(conds[0])
            if data.ndim != 3:
                raise ValueError(f"Expected 3D data, got {data.shape}")
            # need (n_subj, 3, n_feat). Move '3' to axis=1.
            if data.shape[0] == 3:              # (3, n_subj, n_feat)
                data = np.moveaxis(data, 0, 1)  # (n_subj, 3, n_feat)
            elif data.shape[1] == 3:            # already (n_subj, 3, n_feat)
                pass
            elif data.shape[2] == 3:            # (n_subj, n_feat, 3)
                data = np.moveaxis(data, 2, 1)  # (n_subj, 3, n_feat)
            else:
                raise ValueError(
                    f"Could not locate the 3-level axis in {data.shape}")
        else:
            raise ValueError(
                f"stat_fun expected 3 arrays or 1 array; got {len(conds)}")

        # Final shape check
        if data.shape[1] != 3:
            raise ValueError(
                f"Levels must be axis=1 of size 3; got shape {data.shape}")

        data = data.astype(float, copy=False)
        F_vals, _ = f_mway_rm(data, factor_levels=factor_levels, effects='A')
        # F_vals is (n_feat,) for 1 effect
        return F_vals

    thr_prep = f_threshold_mway_rm(
        n_subjects=n_subj, factor_levels=factor_levels, effects='A', pvalue=ALPHA_CF
    )
    # df1 = 3 - 1
    # df2 = (n_subj - 1) * (3 - 1)
    # print(
    #     f"  Threshold PREP (F): {thr_prep:.4f} at CF alpha={ALPHA_CF}")

    F_prep, clusters_prep, p_prep, H0_prep = permutation_cluster_test(
        X_prep,                    # pass THREE arrays (LIGHT, AMBIENT, DARK)
        stat_fun=stat_fun,
        threshold=thr_prep,
        adjacency=adj_prep,
        n_permutations=n_permutations,
        tail=1,                    # F-stat is one-sided positive
        out_type='mask',
        n_jobs=n_jobs,
        seed=seed,
        verbose=True,
        # buffer_size=50000,
    )
    print(
        f"  PREP: {len(clusters_prep)} clusters, min p = {p_prep.min() if len(p_prep) else None}")

    _summarize_F_clusters(F_prep, clusters_prep, p_prep,
                          ALPHA_CLUST, X_prep, ["LIGHT", "AMBIENT", "DARK"])
    results['prep'] = dict(n_clusters=len(clusters_prep),
                           min_p=(float(p_prep.min()) if len(p_prep) else None))
    # --- SAVE PREP ---
    prep_out = os.path.join(
        SAVE_DIR, f"light_main_PREP_last_prep{n_permutations}perm_seed{seed}.npz"
    )

    # a path to any epochs file to recover channel positions for topomaps
    cond_safe = sample_condition.replace(" ", "_").replace("/", "_")
    sample_epochs_path = os.path.join(
        aligned_epochs_dir, sample_pid, cond_safe, f"aligned_epochs_{cond_safe}.fif"
    )

    save_cluster_result_npz(
        path=prep_out,
        segment="PREP",
        F_obs=F_prep,
        clusters=clusters_prep,         # masks because out_type='mask'
        p_vals=p_prep,
        alpha=ALPHA_CF,
        threshold=thr_prep,
        n_permutations=n_permutations,
        seed=seed,
        ch_names=sample_epochs.info["ch_names"],
        n_ch=n_ch, n_freqs=n_f, n_times=n_tp,
        sample_epochs_path=sample_epochs_path,
        # df1=df1, df2=df2,
        alpha_clust=ALPHA_CLUST,
        freqs=freqs_ax, times=t_prep_ax,
    )

    # ===== RESET =====
    print("\n=== RESET segment (main effect of LIGHT) ===")
    Xl_res, Xa_res, Xd_res, sample_epochs2 = collect_light_arrays_for_segment(
        segment="RESET", prep_window=prep_window
    )
    n_subj2, n_ch2, n_f2, n_tr = Xl_res.shape
    print(f"  Data shape: (n_subj={n_subj2}, ch={n_ch2}, f={n_f2}, t={n_tr})")

    # Build adjacency for RESET (note different time length)
    adj_reset = combine_adjacency(ch_adj, n_f2, n_tr)
    print(f"  Adjacency RESET: {adj_reset.shape}")

    # Flatten – keep the three conditions separate (each: n_subj × n_feat)
    X_reset = [_flatten4(Xl_res), _flatten4(Xa_res), _flatten4(Xd_res)]
    n_feat_reset = X_reset[0].shape[1]

    print("[RESET] X_reset shapes:", [x.shape for x in X_reset])
    print(f"[RESET] n_subj={X_reset[0].shape[0]}, n_feat={n_feat_reset}")

    assert X_reset[0].shape[0] == len(participants), \
        f"Expected {len(participants)} subjects, got {X_reset[0].shape[0]}"

    # -- ASSERTS --
    assert X_reset[0].shape[0] == X_reset[1].shape[0] == X_reset[
        2].shape[0], "Different n_subj across Light levels (RESET)"
    assert X_reset[1].shape[1] == n_feat_reset and X_reset[2].shape[
        1] == n_feat_reset, "Different n_feat across levels (RESET)"
    assert adj_reset.shape == (
        n_feat_reset, n_feat_reset), f"Adjacency mismatch: {adj_reset.shape} vs features {n_feat_reset}"
    assert np.isfinite(X_reset[0]).all() and np.isfinite(X_reset[1]).all(
    ) and np.isfinite(X_reset[2]).all(), "Non-finite values (RESET)"

    # Threshold
    thr_reset = f_threshold_mway_rm(
        n_subjects=n_subj2, factor_levels=factor_levels, effects='A', pvalue=ALPHA_CF
    )

    # df1_reset = 3 - 1
    # df2_reset = (n_subj2 - 1) * (3 - 1)
    # print(
    #     f"  Threshold RESET (F): {thr_reset:.4f} at CF alpha={ALPHA_CF}")

    F_reset, clusters_reset, p_reset, H0_reset = permutation_cluster_test(
        X_reset,                   # pass THREE arrays again
        stat_fun=stat_fun,
        threshold=thr_reset,
        adjacency=adj_reset,
        n_permutations=n_permutations,
        tail=1,
        out_type='mask',
        n_jobs=n_jobs,
        seed=seed,
        verbose=True,
        # buffer_size=50000,
    )
    print(
        f"  RESET: {len(clusters_reset)} clusters, min p = {p_reset.min() if len(p_reset) else None}")

    _summarize_F_clusters(F_reset, clusters_reset, p_reset,
                          ALPHA_CLUST, X_reset, ["LIGHT", "AMBIENT", "DARK"])
    results['reset'] = dict(n_clusters=len(clusters_reset),
                            min_p=(float(p_reset.min()) if len(p_reset) else None))
    # --- SAVE RESET ---
    reset_out = os.path.join(
        SAVE_DIR, f"light_main_RESET_last_prep{n_permutations}perm_seed{seed}.npz"
    )

    cond_safe2 = sample_condition.replace(" ", "_").replace("/", "_")
    sample_epochs_path2 = os.path.join(
        aligned_epochs_dir, sample_pid, cond_safe2, f"aligned_epochs_{cond_safe2}.fif"
    )

    save_cluster_result_npz(
        path=reset_out,
        segment="RESET",
        F_obs=F_reset,
        clusters=clusters_reset,
        p_vals=p_reset,
        alpha=ALPHA_CF,
        threshold=thr_reset,
        n_permutations=n_permutations,
        seed=seed,
        ch_names=sample_epochs2.info["ch_names"],
        n_ch=n_ch2, n_freqs=n_f2, n_times=n_tr,
        sample_epochs_path=sample_epochs_path2,
        # df1=df1_reset, df2=df2_reset,
        alpha_clust=ALPHA_CLUST,
        freqs=freqs_ax, times=t_reset_ax,
    )

    # === Save group means inside significant clusters (AFTER permutation) ===
    # X_prep / X_reset are lists of 3 arrays (LIGHT, AMBIENT, DARK), each shaped (n_subj, n_feat)
    def _means_for_mask(mask1d, X_list, labels):
        row = {}
        n_vox = int(mask1d.sum())
        for lab, X in zip(labels, X_list):
            # per-subject mean within cluster
            vals = X[:, mask1d].mean(axis=1)
            row[f"{lab.lower()}_mean"] = float(
                vals.mean())  # mean across subjects
            row[f"{lab.lower()}_sd"] = float(vals.std(ddof=1))
        row["n_subjects"] = int(X_list[0].shape[0])
        row["n_voxels"] = n_vox
        return row
    # --- RECTANGLE HELPERS for LIGHT main effect (within-subject) ---

    def _rectangular_box_indices(mask1d, n_ch, n_f, n_t):
        """
        Given a 1D cluster mask over flattened features, return the
        tightest rectangular box (ch, freq, time) containing the cluster.
        """
        mask3 = mask1d.reshape(n_ch, n_f, n_t)  # (ch, f, t)

        ch_any = mask3.any(axis=(1, 2))   # channels involved anywhere
        f_any = mask3.any(axis=(0, 2))   # freqs involved anywhere
        t_any = mask3.any(axis=(0, 1))   # times involved anywhere

        ch_idx = np.where(ch_any)[0]
        f_idx = np.where(f_any)[0]
        t_idx = np.where(t_any)[0]

        if ch_idx.size == 0 or f_idx.size == 0 or t_idx.size == 0:
            return None

        t0, t1 = int(t_idx[0]), int(t_idx[-1])
        return ch_idx, f_idx, t0, t1

    def _rectangle_means_for_light(mask1d, X_L, X_A, X_D, n_ch, n_f, n_t):
        """
        Compute per-subject means inside the rectangular box that
        tightly circumscribes the cluster, for LIGHT/AMBIENT/DARK.
        """
        box = _rectangular_box_indices(mask1d, n_ch, n_f, n_t)
        if box is None:
            return None
        ch_idx, f_idx, t0, t1 = box

        def _box_means(arr4):
            # arr4: (n_subj, ch, f, t)
            sub = arr4[:, ch_idx][:, :, f_idx][:, :, :, t0:t1+1]
            return sub.mean(axis=(1, 2, 3))  # → (n_subj,)

        vals_L = _box_means(X_L)
        vals_A = _box_means(X_A)
        vals_D = _box_means(X_D)

        return vals_L, vals_A, vals_D, ch_idx, f_idx, t0, t1

    def _rm_rect_effectsize(vals_L, vals_A, vals_D):
        """
        One-way RM ANOVA on rectangle means (3 levels: L, A, D),
        returning F, partial eta^2, and Cohen's f.
        """
        data = np.stack([vals_L, vals_A, vals_D], axis=1)  # (n_subj, 3)
        data3 = data[..., np.newaxis]                      # (n_subj, 3, 1)
        F_arr, _ = f_mway_rm(data3, factor_levels=[3], effects='A')
        F = float(F_arr.squeeze())

        n_subj_loc = data.shape[0]
        df1 = 3 - 1
        df2 = (n_subj_loc - 1) * (3 - 1)

        if not np.isfinite(F) or (F * df1 + df2) <= 0:
            return np.nan, np.nan, np.nan

        eta_p2 = (F * df1) / (F * df1 + df2)
        if not np.isfinite(eta_p2) or eta_p2 >= 1.0:
            return F, np.nan, np.nan

        f = np.sqrt(eta_p2 / (1.0 - eta_p2))
        return F, eta_p2, f

    rows = []
    labels = ["LIGHT", "AMBIENT", "DARK"]

    # PREP clusters - CSV rows
    # PREP clusters - CSV rows (with rectangle effect size)
    for k, mask in enumerate(clusters_prep):
        pv = float(p_prep[k]) if k < len(p_prep) else np.nan
        if np.isfinite(pv) and pv < ALPHA_CLUST:
            mask1d = np.asarray(mask, dtype=bool)

            # 1) Cluster-mask means
            base = dict(segment="PREP", cluster=k, p_value=pv)
            base.update(_means_for_mask(mask1d, X_prep, labels))

            # 2) Rectangle-based means and effect size
            rect_res = _rectangle_means_for_light(
                mask1d,
                Xl_prep, Xa_prep, Xd_prep,   # 4D arrays (n_subj, ch, f, t)
                n_ch, n_f, n_tp
            )
            if rect_res is not None:
                vals_L, vals_A, vals_D, ch_idx, f_idx, t0, t1 = rect_res
                F_rect, eta_p2_rect, f_rect = _rm_rect_effectsize(
                    vals_L, vals_A, vals_D
                )

                base.update(dict(
                    rect_F=F_rect,
                    rect_eta_p2=eta_p2_rect,
                    rect_cohens_f=f_rect,
                    rect_n_channels=int(len(ch_idx)),
                    rect_n_freqs=int(len(f_idx)),
                    rect_n_times=int(t1 - t0 + 1),
                    rect_t0_s=float(t_prep_ax[t0]),
                    rect_t1_s=float(t_prep_ax[t1]),
                    rect_fmin_hz=float(freqs_ax[f_idx[0]]),
                    rect_fmax_hz=float(freqs_ax[f_idx[-1]]),
                ))
            else:
                base.update(dict(
                    rect_F=np.nan,
                    rect_eta_p2=np.nan,
                    rect_cohens_f=np.nan,
                    rect_n_channels=0,
                    rect_n_freqs=0,
                    rect_n_times=0,
                    rect_t0_s=np.nan,
                    rect_t1_s=np.nan,
                    rect_fmin_hz=np.nan,
                    rect_fmax_hz=np.nan,
                ))

            rows.append(base)

    # RESET clusters - CSV rows
    # RESET clusters - CSV rows (with rectangle effect size)
    for k, mask in enumerate(clusters_reset):
        pv = float(p_reset[k]) if k < len(p_reset) else np.nan
        if np.isfinite(pv) and pv < ALPHA_CLUST:
            mask1d = np.asarray(mask, dtype=bool)

            # 1) Cluster-mask means
            base = dict(segment="RESET", cluster=k, p_value=pv)
            base.update(_means_for_mask(mask1d, X_reset, labels))

            # 2) Rectangle-based means and effect size (RESET dims)
            rect_res = _rectangle_means_for_light(
                mask1d,
                Xl_res, Xa_res, Xd_res,   # (n_subj2, ch2, f2, t_reset)
                n_ch2, n_f2, n_tr
            )
            if rect_res is not None:
                vals_L, vals_A, vals_D, ch_idx, f_idx, t0, t1 = rect_res
                F_rect, eta_p2_rect, f_rect = _rm_rect_effectsize(
                    vals_L, vals_A, vals_D
                )

                base.update(dict(
                    rect_F=F_rect,
                    rect_eta_p2=eta_p2_rect,
                    rect_cohens_f=f_rect,
                    rect_n_channels=int(len(ch_idx)),
                    rect_n_freqs=int(len(f_idx)),
                    rect_n_times=int(t1 - t0 + 1),
                    rect_t0_s=float(t_reset_ax[t0]),
                    rect_t1_s=float(t_reset_ax[t1]),
                    rect_fmin_hz=float(freqs_ax[f_idx[0]]),
                    rect_fmax_hz=float(freqs_ax[f_idx[-1]]),
                ))
            else:
                base.update(dict(
                    rect_F=np.nan,
                    rect_eta_p2=np.nan,
                    rect_cohens_f=np.nan,
                    rect_n_channels=0,
                    rect_n_freqs=0,
                    rect_n_times=0,
                    rect_t0_s=np.nan,
                    rect_t1_s=np.nan,
                    rect_fmin_hz=np.nan,
                    rect_fmax_hz=np.nan,
                ))

            rows.append(base)

    if rows:
        out_csv = os.path.join(
            SAVE_DIR, f"light_main_cluster_group_means_seed{seed}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[SAVED] cluster group means - {out_csv}")
    else:
        print("[INFO] No significant clusters at ALPHA_CLUST; no CSV written.")

    # ---- BASELINE CSV for main effect of LIGHT (collapsed over obstacle) ----
    baseline_light_csv = os.path.join(
        SAVE_DIR, f"baseline_light_main_band_power_seed{seed}.csv")
    df_base_light = collect_light_baseline_df(baseline=(0.0, 2.0))

    # Append per-light overall means (across participants)
    append_rows = []
    for light in LIGHT_LEVELS:
        df_l = df_base_light[df_base_light["light"] == light]
        if not df_l.empty:
            append_rows.append({
                "PID": f"LIGHT_MEAN_{light}",
                "light": light,
                "baseline_start_s": 0.0,
                "baseline_end_s": 2.0,
                "theta_db": round(float(df_l["theta_db"].mean()), 3),
                "alpha_db": round(float(df_l["alpha_db"].mean()), 3),
                "beta_db":  round(float(df_l["beta_db"].mean()),  3),
                "theta_db_uv": round(float(df_l["theta_db"].mean()) + 120.0, 3),
                "alpha_db_uv": round(float(df_l["alpha_db"].mean()) + 120.0, 3),
                "beta_db_uv":  round(float(df_l["beta_db"].mean()) + 120.0, 3),

            })

    df_out = pd.concat(
        [df_base_light, pd.DataFrame(append_rows)], ignore_index=True)

    # Stable column order
    col_order = ["PID", "light", "baseline_start_s",
                 "baseline_end_s", "theta_db", "alpha_db", "beta_db", "theta_db_uv", "alpha_db_uv", "beta_db_uv"]
    df_out = df_out[col_order]

    df_out.to_csv(baseline_light_csv, index=False)
    print(
        f"[SAVED] baseline (LIGHT main effect; obstacle-collapsed) - {baseline_light_csv}")

    return results


if __name__ == "__main__":
    out = run_light_main_effect(
        prep_window="last", n_permutations=1000, seed=42, n_jobs=7)
    print("\nSummary:", out)
