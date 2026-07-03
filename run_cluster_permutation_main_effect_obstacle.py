# -*- coding: utf-8 -*-

"""One-way 4-level RM-ANOVA cluster permutation test over OBSTACLE (collapsed across
Light) on ERSP for the PREP and RESET segments"""

import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_test, f_mway_rm, f_threshold_mway_rm, combine_adjacency

from extract_prep_reset import (
    load_and_extract_prep_reset,  
    MEDIAN_PREP_DUR,
   
)
from cluster_config import (
    participants,
    base_dir,
    aligned_epochs_dir,
    sample_pid,
    sample_condition,
    
)

SAVE_DIR = r"D:\\cluster_outputs"
LIGHT_LEVELS = ["LIGHT", "AMBIENT", "DARK"]
LEVELS = [
    ("E_P", "EXPECTED_PRESENT"),
    ("E_A", "EXPECTED_ABSENT"),
    ("U_P", "UNEXPECTED_PRESENT"),
    ("U_A", "UNEXPECTED_ABSENT"),
]
# for consistent CSV column ordering
LEVEL_ORDER = ["E_P", "E_A", "U_P", "U_A"]
LEVEL_LABELS = {
    "E_P": "EXPECTED_PRESENT",
    "E_A": "EXPECTED_ABSENT",
    "U_P": "UNEXPECTED_PRESENT",
    "U_A": "UNEXPECTED_ABSENT",
}

ALPHA_CF = 0.001   # cluster-forming p for F-threshold
ALPHA_CLUST = 0.05  # reporting alpha

# ---------------- helpers ----------------


def _load_epochs(pid: str, condition: str):
    cond_safe = condition.replace(" ", "_").replace("/", "_")
    pth = os.path.join(aligned_epochs_dir, pid, cond_safe,
                       f"aligned_epochs_{cond_safe}.fif")
    if not os.path.exists(pth):
        raise FileNotFoundError(pth)
    return mne.read_epochs(pth, preload=False)


def _assert_finite(name, arr):
    if not np.isfinite(arr).all():
        idx = np.argwhere(~np.isfinite(arr))
        raise ValueError(
            f"{name} has non-finite values; example indices: {idx[:5]}")


def _mean_stack(lst):
    return np.mean(np.stack(lst, axis=0), axis=0)


def _flatten4(x):
    # (n, ch, f, t) -> (n, ch*f*t)
    return x.reshape(x.shape[0], -1)


def _collect_segment(segment: str, prep_window: str = "last"):
    assert segment in ("PREP", "RESET")
    # sample epochs for adjacency/topology
    ep = _load_epochs(sample_pid, sample_condition)

    freqs = None
    times = None

    # dict of per-subject lists for each of the 4 levels
    per_level = {key: [] for key, _ in LEVELS}

    for pid in participants:
        # For each LIGHT, collect arrays for all 4 levels; HARD FAIL if any missing
        per_light = {key: [] for key, _ in LEVELS}
        for light in LIGHT_LEVELS:
            for short, longname in LEVELS:
                cond = f"GLOBAL_{light}_{longname}"
                if cond not in MEDIAN_PREP_DUR:
                    raise RuntimeError(f"Missing prep median for {pid} {cond}")
                prep, reset, t_prep, t_reset, f_cur = load_and_extract_prep_reset(
                    pid, cond, base_dir, prep_window=prep_window
                )
                arr = prep if segment == "PREP" else reset
                if freqs is None:
                    freqs = np.asarray(f_cur).copy()
                if times is None:
                    times = np.asarray(t_prep if segment ==
                                       "PREP" else t_reset).copy()
                per_light[short].append(arr)
        # require all 3 LIGHTS for each level
        for short in per_light:
            if len(per_light[short]) != 3:
                raise RuntimeError(
                    f"{pid} segment={segment}: missing LIGHTS for {short} -> {len(per_light[short])}")
        # collapse across LIGHT
        for short in per_level:
            per_level[short].append(_mean_stack(per_light[short]))

    # stack per subject -> arrays of shape (n_subj, ch, f, t)
    E_P = np.stack(per_level["E_P"], axis=0)
    E_A = np.stack(per_level["E_A"], axis=0)
    U_P = np.stack(per_level["U_P"], axis=0)
    U_A = np.stack(per_level["U_A"], axis=0)

    shp = E_P.shape
    assert all(x.shape == shp for x in (E_A, U_P, U_A)
               ), "Shapes differ across levels"
    for name, arr in zip(["E_P", "E_A", "U_P", "U_A"], [E_P, E_A, U_P, U_A]):
        _assert_finite(name, arr)

    return (E_P, E_A, U_P, U_A), ep, freqs, times


def _pack_masks(mask_list):
    return np.array([np.packbits(m.astype(np.uint8)) for m in mask_list], dtype=object)


def _save_npz(path, segment, F_obs, clusters, p_vals, threshold, n_perm, seed,
              ch_names, n_ch, n_f, n_t, sample_epochs_path, freqs, times):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        F_obs=F_obs,
        p_vals=p_vals,
        masks_packed=_pack_masks(clusters),
        threshold=float(threshold),
        n_permutations=int(n_perm),
        seed=int(seed),
        n_ch=int(n_ch), n_freqs=int(n_f), n_times=int(n_t), n_feat=int(n_ch*n_f*n_t),
        ch_names=np.array(ch_names, dtype=object),
        segment=np.array(segment), effect=np.array('ONEWAY_OBSTACLE'),
        sample_epochs_path=np.array(sample_epochs_path, dtype=object),
        saved_at=np.array(time.strftime("%Y-%m-%d %H:%M:%S")),
        alpha=float(ALPHA_CF), alpha_clust=float(ALPHA_CLUST),
        freqs=np.asarray(freqs), times=np.asarray(times),
    )
    print(f"[SAVE] ONEWAY_OBSTACLE {segment} -> {path}")


# ---------------- core runner ----------------

def run_obstacle_oneway_with_reports(prep_window: str = "last", n_permutations: int = 1000, seed: int = 42, n_jobs: int = 3):
    cond_safe = sample_condition.replace(" ", "_").replace("/", "_")
    sample_epochs_path = os.path.join(
        aligned_epochs_dir, sample_pid, cond_safe, f"aligned_epochs_{cond_safe}.fif")

    results = {}

    for segment in ("PREP", "RESET"):
        print(f"\n=== {segment} (One-way 4-level RM ANOVA: OBSTACLE) ===")
        (E_P, E_A, U_P, U_A), ep, freqs, times = _collect_segment(
            segment, prep_window=prep_window)
        n_subj, n_ch, n_f, n_t = E_P.shape
        ch_adj, ch_names = mne.channels.find_ch_adjacency(
            ep.info, ch_type='eeg')
        adj = combine_adjacency(ch_adj, n_f, n_t)

        # Flatten each level to (n_subj, n_feat)
        X_dict = {
            'E_P': _flatten4(E_P),
            'E_A': _flatten4(E_A),
            'U_P': _flatten4(U_P),
            'U_A': _flatten4(U_A),
        }
        n_feat = X_dict['E_P'].shape[1]
        assert adj.shape == (
            n_feat, n_feat), f"Adjacency mismatch {adj.shape} vs {n_feat}"

        factor_levels = [4]  # one factor with 4 levels

        def stat_fun(EP, EA, UP, UA):
            # expect inputs as (n_subj, n_feat) each
            n0, p0 = EP.shape
            assert all(x.shape == (n0, p0)
                       for x in (EA, UP, UA)), "Shapes differ across levels"
            data = np.stack([EP, EA, UP, UA], axis=1)  # (n_subj, 4, n_feat)
            F_vals, _ = f_mway_rm(
                data, factor_levels=factor_levels, effects='A')
            return F_vals  # (n_feat,)

        thr = f_threshold_mway_rm(
            n_subjects=n_subj, factor_levels=factor_levels, effects='A', pvalue=ALPHA_CF)
        # df1 = 4 - 1
        # df2 = (n_subj - 1) * (4 - 1)
        # print(f"  Threshold (F): {thr:.4f} at CF alpha={ALPHA_CF}")

        # Keep X_list in LEVEL_ORDER to match stat_fun argument order
        X_list = [X_dict[k] for k in LEVEL_ORDER]
        F_obs, clusters, p_vals, H0 = permutation_cluster_test(
            X_list,
            stat_fun=stat_fun,
            threshold=thr,
            adjacency=adj,
            n_permutations=n_permutations,
            tail=1,
            out_type='mask',
            n_jobs=n_jobs,
            seed=seed,
            verbose=True,
        )
        print(
            f"  {segment}: {len(clusters)} clusters; min p = {p_vals.min() if len(p_vals) else None}")

        # Save NPZ (parity with LIGHT)
        out_npz = os.path.join(
            SAVE_DIR, f"obstacle_main_effect_{segment.lower()}_{n_permutations}perm_seed{seed}.npz")
        _save_npz(out_npz, segment, F_obs, clusters, p_vals, thr, n_permutations, seed,
                  ch_names, n_ch, n_f, n_t, sample_epochs_path, freqs, times)

        # ---------------- Cluster group-means CSV (significant clusters only) ----------------
        rows = []

        def _means_for_mask(mask1d):
            row = {}
            n_vox = int(mask1d.sum())
            for key in LEVEL_ORDER:
                # per-subject mean within cluster
                vals = X_dict[key][:, mask1d].mean(axis=1)
                row[f"{LEVEL_LABELS[key].lower()}_mean"] = float(vals.mean())
                row[f"{LEVEL_LABELS[key].lower()}_sd"] = float(
                    vals.std(ddof=1))
            row["n_subjects"] = int(X_list[0].shape[0])
            row["n_voxels"] = n_vox
            return row

        # --- RECTANGLE HELPERS for 4-level OBSTACLE main effect ---

        def _rectangular_box_indices(mask1d, n_ch, n_f, n_t):
            """
            Given a 1D cluster mask over flattened features, return the
            tightest rectangular box (channels, freqs, time) containing the cluster.
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

        def _rectangle_means_for_obstacle(mask1d, E_P_arr, E_A_arr, U_P_arr, U_A_arr,
                                          n_ch, n_f, n_t):
            """
            Compute per-subject means inside the rectangular box that
            tightly circumscribes the cluster, for the four OBSTACLE levels.
            """
            box = _rectangular_box_indices(mask1d, n_ch, n_f, n_t)
            if box is None:
                return None
            ch_idx, f_idx, t0, t1 = box

            def _box_means(arr4):
                # arr4: (n_subj, ch, f, t)
                sub = arr4[:, ch_idx][:, :, f_idx][:, :, :, t0:t1 + 1]
                return sub.mean(axis=(1, 2, 3))  # → (n_subj,)

            vals_EP = _box_means(E_P_arr)
            vals_EA = _box_means(E_A_arr)
            vals_UP = _box_means(U_P_arr)
            vals_UA = _box_means(U_A_arr)

            return vals_EP, vals_EA, vals_UP, vals_UA, ch_idx, f_idx, t0, t1

        def _rm_rect_effectsize_4(vals_EP, vals_EA, vals_UP, vals_UA):
            """
            One-way RM ANOVA on rectangle means
            """
            data = np.stack([vals_EP, vals_EA, vals_UP,
                            vals_UA], axis=1)  # (n_subj, 4)
            # (n_subj, 4, 1)
            data3 = data[..., np.newaxis]
            F_arr, _ = f_mway_rm(data3, factor_levels=[4], effects='A')
            F = float(F_arr.squeeze())

            n_subj_loc = data.shape[0]
            df1_loc = 4 - 1
            df2_loc = (n_subj_loc - 1) * (4 - 1)

            if not np.isfinite(F) or (F * df1_loc + df2_loc) <= 0:
                return np.nan, np.nan, np.nan

            eta_p2 = (F * df1_loc) / (F * df1_loc + df2_loc)
            if not np.isfinite(eta_p2) or eta_p2 >= 1.0:
                return F, np.nan, np.nan

            f_val = np.sqrt(eta_p2 / (1.0 - eta_p2))
            return F, eta_p2, f_val

        # ---- Loop over clusters and build CSV rows (mask + rectangle) ----
        for k, mask in enumerate(clusters):
            pv = float(p_vals[k]) if k < len(p_vals) else np.nan
            if np.isfinite(pv) and pv < ALPHA_CLUST:
                mask1d = np.asarray(mask, dtype=bool)

                # 1) Cluster-mask means
                base = dict(segment=segment, cluster=int(k), p_value=pv)
                base.update(_means_for_mask(mask1d))

                # 2) Rectangle-based means & effect size
                rect_res = _rectangle_means_for_obstacle(
                    mask1d,
                    # 4D arrays (n_subj, ch, f, t) for this segment
                    E_P, E_A, U_P, U_A,
                    n_ch, n_f, n_t
                )

                if rect_res is not None:
                    vals_EP, vals_EA, vals_UP, vals_UA, ch_idx, f_idx, t0, t1 = rect_res
                    F_rect, eta_p2_rect, f_rect = _rm_rect_effectsize_4(
                        vals_EP, vals_EA, vals_UP, vals_UA
                    )

                    base.update(dict(
                        rect_F=F_rect,
                        rect_eta_p2=eta_p2_rect,
                        rect_cohens_f=f_rect,
                        rect_n_channels=int(len(ch_idx)),
                        rect_n_freqs=int(len(f_idx)),
                        rect_n_times=int(t1 - t0 + 1),
                        rect_t0_s=float(times[t0]),
                        rect_t1_s=float(times[t1]),
                        rect_fmin_hz=float(freqs[f_idx[0]]),
                        rect_fmax_hz=float(freqs[f_idx[-1]]),
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
                SAVE_DIR, f"obstacle_main_effect_cluster_group_means_seed{seed}.csv")
            mode = 'a' if os.path.exists(out_csv) else 'w'
            header = not os.path.exists(out_csv)
            pd.DataFrame(rows).to_csv(
                out_csv, index=False, mode=mode, header=header)
            print(f"[SAVED] cluster group means {out_csv}")
        else:
            print(
                "[INFO] No significant clusters ≤ alpha_clust; no group-means rows written for this segment.")

        results[segment] = dict(
            n_clusters=len(clusters),
            min_p=(float(p_vals.min()) if len(p_vals) else None),
        )

   

    return results


if __name__ == "__main__":
    out = run_obstacle_oneway_with_reports(
        prep_window="last", n_permutations=1000, seed=42, n_jobs=3)
    print("\nSummary:")
    print(out)
