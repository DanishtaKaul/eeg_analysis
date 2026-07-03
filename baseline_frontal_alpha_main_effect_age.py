# -*- coding: utf-8 -*-
"""Baseline alpha over a frontal ROI (Fz, F4, FC1, FC2) per participant, for the frontal age t-test"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne


from cluster_config import (
    participants, Young, Old,
    aligned_epochs_dir, sample_condition,
    raw_tfr_dir,
)

try:
    from extract_prep_reset import MEDIAN_PREP_DUR, load_baseline_bands_from_rawdb
except Exception:
    from extract_prep_reset import MEDIAN_PREP_DUR, load_baseline_bands_from_rawdb  

# ---------- Config ----------
SAVE_DIR = Path(r"D:\cluster_outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_WINDOW = (0.0, 2.0)  
ROI = ["Fz", "F4", "FC1", "FC2"]  
SEED = 42
OUT_CSV = SAVE_DIR / f"baseline_alpha_FRONTALROI_age_main_seed{SEED}.csv"

# ---------- Helpers ----------


def _canon(names):
    return [str(n).strip().upper() for n in names]


def _load_epochs_for_pid(pid: str, condition: str):
    cond_safe = condition.replace(" ", "_").replace("/", "_")
    fpath = os.path.join(aligned_epochs_dir, pid, cond_safe,
                         f"aligned_epochs_{cond_safe}.fif")
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Epochs file not found for {pid} {condition}: {fpath}")
    return mne.read_epochs(fpath, preload=False)


def _roi_indices(ch_names, roi_upper):
    chU = _canon(ch_names)
    idx = [i for i, nm in enumerate(chU) if nm in roi_upper]
    return idx


# ---------- Main ----------

def main():
   

    # Load a global sample to fall back on channel names
    ch_names_fallback = None
    try:
       
        for pid0 in participants:
            try:
                ep0 = _load_epochs_for_pid(pid0, sample_condition)
                ch_names_fallback = ep0.info["ch_names"]
                break
            except Exception:
                continue
    except Exception:
        pass

    if ch_names_fallback is None:
        raise RuntimeError(
            "Could not load any epochs to retrieve channel names for ROI mapping.")

    roi_upper = set(_canon(ROI))

    rows = []

    for pid in participants:
        pid = str(pid)
        group = "Young" if pid in Young else ("Old" if pid in Old else "Other")

        # Try to get THIS participant's channel names
        try:
            ep = _load_epochs_for_pid(pid, sample_condition)
            ch_names = ep.info["ch_names"]
        except Exception:
            ch_names = ch_names_fallback

        idx_roi = _roi_indices(ch_names, roi_upper)
        if len(idx_roi) == 0:
            raise RuntimeError(
                f"{pid}: No ROI channels found. Check channel names.")

        per_cond_alpha = []
        for cond_key in MEDIAN_PREP_DUR.keys():
            
            base = f"TFR_{cond_key}_ALL"
            pdir = os.path.join(raw_tfr_dir, pid)
            for suffix in ("-RAWDB.npy", "-times.npy", "-freqs.npy"):
                fcheck = os.path.join(pdir, f"{base}{suffix}")
                if not os.path.exists(fcheck):
                    raise FileNotFoundError(f"Missing RAW dB file: {fcheck}")

            # Load baseline bands (per-channel)
            bl_bands, freqs_raw, times_raw = load_baseline_bands_from_rawdb(
                pid, cond_key, dir_raw=raw_tfr_dir, baseline=BASELINE_WINDOW
            )

            alpha_vec = np.asarray(bl_bands["alpha"], dtype=float)
            if alpha_vec.ndim != 1 or alpha_vec.size < max(idx_roi) + 1:
                raise ValueError(
                    f"{pid} {cond_key}: alpha vector shape {alpha_vec.shape} incompatible with ROI indices {idx_roi}")

            alpha_db = float(alpha_vec[idx_roi].mean())
            alpha_db_uv = alpha_db + 120

            per_cond_alpha.append(alpha_db)

            rows.append({
                "PID": pid,
                "group": group,
                "condition": cond_key,
                "roi": ",".join(ROI),
                "baseline_start_s": BASELINE_WINDOW[0],
                "baseline_end_s": BASELINE_WINDOW[1],
                "alpha_db": alpha_db,
                "alpha_db_uv": alpha_db_uv,
                "n_roi_channels": int(len(idx_roi)),
            })

        if per_cond_alpha:
            pid_mean = float(np.mean(per_cond_alpha))
            pid_mean_uv = pid_mean + 120
            rows.append({
                "PID": pid,
                "group": group,
                "condition": "MEAN_ACROSS_CONDITIONS",
                "roi": ",".join(ROI),
                "baseline_start_s": BASELINE_WINDOW[0],
                "baseline_end_s": BASELINE_WINDOW[1],
                "alpha_db": pid_mean,
                "alpha_db_uv": pid_mean_uv,
                "n_roi_channels": int(len(idx_roi)),
            })

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Group means from participant-level MEAN rows
    df_mean = df[df["condition"] == "MEAN_ACROSS_CONDITIONS"].copy()
    summaries = []
    for grp, dfg in df_mean.groupby("group"):
        if grp not in ("Young", "Old"):
            continue
        summaries.append({
            "PID": f"GROUP_MEAN_{grp}",
            "group": grp,
            "condition": "MEAN_ACROSS_CONDITIONS",
            "roi": ",".join(ROI),
            "baseline_start_s": BASELINE_WINDOW[0],
            "baseline_end_s": BASELINE_WINDOW[1],
            "alpha_db": float(dfg["alpha_db"].mean()),
            "alpha_db_uv": float(dfg["alpha_db_uv"].mean()),
            "n_roi_channels": int(dfg["n_roi_channels"].median() if not dfg["n_roi_channels"].empty else 0),
        })

    df_out = pd.concat([df, pd.DataFrame(summaries)], ignore_index=True)

    # Column order similar to baseline CSV
    col_order = [
        "PID", "group", "condition", "roi", "baseline_start_s", "baseline_end_s",
        "alpha_db", "alpha_db_uv", "n_roi_channels",
    ]
    df_out = df_out[col_order]

    df_out.to_csv(OUT_CSV, index=False)
    print(f"[SAVED] {len(df_out)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
