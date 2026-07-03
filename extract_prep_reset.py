"""Extract prep and reset time windows from saved TFRs, and baseline band power from raw-dB TFRs, for cluster analysis"""

import os
import numpy as np


MEDIAN_PREP_DUR = {
    "GLOBAL_AMBIENT_EXPECTED_ABSENT": 2.78011111111111,
    "GLOBAL_AMBIENT_EXPECTED_PRESENT": 2.95463157894736,
    "GLOBAL_AMBIENT_UNEXPECTED_ABSENT": 3.02029999999999,
    "GLOBAL_AMBIENT_UNEXPECTED_PRESENT": 3.05684210526315,
    "GLOBAL_DARK_EXPECTED_ABSENT": 3.03666666666666,
    "GLOBAL_DARK_EXPECTED_PRESENT": 3.1681,
    "GLOBAL_DARK_UNEXPECTED_ABSENT": 3.37526315789473,
    "GLOBAL_DARK_UNEXPECTED_PRESENT": 3.41347368421052,
    "GLOBAL_LIGHT_EXPECTED_ABSENT": 2.76266666666666,
    "GLOBAL_LIGHT_EXPECTED_PRESENT": 2.95768421052631,
    "GLOBAL_LIGHT_UNEXPECTED_ABSENT": 2.85509999999999,
    "GLOBAL_LIGHT_UNEXPECTED_PRESENT": 3.34009999999999,
}


def load_and_extract_prep_reset(
    pid,
    cond_key,
    base_dir=r"D:\tfr_full",
    baseline_sec=2.0,
    prep_window="last",  


):
    if cond_key not in MEDIAN_PREP_DUR:
        raise ValueError(f"Condition key '{cond_key}' not found.")

    # Load arrays
    ersp_path = os.path.join(base_dir, pid, f"TFR_{cond_key}_ALL-ERSP.npy")
    times_path = os.path.join(base_dir, pid, f"TFR_{cond_key}_ALL-times.npy")
    freqs_path = os.path.join(base_dir, pid, f"TFR_{cond_key}_ALL-freqs.npy")

    ersp = np.load(ersp_path)    # (channels, freqs, time)
    times = np.load(times_path)   # (time,)
    freqs = np.load(freqs_path)


    # Derive sfreq from times instead of hard-coding
    dt = np.median(np.diff(times))
    sfreq = 1.0 / dt
    

    # Samples
    baseline_samples = int(round(baseline_sec * sfreq))
    median_prep_samples = int(round(MEDIAN_PREP_DUR[cond_key] * sfreq))
    prep_extract_samples = int(round(2.7 * sfreq))   # 2.7 s
    reset_extract_samples = int(round(1.0 * sfreq))   # 1.0 s

    # ----- PREP indices -----
    first_start = baseline_samples
    first_end = baseline_samples + prep_extract_samples

    last_start = baseline_samples + \
        (median_prep_samples - prep_extract_samples)
    last_end = baseline_samples + median_prep_samples

    # ----- RESET (first 1. s after end of prep) -----
    reset_start = baseline_samples + median_prep_samples
    reset_end = reset_start + reset_extract_samples

    # Bounds checks
    T = ersp.shape[2]
    assert T == len(times), "Time axis length mismatch"
    assert 0 <= first_start < first_end <= T
    assert 0 <= last_start < last_end <= T
    assert 0 <= reset_start < reset_end <= T

    # Slices
    ersp_prep_first = ersp[:, :, first_start:first_end]
    times_prep_first = times[first_start:first_end]

    ersp_prep_last = ersp[:, :, last_start:last_end]
    times_prep_last = times[last_start:last_end]

    ersp_reset = ersp[:, :, reset_start:reset_end]
    times_reset = times[reset_start:reset_end]

    # Return based on requested window
    if prep_window == "first":
        return ersp_prep_first, ersp_reset, times_prep_first, times_reset, freqs
    elif prep_window == "last":
        return ersp_prep_last,  ersp_reset, times_prep_last,  times_reset, freqs
    elif prep_window == "both":
        
        return (ersp_prep_first, times_prep_first,
                ersp_prep_last,  times_prep_last,
                ersp_reset,      times_reset, freqs)
    else:
        raise ValueError("prep_window must be 'first', 'last', or 'both'")

# --- Baseline extraction from RAW (uncorrected) TFRs in dB ---


BANDS = {"theta": (4, 7), "alpha": (8, 12), "beta": (13, 30)}


def load_baseline_bands_from_rawdb(
    pid,
    cond_key,
    dir_raw=r"D:\tfr_full_rawdb",
    baseline=(0.0, 2.0),
):
    """
    Load uncorrected TFR in dB for pid/cond_key and return baseline band power per channel.
    Requires files: TFR_{cond_key}_ALL-RAWDB.npy, -times.npy, -freqs.npy in dir_raw/pid/.
    Returns:
        baseline_bands_db: dict {band -> (n_channels,) array of dB}
        freqs, times
    """
    base = f"TFR_{cond_key}_ALL"
    p_dir = os.path.join(dir_raw, pid)

    rawdb = np.load(os.path.join(p_dir, f"{base}-RAWDB.npy"))  # (ch,f,t)
    times = np.load(os.path.join(p_dir, f"{base}-times.npy"))  # (t,)
    freqs = np.load(os.path.join(p_dir, f"{base}-freqs.npy"))  # (f,)

    # baseline is the first 0–2 s
    bl_mask = (times >= baseline[0]) & (times <= baseline[1])
    if not np.any(bl_mask):
        raise ValueError("No baseline samples found in requested window.")

    out = {}
    for name, (lo, hi) in BANDS.items():
        fmask = (freqs >= lo) & (freqs <= hi)
        # mean across freqs within band, then across baseline time - per-channel baseline dB
        band_db = rawdb[:, fmask, :].mean(
            axis=1)[:, bl_mask].mean(axis=1)  # (ch,)
        out[name] = band_db
    return out, freqs, times
