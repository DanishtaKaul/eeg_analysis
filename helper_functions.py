# -*- coding: utf-8 -*-
"""Shared helper functions for the gait pipeline: loading, label extraction, gap handling, and preprocessing"""
import os
import numpy as np
import pandas as pd
import kineticstoolkit.lab as ktk
from scipy.signal import find_peaks, detrend
import re
RAW_DIR = r"E:"

def get_existence_from_timeseries(ts_path):
    """
    Reads the TimeSeries file and extracts existence from event names.

    time series adds events like:
      START_T12_Present
      END_T12_Absent

    This function returns: "present" or "absent"

    """
    ts = ktk.load(ts_path)

    # Try to find START_T... first, then END_T...
    for ev in ts.events:
        name = str(ev.name)

        m = re.match(r"^START_T\d+_(.+)$", name)
        if m:
            existence = m.group(1).strip().lower()
            if existence in {"present", "absent"}:
                return existence

        m = re.match(r"^END_T\d+_(.+)$", name)
        if m:
            existence = m.group(1).strip().lower()
            if existence in {"present", "absent"}:
                return existence

    raise ValueError(
        f"Could not find Present/Absent START_T... or END_T... event in: {ts_path}"
    )


def findTrackersFolder(ppid, block):
    # Correct path format (use raw string or double backslashes)
    start_path = os.path.join(RAW_DIR, ppid, "UNITY")
    block_name = None

    # Loop through folders to find the correct block
    for d in os.listdir(start_path):
        if d.lower() == block.lower():
            block_name = d
            break

    s_path = os.path.join(RAW_DIR, ppid, "UNITY", block_name)
    s_folder = None
    for folder in os.listdir(s_path):
        folder_path = os.path.join(s_path, folder)
        if os.path.isdir(folder_path) and folder.lower().startswith('s'):
            s_folder = folder
            break

    if block_name is None or s_folder is None:
        input(f'Cannot find block folder "{block}" in path "{start_path}"')
        return None

    # Construct trackers folder
    path = os.path.join(RAW_DIR, ppid, "UNITY", block_name, s_folder)
    return path


def getSkippedTrialsDict(path_to_csv=r"D:\Gait_Analysis\Logs\skipped_trials_timing.csv"):
    """
    Returns dict: { BLOCK_KEY (upper) : set(trial_nums) }
    where BLOCK_KEY matches motion-tracking filename stem, e.g. "PID 11 AMBIENT EXPECTED"
    """
    if not os.path.exists(path_to_csv):
        print(f"Skip timing CSV not found: {path_to_csv}")
        return {}

    df = pd.read_csv(path_to_csv)
    if df.empty:
        return {}

    df["block_key"] = df["ppid"].astype(str).str.strip().str.upper()
    df["trial_num"] = pd.to_numeric(df["trial_num"], errors="coerce")
    df = df.dropna(subset=["block_key", "trial_num"])
    df["trial_num"] = df["trial_num"].astype(int)

    return df.groupby("block_key")["trial_num"].apply(set).to_dict()


def extract_light(block_id):
    block_id = block_id.upper()
    if "LIGHT" in block_id:
        return "LIGHT"
    elif "AMBIENT" in block_id:
        return "AMBIENT"
    elif "DARK" in block_id:
        return "DARK"
    else:
        return None


def extract_forewarn(_str):
    up_str = _str.upper()
    if "UNEXPECTED" in up_str:  # find unexpected first
        return "UNEXPECTED"
    elif "EXPECTED" in up_str:
        return "EXPECTED"
    else:
        return None


def max_nan_gap(x):
    """
    Returns the length of the longest consecutive NaN run in array x.
    """
    isnan = np.isnan(x)
    if not isnan.any():
        return 0

    # find runs of NaNs
    diffs = np.diff(np.concatenate(([0], isnan.astype(np.int8), [0])))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    return np.max(ends - starts)


def preprocess_motion_tracking_file(file_path, add_events=False):
    """
    
    Preprocess the motion tracking file and return both its df and ts

    Parameters
    ----------
    file_path : String
        Path to motion tracking file

    Returns
    -------
    df : Dataframe
        Dataframe of the mt data
    ts : ktk.timeseries
        Timeseries file of the data

    """
    KEEP_MARKERS = [
        ":Skeleton 001",  # pelvis
        "LFoot",  # left ankle
        "RFoot"
    ]

    # Get the mt file
    df = pd.read_csv(
        file_path,
        on_bad_lines="skip",
        skiprows=[0],
        low_memory=False
    )

    # Drop second column
    df.drop(df.index[1], inplace=True)

    # Rename 4 header rowes to one
    df.columns = df.iloc[0:3, :].apply(
        lambda x: " ".join(x.dropna().astype(str)), axis=0
    )
    df.drop(df.index[0:3], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop unlabelled columns
    df = df[df.columns.drop(list(df.filter(regex="Unlabeled")))]

    # Replace positions with pos[i]
    df.columns = (
        df.columns
        .str.replace("Position X", "Pos[0]")
        .str.replace("Position Y", "Pos[1]")
        .str.replace("Position Z", "Pos[2]")
    )

    # Rename time columns
    df = df.rename(columns={
        "Time (Seconds)": "Time",
        "Name Time (Seconds)": "Time"
    })

    # Drop first column and set time to index
    df = df.drop(df.columns[0], axis=1)
    df = df.astype(float)
    df = df.set_index("Time")

    # Only keep the markers specified
    pattern = "|".join(KEEP_MARKERS)
    df = df.loc[:, df.columns.str.contains(pattern)]

    discard = False
    for col in df.columns:
        if max_nan_gap(df[col]) > 15:
            discard = True
            break
        df[col] = df[col].interpolate(
            method="linear",
            limit=15,
            limit_direction="both"
        )

    if discard:
        # fill any remaining gaps with 0 or the last valid value
        input(f'Gap found > 15 | Gap size: (max_nan_gap(df[col]))')
        df = df.ffill().bfill()

    pelvis_col = [
        c for c in df.columns
        if ":Skeleton 001" in c and "Pos[2]" in c
    ][0]

    lfoot_col = [
        c for c in df.columns
        if "LFoot" in c and "Pos[2]" in c
    ][0]

    rfoot_col = [
        c for c in df.columns
        if "RFoot" in c and "Pos[2]" in c
    ][0]

    pelvis_ap = df[pelvis_col].values
    lfoot_ap = df[lfoot_col].values
    rfoot_ap = df[rfoot_col].values

    ap_rel = detrend(lfoot_ap - pelvis_ap, type="linear")
    rp_rel = detrend(rfoot_ap - pelvis_ap, type="linear")

    time = df.index.values

    # KTK TimeSeries (events only)
    ts = ktk.TimeSeries(
        time=time,
        data={
            "pelvis_ap": pelvis_ap,
            "lfoot_ap": lfoot_ap,
            "rfoot_ap": rfoot_ap,
            "LFoot_relPelvis": ap_rel,
            "RFoot_relPelvis": rp_rel
        }
    )

    if add_events:

        """
        PEAKS
        """

        peaks, _ = find_peaks(
            ap_rel,
            prominence=0.3,
            distance=120
        )

        r_peaks, _ = find_peaks(
            rp_rel,
            prominence=0.3,
            distance=120
        )

        l_steps = len(peaks)
        r_steps = len(r_peaks)
        total_steps = len(peaks) + len(r_peaks)

        # Add peaks to ts events
        for idx in peaks:
            ts = ts.add_event(time[idx], "HS_L")

        for idx in r_peaks:
            ts = ts.add_event(time[idx], "HS_R")

        """
        GROUPS
        """

        all_peaks = np.sort(
            np.concatenate([peaks, r_peaks])
        )
        all_peak_times = time[all_peaks]

        GROUP_GAP_SEC = 1.5

        groups = []
        start_idx = 0
        MIN_STEPS_PER_GROUP = 4

        # Create groups
        for i in range(1, len(all_peak_times)):
            if all_peak_times[i] - all_peak_times[i - 1] > GROUP_GAP_SEC:
                end_idx = i - 1

                # number of steps in this group
                n_steps = end_idx - start_idx + 1

                if n_steps >= MIN_STEPS_PER_GROUP:
                    groups.append((start_idx, end_idx))

                start_idx = i

        end_idx = len(all_peak_times) - 1
        n_steps = end_idx - start_idx + 1

        end_idx = len(all_peak_times) - 1
        n_steps = end_idx - start_idx + 1

        if n_steps >= MIN_STEPS_PER_GROUP:
            groups.append((start_idx, end_idx))

        group_windows = []

        for g_start, g_end in groups:
            w_start = all_peak_times[g_start]
            w_end = all_peak_times[g_end]

            # Add 1 sec of padding to all groups
            PAD_SEC = 1.0
            t_min = time[0]
            t_max = time[-1]

            w_start = max(w_start - PAD_SEC, t_min)
            w_end = min(w_end + PAD_SEC,   t_max)

            group_windows.append((w_start, w_end))

        group_windows = group_windows[-41:]

        for w_start, w_end in reversed(group_windows):

            ts = ts.add_event(w_start, f"GROUP_START")
            ts = ts.add_event(w_end,   f"GROUP_END")

    return df, ts
