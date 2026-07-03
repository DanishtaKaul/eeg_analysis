

# -*- coding: utf-8 -*-
"""
Compute speed and cadence from VALID EEG trials only.

Pipeline:
1) Get valid EEG trials (suspicious removed + EEG-matched only)
2) Extract individual strides and cadence per trial
3) Remove entire trials failing cadence threshold
4) Remove individual bad strides 
5) Remove trials with < 2 strides remaining
6) Pool per PID x light x obstacle: mean speed across surviving strides,
   mean cadence across surviving trials
7) Save speed_mean and cadence per PID x light x obstacle
"""

from helper_functions import extract_light, extract_forewarn, get_existence_from_timeseries
from eeg_final_dataset_for_gait_analysis_with_obstacle import get_valid_eeg_trials
import kineticstoolkit.lab as ktk
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, r"D:\Gait_Analysis")


# ======================================================
# AGE GROUP DEFINITIONS
# ======================================================

Young = {
    "PID 3", "PID 4", "PID 5", "PID 6", "PID 7", "PID 8", "PID 9",
    "PID 10", "PID 11", "PID 14", "PID 15", "PID 18", "PID 19",
    "PID 20", "PID 23", "PID 26", "PID 30", "PID 31", "PID 35",
    "PID 50", "PID 57", "PID 58"
}

Old = {
    "PID 13", "PID 17", "PID 22", "PID 24", "PID 25", "PID 27",
    "PID 28", "PID 29", "PID 32", "PID 33", "PID 36", "PID 38",
    "PID 41", "PID 43", "PID 45", "PID 46", "PID 49",
    "PID 52", "PID 55"
}

# ======================================================
# HELPER FUNCTIONS
# ======================================================


def get_hs_times(ts):
    hs_L = sorted([ev.time for ev in ts.events if ev.name == "HS_L"])
    hs_R = sorted([ev.time for ev in ts.events if ev.name == "HS_R"])
    return hs_L, hs_R


def get_indices_from_times(time_array, event_times):
    return [int(np.argmin(np.abs(time_array - t))) for t in event_times]


def extract_raw_strides(ts):
    hs_L, hs_R = get_hs_times(ts)

    if len(hs_L) < 2:
        raise ValueError("Not enough HS_L events to compute stride metrics.")

    time = ts.time
    lfoot_ap = ts.data["lfoot_ap"]

    hs_L_indices = get_indices_from_times(time, hs_L)

    stride_lengths = []
    for i in range(len(hs_L_indices) - 1):
        idx1 = hs_L_indices[i]
        idx2 = hs_L_indices[i + 1]
        stride_lengths.append(abs(lfoot_ap[idx2] - lfoot_ap[idx1]))

    stride_lengths = np.array(stride_lengths)
    stride_times = np.diff(hs_L)

    min_len = min(len(stride_lengths), len(stride_times))
    stride_lengths = stride_lengths[:min_len]
    stride_times = stride_times[:min_len]

    speeds = stride_lengths / stride_times

    return stride_lengths, stride_times, speeds


def compute_cadence(ts):
    hs_L, hs_R = get_hs_times(ts)

    first_step = min(hs_L[0], hs_R[0])
    last_step = max(hs_L[-1], hs_R[-1])
    duration = last_step - first_step

    if duration <= 0:
        raise ValueError(
            "Non-positive walking duration in cadence calculation.")

    total_steps = len(hs_L) + len(hs_R)
    return (total_steps / duration) * 60


# ======================================================
# OUTLIER THRESHOLDS
# ======================================================

stride_thresholds = {
    "stride_length": (0.6, 1.7),
    "stride_time": (0.85, 2.0),
    "speed": (0.3, 2.0),
}

cadence_threshold = (60, 141)

# ======================================================
# GET VALID EEG TRIALS
# ======================================================

valid_eeg_trials = get_valid_eeg_trials()
print(f"Number of valid EEG trials: {len(valid_eeg_trials)}")

# ======================================================
# MAIN LOOP — extract strides + cadence per trial
# ======================================================

stride_rows = []
trial_rows = []

for i, ts_path in enumerate(valid_eeg_trials, 1):

    if i % 100 == 0 or i == 1:
        print(f"[{i}/{len(valid_eeg_trials)}] Processing {os.path.basename(ts_path)}")

    ts = ktk.load(ts_path)

    pid = os.path.basename(os.path.dirname(ts_path))

    if pid in Young:
        age = "young"
    elif pid in Old:
        age = "old"
    else:
        raise ValueError(f"PID {pid} not found in age groups.")

    light = str(extract_light(ts_path)).lower()
    forewarn = str(extract_forewarn(ts_path)).lower()
    existence = get_existence_from_timeseries(ts_path)
    obstacle = f"{forewarn}_{existence}"
    trial_file = os.path.basename(ts_path)

    # Extract strides
    stride_lengths, stride_times, speeds = extract_raw_strides(ts)

    for j in range(len(stride_lengths)):
        stride_rows.append({
            "pid": pid,
            "age": age,
            "light": light,
            "obstacle": obstacle,
            "trial_file": trial_file,
            "stride_idx": j,
            "stride_length": stride_lengths[j],
            "stride_time": stride_times[j],
            "speed": speeds[j],
        })

    # Cadence per trial
    cadence = compute_cadence(ts)

    trial_rows.append({
        "pid": pid,
        "age": age,
        "light": light,
        "obstacle": obstacle,
        "trial_file": trial_file,
        "cadence": cadence,
    })


# ======================================================
# BUILD DATAFRAMES
# ======================================================

stride_df = pd.DataFrame(stride_rows)
trial_df = pd.DataFrame(trial_rows)

print(f"\nTotal raw strides: {len(stride_df)}")
print(f"Total trials: {len(trial_df)}")


# ======================================================
# STEP 1: REMOVE ENTIRE TRIALS FAILING CADENCE THRESHOLD
# ======================================================

print("\n=== Step 1: Trial-level outlier removal (cadence) ===")

trial_df["cadence_fail"] = (
    (trial_df["cadence"] <= cadence_threshold[0]) |
    (trial_df["cadence"] >= cadence_threshold[1])
)

n_cadence_fail = trial_df["cadence_fail"].sum()
print(f"Cadence failures: {n_cadence_fail}")

excluded_trials = trial_df[trial_df["cadence_fail"]].copy()
excluded_trials.to_csv(
    r"D:\Gait_Analysis\eeg_excluded_trials_cadence_with_obstacle.csv", index=False)
print("Saved: eeg_excluded_trials_cadence_with_obstacle.csv")

bad_trial_files = set(trial_df[trial_df["cadence_fail"]]["trial_file"])
trial_df = trial_df[~trial_df["cadence_fail"]].copy()
stride_df = stride_df[~stride_df["trial_file"].isin(bad_trial_files)].copy()

print(f"Remaining trials: {len(trial_df)}")
print(f"Remaining strides: {len(stride_df)}")


# ======================================================
# STEP 2: REMOVE INDIVIDUAL BAD STRIDES
# ======================================================

print("\n=== Step 2: Stride-level outlier removal ===")

stride_df["stride_length_fail"] = (
    (stride_df["stride_length"] <= stride_thresholds["stride_length"][0]) |
    (stride_df["stride_length"] >= stride_thresholds["stride_length"][1])
)
stride_df["stride_time_fail"] = (
    (stride_df["stride_time"] <= stride_thresholds["stride_time"][0]) |
    (stride_df["stride_time"] >= stride_thresholds["stride_time"][1])
)
stride_df["speed_fail"] = (
    (stride_df["speed"] <= stride_thresholds["speed"][0]) |
    (stride_df["speed"] >= stride_thresholds["speed"][1])
)
stride_df["any_stride_fail"] = (
    stride_df["stride_length_fail"] |
    stride_df["stride_time_fail"] |
    stride_df["speed_fail"]
)

for var in ["stride_length", "stride_time", "speed"]:
    print(f"{var}: {stride_df[f'{var}_fail'].sum()} strides removed")

n_stride_fail = stride_df["any_stride_fail"].sum()
print(f"Total strides removed (unique): {n_stride_fail}")

excluded_strides = stride_df[stride_df["any_stride_fail"]].copy()
excluded_strides.to_csv(
    r"D:\Gait_Analysis\eeg_excluded_strides_with_obstacle.csv", index=False)
print("Saved: eeg_excluded_strides_with_obstacle.csv")

stride_df = stride_df[~stride_df["any_stride_fail"]].copy()
print(f"Remaining strides: {len(stride_df)}")


# ======================================================
# STEP 3: REMOVE TRIALS WITH < 2 STRIDES REMAINING
# ======================================================

print("\n=== Step 3: Remove trials with < 2 strides remaining ===")

strides_per_trial = stride_df.groupby(
    "trial_file").size().reset_index(name="n_strides")
thin_trials = set(
    strides_per_trial[strides_per_trial["n_strides"] < 2]["trial_file"])

all_remaining_trials = set(trial_df["trial_file"])
trials_with_strides = set(stride_df["trial_file"])
no_strides_left = all_remaining_trials - trials_with_strides

bad_thin_trials = thin_trials | no_strides_left
print(f"Trials with < 2 strides remaining: {len(bad_thin_trials)}")
if len(bad_thin_trials) > 0:
    thin_info = trial_df[trial_df["trial_file"].isin(bad_thin_trials)].copy()
    thin_info.to_csv(
        r"D:\Gait_Analysis\eeg_excluded_trials_too_few_strides_with_obstacle.csv", index=False)
    print("Saved: eeg_excluded_trials_too_few_strides_with_obstacle.csv")

stride_df = stride_df[~stride_df["trial_file"].isin(bad_thin_trials)].copy()
trial_df = trial_df[~trial_df["trial_file"].isin(bad_thin_trials)].copy()

print(f"Remaining trials: {len(trial_df)}")
print(f"Remaining strides: {len(stride_df)}")


# ======================================================
# POOL PER PID x LIGHT x OBSTACLE
# ======================================================

print("\n=== Pooling per PID x light x obstacle ===")

group_cols = ["pid", "age", "light", "obstacle"]

# Speed: pool all strides per PID x light x obstacle, take mean
speed_agg = stride_df.groupby(group_cols).agg(
    speed_mean=("speed", "mean"),
    n_strides=("speed", "size"),
).reset_index()

# Cadence: average across surviving trials per PID x light x obstacle
cadence_agg = trial_df.groupby(group_cols).agg(
    cadence=("cadence", "mean"),
    n_trials=("cadence", "size"),
).reset_index()


# ======================================================
# MERGE AND SAVE
# ======================================================

final_df = speed_agg.merge(cadence_agg, on=group_cols, how="outer")

print(f"\nFinal dataset: {len(final_df)} rows (PID x light x obstacle)")
print(f"Unique PIDs: {final_df['pid'].nunique()}")


# ======================================================
# CHECKS
# ======================================================

print("\n=== CHECKS ===")

checks = {
    "Speed (m/s)": ("speed_mean", 0.3, 2.0),
    "Cadence (steps/min)": ("cadence", 60, 141),
}

print(f"{'Variable':<25} {'Min':>10} {'Max':>10} {'Threshold':>20} {'OK?':>5}")
print("-" * 75)

for label, (col, lo, hi) in checks.items():
    mn = final_df[col].min()
    mx = final_df[col].max()
    thresh = f"{lo} - {hi}"
    ok = "YES" if mn > lo and mx < hi else "NO"
    print(f"{label:<25} {mn:>10.3f} {mx:>10.3f} {thresh:>20} {ok:>5}")

print(f"\nMin strides per cell: {final_df['n_strides'].min()}")
print(f"Min trials per cell: {final_df['n_trials'].min()}")

print("\nCells with fewest strides:")
print(final_df.nsmallest(10, "n_strides")[
      group_cols + ["n_strides", "n_trials"]].to_string(index=False))

# Save combined CSV for mixed models
speed_final = final_df[["pid", "age", "light", "obstacle",
                        "speed_mean", "n_strides"]].copy()
speed_final = speed_final.rename(columns={"speed_mean": "speed"})
speed_final.to_csv(
    r"D:\Gait_Analysis\eeg_gait_speed_with_obstacle_final.csv", index=False)
print("Saved: eeg_gait_speed_with_obstacle_final.csv")

cadence_final = final_df[["pid", "age", "light",
                          "obstacle", "cadence", "n_trials"]].copy()
cadence_final.to_csv(
    r"D:\Gait_Analysis\eeg_gait_cadence_with_obstacle_final.csv", index=False)
print("Saved: eeg_gait_cadence_with_obstacle_final.csv")

retained = final_df[["pid", "age", "light", "obstacle", "n_trials"]].copy()
retained.to_csv(
    r"D:\Gait_Analysis\eeg_gait_retained_trials_per_light_obstacle.csv", index=False)
print("Saved: eeg_gait_retained_trials_per_light_obstacle.csv")

print("\nFinished.")
