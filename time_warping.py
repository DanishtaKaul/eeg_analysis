"""Time-warp each epoch's preparation phase to a common duration so epochs can be compared across trials"""

import numpy as np
import mne
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import csv
import config
import os
import pandas as pd

from collections import defaultdict


def time_warp_preparation(raw, crossing_epochs, kept_indices):
    """
    Warp each epoch's prep segment to a global median duration (per condition),
    then rebuild the full epoch as baseline + warped prep + reset. Trials whose
    stretch factor is a z-score outlier (|z| > 2.5) are excluded. Returns the
    aligned epochs grouped by condition.
    """

    sfreq = raw.info['sfreq']
    pid = raw.info['subject_info'].get('pid_str', 'unknown_pid')

    total_ce = len(crossing_epochs)
    skipped_ce = sum(ep.get('SKIPPED', False) for ep in crossing_epochs)
    bad_hits = [i for i in kept_indices if i <
                total_ce and crossing_epochs[i].get('SKIPPED', False)]
    print(f"[AUDIT {pid}] crossing_epochs={total_ce} | skipped={skipped_ce} | kept={len(kept_indices)} | kept∩skipped={len(bad_hits)} (e.g., {bad_hits[:10]})")

    

    # 1) Build the SKIP-filtered list that AutoReject indexed into
    attempted = [ep for ep in crossing_epochs if not ep.get('SKIPPED', False)]

    # guard: AR indices must be valid for the attempted list
    if len(kept_indices) and max(kept_indices) >= len(attempted):
        raise ValueError(
            f"[{pid}] kept_indices max={max(kept_indices)} out of range for attempted={len(attempted)}"
        )

    # Use AR indices against attempted (NOT crossing_epochs)
    epochs = [attempted[i] for i in kept_indices]

    # prove none of these point to SKIPPED entries in the original array
    attempted_map = [j for j, ep in enumerate(
        crossing_epochs) if not ep.get('SKIPPED', False)]
    bad_after_fix = [i for i in kept_indices
                     if crossing_epochs[attempted_map[i]].get('SKIPPED', False)]
    print(
        f"[AUDIT {pid}] kept∩skipped after fix = {len(bad_after_fix)} (should be 0)")

    # input("..")

    aligned_data = []
    warped_factors = []
    original_lengths = []
    median_keys = []
    target_lengths = []
    final_indices = []

    # Pre-load all global medians into a dict
    global_medians = {}
    with open(config.median_prep_csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            global_medians[row['label'].strip()] = float(row['median'])

    # Loop through epochs
    for idx, ep in enumerate(epochs):
        data = ep['data']
        prep_sample = ep['prep_start_sample_offset']
        cs_off = ep['crossing_start_sample_offset']
        ce_off = ep['crossing_end_sample_offset']
        reset_end_off = ep['reset_end_sample_offset']
        key = ep['median_key']

        

        prep = data[:, prep_sample:cs_off]
        # cross = data[:, cs_samp:ce_off]
        reset = data[:, ce_off:reset_end_off]
        baseline_data = ep['baseline_data']

        # Calculate new reset length
        # cross_len = cross.shape[1]

        original_len = prep.shape[1]

        # Lookup the target length
        if key not in global_medians:
            raise ValueError(f"No median for key '{key}' in CSV")
        target_len = int(round(global_medians[key] * sfreq))

        if prep.shape[1] <= 0:
            print(ep)
            input(
                f'prep shape : {prep.shape[1]} | prep_sample : {prep_sample} | cs_off : {cs_off} ')

        # Warp prep
        orig_x = np.linspace(0, 1, original_len)
        targ_x = np.linspace(0, 1, target_len)
        warped = np.zeros((prep.shape[0], target_len))
        for ch in range(prep.shape[0]):
            f_interp = interp1d(
                orig_x, prep[ch], kind='linear', fill_value='extrapolate')
            warped[ch] = f_interp(targ_x)

        # Build full epoch
        full = np.hstack((baseline_data, warped, reset))

        # Debug statements about the epoch structure
        print(f"\n ==== {key} ====")
        print(
            f" warped : {warped.shape[1]} |  reset : {reset.shape[1]} |")
        print(f"Total epoch samples: {full.shape[1]}")

        if cs_off < 0:
            input(ep['trial'])

        # Record
        aligned_data.append(full)
        warped_factors.append(target_len / original_len)
        original_lengths.append(original_len)
        median_keys.append(key)
        target_lengths.append(target_len)
        final_indices.append(idx)

    # 2) Summary CSV & plot
    # --- STEP 4: Compute z-scores and flag outliers ---
    mean_sf = np.mean(warped_factors)
    std_sf = np.std(warped_factors)
    z_scores = [(sf - mean_sf) / std_sf for sf in warped_factors]

    outlier_types = []
    for z in z_scores:
        if z > 2.5:
            outlier_types.append("Too Stretched")
        elif z < -2.5:
            outlier_types.append("Too Compressed")
        else:
            outlier_types.append("OK")

    # Remove "Too Stretched" and "Too Compressed" trials ---
    valid_indices = [i for i, flag in enumerate(outlier_types) if flag == "OK"]

    # Store full lists before filtering for logging excluded trials later
    warped_factors_all = warped_factors.copy()
    z_scores_all = z_scores.copy()
    original_lengths_all = original_lengths.copy()
    target_lengths_all = target_lengths.copy()
    median_keys_all = median_keys.copy()

    # Filter all relevant lists to exclude outliers
    aligned_data = [aligned_data[i] for i in valid_indices]
    warped_factors = [warped_factors[i] for i in valid_indices]
    original_lengths = [original_lengths[i] for i in valid_indices]
    median_keys = [median_keys[i] for i in valid_indices]
    target_lengths = [target_lengths[i] for i in valid_indices]
    final_indices = [final_indices[i] for i in valid_indices]

    print(
        f"Excluded {len(outlier_types) - len(valid_indices)} outlier trials based on z-score")

    # 3) Build MNE EpochsArray
    grouped_data = defaultdict(list)
    grouped_events = defaultdict(list)
    grouped_meta = defaultdict(list)

    for i, full in enumerate(aligned_data):
        ep = epochs[final_indices[i]]
        crossing_ep = ep
        key = ep['median_key']

        # Split warped prep and reset
        warped_prep_len = target_lengths[i]
        warped_prep = full[:, :warped_prep_len]
        reset = full[:, warped_prep_len:]

        # Store into epoch metadata
        crossing_ep['warped_data'] = full
        crossing_ep['warped_prep'] = warped_prep
        crossing_ep['warped_prep_len'] = warped_prep_len
        crossing_ep['reset_data'] = reset
        crossing_ep['reset_len'] = reset.shape[1]

        grouped_data[key].append(full)
        grouped_events[key].append([ep['epoch_start_sample'], 0, 1])
        grouped_meta[key].append(crossing_ep)

    aligned_epochs_by_condition = {}

    for key in grouped_data:
        shapes = [t.shape for t in grouped_data[key]]
        unique_shapes = set(shapes)
        print(f"[{key}] Unique shapes:", unique_shapes)

        shapes = [t.shape for t in grouped_data[key]]
        unique_shapes = set(shapes)

        if len(unique_shapes) > 1:
            print(
                f"[ERROR] Condition '{key}' in PID {pid} has mismatched trial shapes:")
            for i, shape in enumerate(shapes):
                print(f"  Trial {i}: shape {shape}")
            input("Press Enter to continue...")

        expected_shape = None
        for i, arr in enumerate(grouped_data[key]):
            print(f"[{pid} | {key}] Trial {i} shape: {arr.shape}")
            if expected_shape is None:
                expected_shape = arr.shape
            elif arr.shape != expected_shape:
                print(
                    f"[ERROR] Mismatched shape at trial {i}: expected {expected_shape}, got {arr.shape}")

        data = np.stack(grouped_data[key])
        events = np.array(grouped_events[key])
        metadata_df = pd.DataFrame(grouped_meta[key])

        aligned_epochs = mne.EpochsArray(
            data=data,
            info=raw.info,
            events=events,
            tmin=0.0,
            event_id={'aligned_prep': 1},
            metadata=metadata_df
        )
        aligned_epochs_by_condition[key] = aligned_epochs

    # --- STEP 5: Save warp summary CSV with z-scores ---
    output_dir = os.path.join("timewarp", pid)
    # Create folder if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"warp_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'stretch', 'z_score', 'outlier_type',
                         'orig_samp', 'target_samp', 'label'])
        for i, (sf, z, flag, orig, targ, label) in enumerate(zip(
                warped_factors, z_scores, outlier_types,
                original_lengths, target_lengths, median_keys)):
            writer.writerow([i, sf, z, flag, orig, targ, label])
        writer.writerow([])
        writer.writerow(
            [f"Total outliers (|z| > 2.5): {sum(1 for f in outlier_types if f != 'OK')}"])

    # --- STEP 6: Save excluded outliers to a separate CSV ---
    outlier_trials = []
    for i, flag in enumerate(outlier_types):
        if flag != "OK":
            outlier_trials.append({
                'pid': pid,
                'epoch': i,
                'outlier_type': flag,
                'stretch': warped_factors_all[i],
                'z_score': z_scores_all[i],
                'orig_samp': original_lengths_all[i],
                'target_samp': target_lengths_all[i],
                'label': median_keys_all[i]
            })

    if outlier_trials:
        outlier_path = os.path.join(
            output_dir, f"excluded_zscore_trials_{pid}.csv")
        fieldnames = list(outlier_trials[0].keys())
        with open(outlier_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(outlier_trials)
            writer.writerow({})
            writer.writerow({'pid': f"Total excluded: {len(outlier_trials)}"})

    if outlier_trials:
        outlier_path = os.path.join(
            "timewarp", f"excluded_zscore_trials_all.csv")
        with open(outlier_path, "a", newline="") as f:
            fieldnames = outlier_trials[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(outlier_trials)
            writer.writerow({})
            writer.writerow({'pid': f"Total excluded: {len(outlier_trials)}"})

    # --- STEP 7: Plot stretch factors ---
    plt.figure()
    plt.plot(warped_factors, 'o-')
    plt.axhline(1, linestyle='--', color='gray')
    plt.title(f'Prep Stretch Factors — {pid}')
    plt.xlabel('Epoch idx')
    plt.ylabel('stretch factor')
    plt.savefig(os.path.join(output_dir, f"warp_plot_{pid}.png"), dpi=150)
    plt.close()


    return aligned_epochs_by_condition
