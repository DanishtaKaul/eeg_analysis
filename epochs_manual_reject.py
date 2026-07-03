# -*- coding: utf-8 -*-
"""
Manual rejection script-inspect each condition visually and manually drop remaining bad epochs.
"""

import os
import mne
import csv


def clean_epochs_manually(ppid, manual_drops):
    """
    Loads the time-warped .fif epochs per condition for a participant,
    allows manual inspection and drops bad epochs.

    Parameters
    ----------
    ppid : str
        Participant ID 
    manual_drops : dict
        Dictionary like {"LEP": [0, 2], "DEP": [4]} where each key is a
        condition and values are the trial numbers of epochs to drop manually.

    Returns
    -------
    Saves cleaned .fif files and a CSV log of dropped epochs.
    """

    # Path to participant's saved aligned epochs 

    base_dir = os.path.join(r"D:\aligned_epochs", ppid)
    cleaned_epochs = {}  # to store cleaned data to use it later

    # Loop through each condition (e.g., LEP, DEP)
    for cond in os.listdir(base_dir):
        cond_path = os.path.join(base_dir, cond)

        # Construct full path to the .fif file
        fif_file = os.path.join(cond_path, f"aligned_epochs_{cond}.fif")

        # Skip if file not found
        if not os.path.exists(fif_file):
            print(f"[{ppid}] Missing file for {cond}, skipping.")
            continue

        print(f"\nLoading {cond}: {fif_file}")
        epochs = mne.read_epochs(fif_file, preload=True)

        # for i, row in epochs.metadata.iterrows():
        # print(f"Epoch {i} → trial_num: {row['trial_num']}")

        # === STEP 1: Plot the epochs for visual inspection ===
        # epochs.plot(n_channels=30, title=f"{ppid} — {cond} (Inspect)")

        # === STEP 2: Drop manually flagged epochs (by trial_num) ===
        if cond in manual_drops:
            bad_trial_nums = manual_drops[cond]
            if bad_trial_nums and epochs.metadata is not None:
                mask = ~epochs.metadata['trial_num'].isin(bad_trial_nums)
                n_drop = (~mask).sum()
                print(
                    f"[{ppid}] Dropping {n_drop} epochs from {cond} (trial_nums: {bad_trial_nums})")
                epochs = epochs[mask.values]

        # === STEP 3: Save the cleaned epochs back ===
        epochs.save(fif_file, overwrite=True)
        cleaned_epochs[cond] = epochs

    # === STEP 4: Save a CSV log of which epochs were dropped ===
    log_path = os.path.join(base_dir, "manual_drops.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Dropped_Trial_Num"])
        for cond, trial_nums in manual_drops.items():
            for t in trial_nums:
                writer.writerow([cond, t])

    print(f"[{ppid}] Manual rejection complete and saved.")

    return cleaned_epochs


# ========= ENTRY POINT =============
if __name__ == "__main__":
    from scripts.manual_reject_config import participants_to_clean  # central config

    for pid, drop_dict in participants_to_clean.items():
        clean_epochs_manually(pid, drop_dict)
