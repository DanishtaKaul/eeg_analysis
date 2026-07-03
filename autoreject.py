
"""Clean epochs with AutoReject"""

import mne
import numpy as np
from config import pre_crossing_sec
from autoreject import AutoReject  
import os
import csv
import pandas as pd


def autoreject(raw, crossing_epochs):
    """
    Clean EEG epochs with AutoReject: interpolate bad channels and reject bad epochs.
    Returns the cleaned epochs and the indices of the epochs that were kept.
    """

    # ----------------------------------
    # STEP 1: Extract data + events
    # ----------------------------------

    epoch_data = []  # Temporarily hold EEG for each epoch
    events = []      # Store one event per epoch for MNE
    max_len = 0

    # remove skipped epochs
    crossing_epochs = [e for e in crossing_epochs if not e["SKIPPED"]]

    # Loop over each epoch (trial) in the list of crossing_epochs
    for index, epoch in enumerate(crossing_epochs):
        data = epoch['data']  # use all of epoch EEG segment

        # get eeg data for this trial
        # Save this trial’s EEG data into a list 
        epoch_data.append(data)

        # Create a synthetic event marker for MNE
        # Format: [sample index, 0, event ID]
        events.append([epoch['epoch_start_sample'], 0, 1])

    # Stack all epochs into 3D NumPy array:
    # Shape = (n_epochs, n_channels, n_samples)

    # Print shape of first epoch
    print("Shape of first epoch['data']:", epoch_data[0].shape)
    print("raw.info['nchan']:", raw.info['nchan'])

    # Print diagnostic info about epoch lengths
    lengths = [d.shape[1] for d in epoch_data]
    unique_lengths = set(lengths)
    print(f"Unique epoch lengths (samples): {unique_lengths}")

    if len(unique_lengths) > 1:
        print("WARNING: Not all epochs have the same number of time points!")
        for i, length in enumerate(lengths):
            print(f"Epoch {i} length: {length} samples")

    data = np.stack(epoch_data)

    # Convert events list to NumPy array for MNE compatibility
    events = np.array(events)

    # ----------------------------------
    # STEP 3: Convert to MNE EpochsArray
    # ----------------------------------

    epochs = mne.EpochsArray(  # Create a new EpochsArray object 
        data=data,
        info=raw.info,           # Reuse metadata (channels, sfreq, etc.)
        # Provide the synthetic event markers (one per epoch) that tell MNE where each epoch starts in the original recording
        events=events,
        tmin=-pre_crossing_sec,  # time from the start of the epoch to the crossing event
        event_id={"crossing": 1}  # Label for all events
    )
    # -------------------------------------------
    # STEP 4: Run AutoReject to clean EEG epochs
    # -------------------------------------------

    # Initialize AutoReject with parameter options for interpolation,
    ar = AutoReject(
        # Maximum number of channels AutoReject may interpolate per epoch
        n_interpolate=[1, 2],
        random_state=42              # Fixes random seed for reproducibility of results
    )

    # Fit AutoReject on epochs, performing two key operations:
    # 1. Interpolate bad channels 
    # 2. Reject entire epochs if too many channels are bad or data is too noisy
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    # Determine indices of epochs that were NOT rejected (kept for analysis)
    kept_indices = np.where(~reject_log.bad_epochs)[0]

    # Visualize detailed reject/interpolation matrix
    reject_log.plot('horizontal')

    print(f"Optimal n_interpolate: {ar.n_interpolate_}")
    # Number of sensors that must agree to mark a channel as bad
    print(f"Optimal consensus: {ar.consensus_}")

    # STEP 5: Visualization after AutoReject
    epochs_clean.plot_psd(fmax=40)
    # epochs_clean.plot(n_channels=32,
    # title='Cleaned Epochs After AutoReject')
    # evoked = epochs_clean.average()
    # evoked.plot(title='Evoked Response After AutoReject')


# =============================================================================
#     Save the epochs and kept indecies
# =============================================================================

    # Get participant ID 
    ppid = raw.info['subject_info'].get('pid_str', 'unknown_pid')

    # Create output directory
    output_dir = f"autoreject/{ppid}"
    os.makedirs(output_dir, exist_ok=True)

    # Save kept indices to a CSV file
    kept_indices_path = os.path.join(output_dir, f"kept_indices_{ppid}.csv")
    with open(kept_indices_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Kept Epoch Index"])
        for idx in kept_indices:
            writer.writerow([idx])

    epochs_clean_path = os.path.join(output_dir, f"epochs_clean.fif")
    epochs_clean.save(epochs_clean_path, overwrite=True)
    print(f"Kept indices saved to: {output_dir}")

    # Save the reject_log
    reject_log_df = pd.DataFrame(
        reject_log.labels, columns=raw.info['ch_names'])
    reject_log_df.insert(0, 'bad_epoch', reject_log.bad_epochs)

    reject_log_path = os.path.join(output_dir, f"reject_log_{ppid}.csv")
    reject_log_df.to_csv(reject_log_path, index=False)
    print(f"Reject log saved to: {reject_log_path}")

    # Save cleaned epochs to file
    epochs_clean_path = os.path.join(output_dir, f"epochs_clean.fif")
    epochs_clean.save(epochs_clean_path, overwrite=True)

    # Find which epochs were dropped
    dropped_indices = np.where(reject_log.bad_epochs)[0]

    print("\n========== AutoReject Summary ==========")
    print(f"Total epochs: {len(reject_log.bad_epochs)}")
    print(f"Total dropped: {len(dropped_indices)}")
    print("Dropped indices:", dropped_indices)
    print("========================================")
    print(f"\n Total epochs dropped: {len(dropped_indices)}")

    print("\nDropped epochs per condition:")
    

    # Pull condition info from dropped epochs
    if len(dropped_indices) > 0:
        dropped_info = []
        for idx in dropped_indices:
            ep = crossing_epochs[idx]
            dropped_info.append({
                'LightCondition': ep['light_condition'],
                'ForewarningLevel': ep['forewarn_condition'],
                'ExistanceLevel': ep['existance_condition']
            })

        # Convert to DataFrame
        dropped_meta_df = pd.DataFrame(dropped_info)

        # Count how many dropped per condition
        drop_summary = (
            dropped_meta_df
            .groupby(['LightCondition', 'ForewarningLevel', 'ExistanceLevel'])
            .size()
            .reset_index(name='n_dropped')
            .sort_values('n_dropped', ascending=False)
        )

        print("Dropped epochs per condition:\n", drop_summary)

        # Save to CSV
        drop_summary.to_csv(
            os.path.join(output_dir, f"dropped_summary_{ppid}.csv"),
            index=False
        )

        
        print(f"WARNING: {len(dropped_indices)} epochs dropped for {ppid}")

    # Save total summary to CSV
    total_summary_path = os.path.join(
        output_dir, f"total_dropped_summary_{ppid}.csv")
    with open(total_summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Participant", "Total_Epochs",
                        "Dropped_Epochs", "Kept_Epochs"])
        writer.writerow([ppid, len(reject_log.bad_epochs),
                        len(dropped_indices), len(kept_indices)])

    print(f"[INFO] Total dropped summary saved to {total_summary_path}")
    # Return cleaned EEG epochs (with interpolated channels) and indices of kept epochs
    return epochs_clean, kept_indices
