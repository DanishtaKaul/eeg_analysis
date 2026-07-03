# -*- coding: utf-8 -*-
"""Compute and save full-channel ERSP maps per condition (for cluster tests)"""

# === compute_full_tfrs.py ===


import os
import numpy as np
import mne
from scripts.time_frequency_plots import compute_ersp
# Reuse existing baseline correction logic


def compute_full_channel_tfrs(ppid, aligned_epochs_by_condition, out_dir=r"D:\tfr_full"):
    freqs = np.linspace(4, 30, 30)
    n_cycles = freqs / 2
    participant_dir = os.path.join(out_dir, ppid)
    os.makedirs(participant_dir, exist_ok=True)

    for cond, epochs in aligned_epochs_by_condition.items():  # loop over all conditions
        print(f"[{ppid} | {cond}] Computing full-channel TFR")

        # Picks all EEG channels
        picks = mne.pick_types(epochs.info, eeg=True)

        tfr = epochs.compute_tfr(
            freqs=freqs,
            n_cycles=n_cycles,
            method='morlet',
            use_fft=True,
            return_itc=False,
            picks=picks,
            decim=1,
            n_jobs=1
        )

        # shape: (n_channels, n_freqs, n_times), baseline correction applied and averaged over trials
        ersp = compute_ersp(tfr)

        # (n_epochs, n_channels, n_freqs, n_times)
        print(f"[DEBUG] TFR shape for {cond}: {tfr.data.shape}")
        print(f"[DEBUG] ERSP shape for {cond}: {ersp.shape}")

        # Create base filename 
        base_name = f"TFR_{cond.replace(' ', '_')}_ALL"

        # Save ERSP, times, and freqs separately (per condition)
        np.save(os.path.join(participant_dir, f"{base_name}-ERSP.npy"), ersp)
        np.save(os.path.join(participant_dir,
                f"{base_name}-times.npy"), tfr.times)
        np.save(os.path.join(participant_dir, f"{base_name}-freqs.npy"), freqs)

        print(f"Saved ERSP, times, freqs for {cond}")
