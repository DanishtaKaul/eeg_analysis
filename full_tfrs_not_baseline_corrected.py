# -*- coding: utf-8 -*-
"""
Saves full-channel TFRs WITHOUT baseline correction (power in dB).

Outputs per condition:
  - TFR_<COND>_ALL-RAWDB.npy   (n_channels, n_freqs, n_times)
  - TFR_<COND>_ALL-times.npy   (n_times,)
  - TFR_<COND>_ALL-freqs.npy   (n_freqs,)
"""

import os
import numpy as np
import mne
from mne.time_frequency import tfr_morlet


def compute_full_channel_tfrs_no_baseline(ppid, aligned_epochs_by_condition, out_dir=r"D:\tfr_full_rawdb"):
    # same as baseline-corrected script
    freqs = np.linspace(4, 30, 30)
    n_cycles = freqs / 2.0
    participant_dir = os.path.join(out_dir, ppid)
    os.makedirs(participant_dir, exist_ok=True)

    for cond, epochs in aligned_epochs_by_condition.items():
        print(f"[{ppid} | {cond}] Computing full-channel TFR (no baseline)")

        if len(epochs) == 0:
            print(f"[WARN] {ppid} | {cond}: 0 trials — skipping")
            continue

        picks = mne.pick_types(epochs.info, eeg=True, eog=False, misc=False)

        # Per-epoch TFR (no averaging yet)
        tfr = tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            picks=picks,
            decim=1,
            average=False,
            n_jobs=1,
        )
        # tfr.data: (n_epochs, n_channels, n_freqs, n_times)
        n_ep, n_ch, n_f, n_t = tfr.data.shape
        print(
            f"[DEBUG] TFR shape for {cond}: epochs={n_ep}, ch={n_ch}, freqs={n_f}, times={n_t}")

        
        raw_db = (10.0 * np.log10(tfr.data)).mean(axis=0)
        print(f"[DEBUG] RAW dB shape for {cond}: {raw_db.shape}")
        print("[DEBUG] RAW dB range:", np.min(raw_db), "to", np.max(raw_db))

        base_name = f"TFR_{cond.replace(' ', '_')}_ALL"
        np.save(os.path.join(participant_dir,
                f"{base_name}-RAWDB.npy"), raw_db.astype(np.float32))
        np.save(os.path.join(participant_dir,
                f"{base_name}-times.npy"), tfr.times)
        np.save(os.path.join(participant_dir, f"{base_name}-freqs.npy"), freqs)

        print(
            f"[SAVED] {ppid} | {cond}: RAWDB, times, freqs - {participant_dir}")
