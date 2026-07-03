"""Compute per-channel baseline-corrected TFRs (ERSP, percent change) for Fz, Cz, POz and save them per condition"""

import os
import numpy as np
import mne
import sys


def collect_tfrs(ppid, aligned_epochs_by_condition, freq_min=4.0, freq_max=30.0, n_freqs=30):
    """
    Compute and save baseline-corrected TFRs (ERSP, percent change from baseline)
    for each condition at channels Fz, Cz, POz.

    ERSP: log10 power is baseline-corrected against the first 2 s of each trial
    (standing period), averaged across trials, then converted to percent change.
    Returns a dict of saved file paths keyed by (condition, channel).

    """
    for cond, epochs in aligned_epochs_by_condition.items():
        print(f"[{ppid} | {cond}] TFR input has {len(epochs)} epochs")

    freqs = np.linspace(freq_min, freq_max, n_freqs)
    n_cycles = freqs / 2
    channels_of_interest = ['POz', 'Cz', 'Fz']
    output_dir = os.path.join("tfr_data", ppid)
    os.makedirs(output_dir, exist_ok=True)

    saved_files = {}

    for cond, epochs in aligned_epochs_by_condition.items():
        # Check that all epochs have equal length (samples)
        epoch_lengths = [e.shape[-1] for e in epochs.get_data()]
        unique_lengths = set(epoch_lengths)
        print(f"[{ppid} | {cond}] Unique epoch lengths: {unique_lengths}")
        

        for ch in channels_of_interest:
            if ch not in epochs.info['ch_names']:
                print(f"[{ppid}] Channel {ch} not found in {cond}")
                continue

            picks = mne.pick_channels(epochs.info['ch_names'], [ch])
            print(f"[{ppid} | {cond}] Trials: {len(epochs)}, Channel: {ch}]")

            # Validate that only one channel was picked
            picked_ch_names = [epochs.info['ch_names'][i] for i in picks]
            print(
                f"[{ppid} | {cond} | {ch}] Picks: {picks} -> {picked_ch_names}", flush=True)

            if len(picks) != 1:
                print(f"[DEBUG] Picks: {picks}", flush=True)
                print(
                    f"[DEBUG] Picked channels: {picked_ch_names}", flush=True)
                
                sys.stdout.flush()
                raise ValueError(
                    f"[ERROR] Expected 1 pick for channel {ch}, got {len(picks)}: {picked_ch_names}")


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

            # Apply ERSP logic
            ersp = compute_ersp(tfr)

            # Log ERSP shape and stats
            print(
                f"[{ppid} | {cond} | {ch}] ERSP shape: {ersp.shape}, Times: {len(tfr.times)}")
            print(
                f"[{ppid} | {cond} | {ch}] ERSP min: {ersp.min():.2f}, max: {ersp.max():.2f}, mean: {ersp.mean():.2f}")
            # input("")
            max_idx = np.unravel_index(np.argmax(ersp[0]), ersp[0].shape)
            freq_with_max = freqs[max_idx[0]]
            time_with_max = tfr.times[max_idx[1]]

            print(
                f"Max ERSP at {freq_with_max:.1f} Hz, time {time_with_max:.2f} s")

            # pause for step-by-step inspection
            # input('Press Enter to continue..')

            if ersp.shape[0] == 1:
                print(f"[WARN] Only one trial for {ppid}, {cond}, {ch}")

            # Save ERSP result
            filename = f"TFR_{cond.replace(' ', '_')}_{ch}.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, ersp)  # shape: (n_channels, n_freqs, n_times)
            saved_files[(cond, ch)] = filepath
            print(f"[{ppid} | {cond}] Saved: {filename}")

    return saved_files


def compute_ersp(tfr):
    """
    Parameters:
    - tfr: MNE TimeFrequency object with shape (n_epochs, n_channels, n_freqs, n_times)

    Returns:
    - ersp_percent: np.ndarray, shape (n_channels, n_freqs, n_times), ERSP in percent
    """
    # Apply log10(Power / Baseline) per trial
    tfr.apply_baseline(mode='logratio', baseline=(0, 2.0)
                       )  # baseline from 0–2 s

    # Average log-ratio across trials, result is in log10 units
    mean_log_ratio = tfr.data.mean(axis=0)  # shape: (ch, freqs, times)


    ersp_percent = 100 * (10 ** mean_log_ratio - 1)

    # average percent change in power at a specific freq, channel and time point relative to baseline
    return ersp_percent
