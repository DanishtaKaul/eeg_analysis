"""Entry point for EEG group-level analysis

The analysis runs in stages that are switched on by uncommenting the relevant
call inside main(). Run them one at a time and in order, because each stage uses
the output of the previous one.

  1. process_multiple_experiments()  - runs the full per-participant pipeline
     (in the order defined in block_pipeline.py) and saves each participant's
     aligned epochs and per-participant TFRs.
  2. save_median_prep_durations()- saves the per-condition median preparation
     durations to CSV.
  3. tfr_setup()- averages each participant's TFRs into one
     group-average TFR per condition and channel, and saves the PREP and RESET
     plots.
  4. plot_prep_grid() - builds the multi-condition TFR grid figures
     (Fz/Cz/POz, PREP and RESET).
"""

from scripts.navigation import navigate_experiment, process_multiple_experiments
from scripts.time_frequency_plots import collect_tfrs
#from scripts.block_pipeline import process_block
import matplotlib.pyplot as plt
import os
import mne
import joblib  # for savinf crossing epochs
import pandas as pd
import numpy as np
import config
import csv
from config import (
    experiments,
    montage_path,
    debug,
    meta_info_path,
    experiment_root,
    longest_crossing_duration
)
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import re
import glob



def main():

    # process_multiple_experiments()

    # save_median_prep_durations()

    # input(
    #     f"Crossing duration average : {config.crossing_durations / config.crossing_durations_count}")

    # tfr_setup()
    # for ch in ['Fz', 'Cz', 'POz']:
    #     plot_prep_grid(ch=ch, phase='PREP')
    #     plot_prep_grid(ch=ch, phase='RESET')

    
    rows = []

    for trial_markers in config.all_obs_on_off_markers:
        for m in trial_markers:
            rows.append({
                'label': m.get('label'),
                'block': m.get('block'),
                'sample': m.get('sample'),
                'trial_num': m.get('trial_num'),  # returns None if missing
                'light': m.get('light'),
                'forewarn': m.get('forewarn'),
                'existance': m.get('existance')
            })

    df = pd.DataFrame(rows)

    out_csv = r"obstacle_on_off_markers.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved obstacle on/off markers to {out_csv}")

    input('Done')

    aligned_root = 'aligned_epochs'

    all_light_cond = ['LIGHT', 'AMBIENT', 'DARK']
    all_forewarn_cond = ['EXPECTED', 'UNEXPECTED']
    all_existance_cond = ['ABSENT', 'PRESENT']

    all_conditions = [
        f'GLOBAL {light} {forewarn} {existance}'
        for light in all_light_cond
        for forewarn in all_forewarn_cond
        for existance in all_existance_cond
    ]

    for cond in all_conditions:
        cond_underscore = cond.replace(' ', '_')
        for pid in os.listdir(aligned_root):
            current_path = os.path.join(
                aligned_root, pid, cond_underscore, f'aligned_epochs_{cond_underscore}.fif')

            if not os.path.exists(current_path):
                print(f'[SKIP] Missing: {current_path}')
                continue

            try:
                epochs = mne.read_epochs(current_path, preload=True)

                data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
                

            except Exception as e:
                print(f"[ERROR] Could not load {current_path}: {e}")

    return

freqs = np.linspace(4.0, 30.0, 30)


def tfr_setup():
    # Folder where per-participant TFR data is stored
    tfr_root = "tfr_data"

    # Output folder for group-average TFRs and plots
    out_dir = "Group_TFRs"
    os.makedirs(out_dir, exist_ok=True)

    # Channels and experimental conditions
    all_ch = ['POz', 'Cz', 'Fz']
    all_light_cond = ['LIGHT', 'AMBIENT', 'DARK']
    all_forewarn_cond = ['EXPECTED', 'UNEXPECTED']
    all_existance_cond = ['ABSENT', 'PRESENT']

    # Create all combinations of conditions as keys
    all_conditions = [
        f'GLOBAL {light} {forewarn} {exist}'
        for light in all_light_cond
        for forewarn in all_forewarn_cond
        for exist in all_existance_cond
    ]

    # Median preparation durations (in seconds) per condition
    median_prep_dur = {
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

    # Use ONLY for display text (labels/annotations), not for masks/extents
    median_prep_ms_display = {

        "GLOBAL_AMBIENT_EXPECTED_ABSENT": 2780.11,
        "GLOBAL_AMBIENT_EXPECTED_PRESENT": 2954.63,
        "GLOBAL_AMBIENT_UNEXPECTED_ABSENT": 3020.30,
        "GLOBAL_AMBIENT_UNEXPECTED_PRESENT": 3056.84,
        "GLOBAL_DARK_EXPECTED_ABSENT": 3036.67,
        "GLOBAL_DARK_EXPECTED_PRESENT": 3168.10,
        "GLOBAL_DARK_UNEXPECTED_ABSENT": 3375.26,
        "GLOBAL_DARK_UNEXPECTED_PRESENT": 3413.47,
        "GLOBAL_LIGHT_EXPECTED_ABSENT": 2762.67,
        "GLOBAL_LIGHT_EXPECTED_PRESENT": 2957.68,
        "GLOBAL_LIGHT_UNEXPECTED_ABSENT": 2855.10,
        "GLOBAL_LIGHT_UNEXPECTED_PRESENT": 3340.10,

    }

    sfreq = 500  # Sampling frequency in Hz

    for cond in all_conditions:
        for ch in all_ch:
            key = f"TFR_{cond.replace(' ', '_')}_{ch}"
            all_trials = []

            # Loop through participant folders
            for pid in os.listdir(tfr_root):
                filepath = os.path.join(tfr_root, pid, key + ".npy")

                if not os.path.exists(filepath):
                    raise FileNotFoundError(
                        f"[ERROR] Missing file: {filepath}")
                

                try:

                    data = np.load(filepath)
                    print(f"[DEBUG] {filepath} shape: {data.shape}")
                    

                    # Ensure shape is always (n_trials, n_freqs, n_times)

                    if data.ndim != 3:
                        print(filepath)
                        print(f'Data does not have 3 dimensions {data.ndim}')
                        

                    all_trials.append(data)

                except Exception as e:
                    print(f"[ERROR] Loading {filepath}: {e}")

            
            # Combine all trials across participants (with shape check)

            if not all_trials:
                raise ValueError(f"[ERROR] No trials found for {key}")
            # ---- DEBUG: Check time lengths across participants ----
            time_lengths = [trial.shape[2] for trial in all_trials]
            if len(set(time_lengths)) > 1:
                print(
                    f"[DEBUG] Unique time lengths for {key}: {set(time_lengths)}")
                #input("")
            

            # Get expected shape from first valid trial
            expected_shape = all_trials[0].shape[1:]  # (freqs, times)
            valid_trials = []

            for i, trial in enumerate(all_trials):
                if trial.shape[1:] != expected_shape:
                    raise ValueError(
                        f"[ERROR] Shape mismatch for {key}. "
                        f"Participant index {i} has shape {trial.shape}, expected (n_trials, {expected_shape})."
                    )
                
                valid_trials.append(trial)
            if len(valid_trials) == 0:
                raise ValueError(f"[ERROR] No valid trials remain for {key}")
            

            # Safe concatenation
            all_trials = np.concatenate(valid_trials, axis=0)

            # ---- Plotting ----
            

            # Compute average TFR across trials
            mean_tfr = np.mean(all_trials, axis=0)

            # Save the averaged TFR
            np.save(os.path.join(out_dir, f"{key}_mean.npy"), mean_tfr)

            # Extract frequency and time dimensions
            n_freqs, n_times = mean_tfr.shape
            freqs = np.linspace(4.0, 30.0, n_freqs)

            # Define phase durations
            baseline_sec = 2.0
            prep_sec = median_prep_dur[cond.replace(' ', '_')]
            reset_sec = 1.3
            total_sec = baseline_sec + prep_sec + reset_sec

            # time in ms
            baseline_ms = baseline_sec * 1000.0
            prep_ms = prep_sec * 1000.0
            reset_ms = reset_sec * 1000.0
            total_ms = (baseline_sec + prep_sec + reset_sec) * 1000.0

            # Create time axis in milliseconds: baseline start (0) to reset end
            times_sec = np.linspace(0, total_sec, n_times)

            times_ms = times_sec * 1000.0

            # Create masks to split PREP and RESET phases
            prep_mask = times_sec < (baseline_sec + prep_sec)

            reset_mask = times_sec >= (baseline_sec + prep_sec)

            # Slice TFR data into prep and reset segments
            prep_tfr = mean_tfr[:, prep_mask]
            reset_tfr = mean_tfr[:, reset_mask]
            prep_times = times_sec[prep_mask]
            reset_times = times_sec[reset_mask]

            prep_times_ms = times_ms[prep_mask]
            reset_times_ms = times_ms[reset_mask]

            # ---------- Plot 1: Baseline + Preparation ----------
            plt.figure(figsize=(8, 4))
            plt.imshow(prep_tfr, aspect='auto', origin='lower',
                       extent=[prep_times_ms[0], prep_times_ms[-1],
                               freqs[0], freqs[-1]],
                       cmap='Spectral_r', vmin=-50, vmax=50)
            plt.xticks([])
            
            
            # This is the new vertical dashed line at baseline end 
            plt.axvline(x=baseline_ms, color='black',
                        linestyle='dotted', label='Baseline End')

           
            # Place labels just above the x-axis, slightly below the lowest frequency
            y_pos = freqs[0] - 0.7  

            plt.text(baseline_ms / 2, y_pos, '2000 ms',
                     ha='center', va='top', fontsize=10)

            disp_ms = median_prep_ms_display[cond.replace(
                ' ', '_')]  # display-only value
            plt.text(baseline_ms + (prep_ms / 2), y_pos, f'{disp_ms} ms',
                     ha='center', va='top', fontsize=10)

            plt.title(f"{key} — Baseline + Preparation")
            plt.gca().xaxis.set_label_coords(0.5, -0.12)
            plt.xlabel("Time (ms)")

            plt.ylabel("Frequency (Hz)")
            plt.colorbar(label='Spectral Power change (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{key}_PREP.png"))
            plt.close()

            # ---------- Plot 2: Reset phase ----------
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(reset_tfr, aspect='auto', origin='lower',
                           extent=[reset_times_ms[0], reset_times_ms[-1],
                                   freqs[0], freqs[-1]],
                           cmap='Spectral_r', vmin=-80, vmax=80)
            ax.set_xticks([])

            # Place '1.3 s' centered below the x-axis using axis coordinates
            ax.annotate(f'{int(reset_ms)} ms',  # no rounding
                        xy=(0.5, -0.02), xycoords='axes fraction',
                        ha='center', va='top', fontsize=10)

            ax.set_title(f"{key} — Reset Phase")
            ax.set_xlabel("Time (ms)")
            ax.xaxis.set_label_coords(0.5, -0.12)  # move xlabel lower

            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label='Spectral Power change (%)')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{key}_RESET.png"))
            plt.close()


def plot_prep_grid(tfr_dir="Group_TFRs", out_dir="Group_TFRs/GridPlots",
                   ch='Fz', phase='PREP', vmin=-60, vmax=60):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(out_dir, exist_ok=True)

    all_light_cond = ['LIGHT', 'AMBIENT', 'DARK']
    all_forewarn_cond = ['EXPECTED', 'UNEXPECTED']
    all_existence_cond = ['ABSENT', 'PRESENT']

    all_conditions = [
        f'GLOBAL {light} {forewarn} {exist}'
        for light in all_light_cond
        for forewarn in all_forewarn_cond
        for exist in all_existence_cond
    ]

    median_prep_dur = {
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

    # Use ONLY for display text (labels/annotations), not for masks/extents
    median_prep_ms_display = {

        "GLOBAL_AMBIENT_EXPECTED_ABSENT": 2780,
        "GLOBAL_AMBIENT_EXPECTED_PRESENT": 2950,
        "GLOBAL_AMBIENT_UNEXPECTED_ABSENT": 3020,
        "GLOBAL_AMBIENT_UNEXPECTED_PRESENT": 3060,
        "GLOBAL_DARK_EXPECTED_ABSENT": 3040,
        "GLOBAL_DARK_EXPECTED_PRESENT": 3170,
        "GLOBAL_DARK_UNEXPECTED_ABSENT": 3380,
        "GLOBAL_DARK_UNEXPECTED_PRESENT": 3410,
        "GLOBAL_LIGHT_EXPECTED_ABSENT": 2760,
        "GLOBAL_LIGHT_EXPECTED_PRESENT": 2960,
        "GLOBAL_LIGHT_UNEXPECTED_ABSENT": 2860,
        "GLOBAL_LIGHT_UNEXPECTED_PRESENT": 3340,


    }

    # 3 rows × 4 cols, keep room for colorbar on the right
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    for i, cond in enumerate(all_conditions):
        key = f"TFR_{cond.replace(' ', '_')}_{ch}"
        file_path = os.path.join(tfr_dir, f"{key}_mean.npy")
        if not os.path.exists(file_path):
            print(f"[SKIP] Missing {file_path}")
            continue

        data = np.load(file_path)
        n_freqs, n_times = data.shape
        freqs = np.linspace(4.0, 30.0, n_freqs)

        baseline_sec = 2.0
        prep_sec = median_prep_dur[cond.replace(" ", "_")]
        reset_sec = 1.3
        total_sec = baseline_sec + prep_sec + reset_sec
        times_sec = np.linspace(0, total_sec, n_times)
        times_ms = times_sec * 1000.0

        baseline_ms = baseline_sec * 1000.0
        prep_ms = prep_sec * 1000.0
        reset_ms = reset_sec * 1000.0

        # Font sizes
        fs_title = 11
        fs_axis = 9
        fs_ticks = 9
        fs_annot = 7  # duration annotations

        if phase.upper() == 'PREP':
            mask = times_sec < (baseline_sec + prep_sec)
            # Show PREP as 0 … (baseline+prep), with a 2 s divider
            times = times_ms[mask]  # no shift
            x0, x1 = times[0], times[-1]
        else:  # RESET
            mask = times_sec >= (baseline_sec + prep_sec)
            # Show RESET as 0 … reset_sec
            times = (times_sec[mask] - (baseline_sec + prep_sec)) * 1000.0

            x0, x1 = times[0], times[-1]

        tfr = data[:, mask]

        ax = axes[i]
        im = ax.imshow(
            tfr, aspect='auto', origin='lower',
            extent=[x0, x1, freqs[0], freqs[-1]],
            cmap='Spectral_r', vmin=vmin, vmax=vmax
        )
        ax.set_xticks([])

        
        ax.set_title(cond.replace("GLOBAL ", "").title(), fontsize=fs_title)

        if i % 4 == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=fs_axis,  labelpad=10)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        # X-axis label only on bottom row
        if i >= 8:
            ax.set_xlabel("Time (ms)", fontsize=fs_axis)
            ax.xaxis.set_label_coords(0.5, -0.15)
        else:
            ax.set_xlabel("")

        ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
        ax.set_yticks(np.arange(4, 30, 5))
        ax.set_ylim(freqs[0], freqs[-1])

        # --- PREP---
        if phase.upper() == 'PREP':
            # Vertical dotted line at 2000 ms 
            if 0.0 <= baseline_ms <= (baseline_sec + prep_sec) * 1000.0:
                ax.axvline(baseline_ms, linestyle=':', color='k', linewidth=1)

            # Centers as fractions of the plotted PREP window 
            total_prep_window_ms = baseline_ms + prep_ms
            baseline_center_frac = (baseline_ms / 2.0) / total_prep_window_ms
            prep_center_frac = (baseline_ms + (prep_ms / 2.0)
                                ) / total_prep_window_ms

            # Show baseline in ms 
            ax.annotate(f'{int(baseline_ms)} ms',
                        xy=(baseline_center_frac, -0.02),
                        xycoords='axes fraction',
                        ha='center', va='top', fontsize=fs_annot+2)
            disp_ms = median_prep_ms_display[cond.replace(
                ' ', '_')]  # display-only value
            ax.annotate(f'{disp_ms} ms',
                        xy=(prep_center_frac, -0.02),
                        xycoords='axes fraction',
                        ha='center', va='top', fontsize=fs_annot+2)

        else:
            ax.annotate(f'{int(reset_ms)} ms',
                        xy=(0.5, -0.02),
                        xycoords='axes fraction',
                        ha='center', va='top', fontsize=fs_annot+2)
            

    # Colorbar outside on the right
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Spectral Power Change (%)", fontsize=fs_title, labelpad=10)

    fig.suptitle(f"{ch}", fontsize=14)  # fz,cz heading

    # leave room for labels + colorbar
    fig.tight_layout(rect=[0.05, 0.05, 0.90, 0.94])
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{ch}_{phase}_grid.png"), dpi=300)
    plt.close()

def save_median_prep_durations():
    # =========================================================================
    #     SAVE MEDIAN PREP DURATIONS
    # =========================================================================

    csv_filename = config.median_prep_csv_path
    fieldnames = ["label", "median"]

    # # Step 1: Load existing data 
    existing_rows = {}

    if os.path.exists(csv_filename):
        with open(csv_filename, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = row["label"]
                median = float(row["median"])
                existing_rows[label] = median

    # Step 2: Update or insert rows from  config
    for label, median in config.condition_preparation_median_times.items():
        # overwrites if label already exists
        # global median prep duration per condition across participants
        existing_rows[label] = np.median(median)

    # Step 3: Write all updated rows back to file
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for label in sorted(existing_rows):
            writer.writerow({
                "label": label,
                "median": existing_rows[label]
            })


# Lets it run directly from terminal
if __name__ == "__main__":
    main()



