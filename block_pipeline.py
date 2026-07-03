"""Run the full per-participant EEG pipeline"""

from scripts.preprocessing import filter_and_detrend_data, apply_ICA
from scripts.time_warping import time_warp_preparation
from scripts.events import create_trial_events
from scripts.epoching import identify_epochs
from scripts.autoreject import autoreject
from scripts.epochs_manual_reject import clean_epochs_manually
from scripts.manual_reject_config import participants_to_clean
from scripts.time_frequency_plots import collect_tfrs
from scripts.compute_full_tfrs import compute_full_channel_tfrs
#from scripts.compute_trialwise_ersp import compute_trialwise_ersp_for_participant
from scripts.trial_counts import summarize_counts_no_pool
from scripts.full_tfrs_not_baseline_corrected import compute_full_channel_tfrs_no_baseline
from pathlib import Path
import mne
import re
import pandas as pd
import os
import numpy as np
import joblib
import config

def _ensure_condition_col(df):
    if 'Condition' in df.columns:
        return 'Condition'
    # LightCondition + ForewarningLevel + ExistanceLevel
    L = (df['LightCondition'].astype(str).str.upper()
         if 'LightCondition' in df.columns else 'UNKNOWN')
    F = (_expectedness_series(df, 'Condition').astype(str)
         if ('ForewarningLevel' in df.columns) else 'UNKNOWN')
    X = (df['ExistanceLevel'].astype(str).str.upper()
         if 'ExistanceLevel' in df.columns else 'UNKNOWN')
    df['Condition'] = 'GLOBAL_' + L + '_' + F + '_' + \
        X  # e.g., GLOBAL_AMBIENT_EXPECTED_PRESENT
    return 'Condition'


def _expectedness_series(df, cond_col):
    # prefer a forewarning/expectation column
    cand = ['ForewarningLevel', 'ForewarnLevel', 'ForewarnCondition',
            'ForewarningCondition', 'Forewarn', 'Forewarning',
            'Expectation', 'Expectedness']
    src = next((c for c in cand if c in df.columns), None)

    def norm(x: str):
        s = str(x).strip().upper()
        if s in {'EXPECTED', 'FOREWARN', 'FOREWARNED'}:
            return 'EXPECTED'
        if s in {'UNEXPECTED', 'NOTEXPECTED', 'NOFOREWARN', 'NOT_FOREWARN', 'NOTFOREWARN', 'NO_FOREWARNING'}:
            return 'UNEXPECTED'
        if 'FOREWARN' in s:
            return 'EXPECTED'
        if 'UNEXPECT' in s or ('NOT' in s and 'FOREWARN' in s):
            return 'UNEXPECTED'
        return 'UNKNOWN'

    if src:
        return df[src].apply(norm).rename('Expectedness')

    # fallbacks 
    if 'ObstacleCondition' in df.columns:
        s = df['ObstacleCondition'].astype(str).str.upper()
    else:
        s = df[cond_col].astype(str).str.upper()
    return s.str.contains('UNEXPECTED').map({True: 'UNEXPECTED', False: None})\
            .fillna(s.str.contains('EXPECTED').map({True: 'EXPECTED'}))\
            .fillna('UNKNOWN').rename('Expectedness')


def _light_series(df, cond_col):
    """Return LIGHT / AMBIENT / DARK / UNKNOWN."""
    if 'LightCondition' in df.columns:
        return df['LightCondition'].astype(str).str.upper().rename('LIGHT')
    pat = re.compile(r'GLOBAL_(LIGHT|AMBIENT|DARK)')
    vals = (df[cond_col].astype(str).str.upper()
            .apply(lambda s: (m.group(1) if (m := pat.search(s)) else 'UNKNOWN')))
    return vals.rename('LIGHT')


def process_experiment(raw_files, trial_result_paths):
    """
    Run the full pipeline for one participant: combine metadata, annotate and
    resample raw blocks, filter/detrend, apply ICA, epoch, run AutoReject,
    time-warp the preparation phase, apply manual rejection, and save aligned
    epochs, trial counts, and TFRs.
    """

    # =============================================================================
    # Combine all trial result CSV files into one DataFrame
    # =============================================================================

    csv_list = []  # Initialize an empty list to hold individual CSV DataFrames

    for csv_path in trial_result_paths:
        meta = pd.read_csv(csv_path, index_col=None, header=0)  # Read CSV file
        csv_list.append(meta)  # Append the DataFrame to the list

    meta_combined = pd.concat(
        csv_list, axis=0, ignore_index=True)  # Concatenate all CSVs

    # =============================================================================
    # Annotate each raw file with its block name (filename) as description
    # =============================================================================

    for raw_f, trial_csv in zip(raw_files, trial_result_paths):
        # Get the full file path of the raw EEG file
        condition = raw_f.filenames[0]
        # Extract the filename from the full path (e.g., 'PID 5 LIGHT EXPECTED')
        filename = re.search(r'[^\\/]+$', condition).group()
        # Annotate the entire duration of the raw file with the filename
        raw_f.annotations.append(
            onset=0,                    # start of file
            duration=raw_f.times[-1],   # entire duration
            description=filename        # label
        )

    # =============================================================================
    # Resample raw files to a uniform sampling frequency (500 Hz)
    # =============================================================================

    resample_log_path = "resampled_log.txt"  # Output log file
    target_sfreq = 500.0                     # Desired frequency in Hz

    for raw_f in raw_files:
        if raw_f.info['sfreq'] != target_sfreq:

            original_sfreq = raw_f.info['sfreq']
            file_name = getattr(raw_f, 'filenames', ['(unknown)'])[0]

            # Up and down resampling to remove rounding errors
            raw_f.resample(sfreq=500.01,
                           npad="auto",
                           method="polyphase",
                           window="auto",
                           pad="auto")
            raw_f.resample(sfreq=500.0,
                           npad="auto",
                           method="polyphase",
                           window="auto",
                           pad="auto")

            with open(resample_log_path, 'a') as log_file:
                log_file.write(
                    f"Resampled file ({file_name}): FROM: {original_sfreq} TO: {raw_f.info['sfreq']}\n")

    # =============================================================================
    # Concatenate raw EEG files
    # =============================================================================

    all_raw_files = []  # Initialize list to hold processed raw files

    for raw_f in raw_files:
        all_raw_files.append(raw_f)

    # Concatenate all processed raw files into one continuous raw object
    experiment_raw = mne.concatenate_raws(
        all_raw_files, on_mismatch='warn')


# =============================================================================
#     Check units
# =============================================================================

    # === Check EEG channel units ===
    original_units = [ch['unit'] for ch in experiment_raw.info['chs']]
    print(f'Original units: {original_units}')

    # Get the participant ID
    ppid = experiment_raw.info['subject_info'].get('pid_str', 'unknown_pid')
    eeg_data = experiment_raw.get_data(picks='eeg')

    print(f'processing {ppid}')

    print("EEG signal range (V):", np.min(eeg_data), "to", np.max(eeg_data))
    for ch_idx, ch_name in enumerate(experiment_raw.ch_names):
        ch_data = eeg_data[ch_idx]
        ch_min = ch_data.min()
        ch_max = ch_data.max()
        if np.abs(ch_min) > 1e-3 or np.abs(ch_max) > 1e-3:
            print(f"{ch_name}: min={ch_min:.6f} V, max={ch_max:.6f} V")

    # Apply filtering, detrending, and rereferencing
    experiment_raw = filter_and_detrend_data(experiment_raw)

# =============================================================================
#     Apply ICA | Bandpassing 1-40Hz
# =============================================================================
    print("apply ICA")
    # raw plot before ICA
    print("BEFORE ICA")
    # experiment_raw.plot(n_channels=30, title=f"Raw Data - {ppid} Before ICA")

    # Construct the ICA-cleaned file path safely
    ica_dir = Path("ICA_cleaned_raws") / ppid
    ica_file = ica_dir / f"{ppid}_ica_cleaned_raw.fif"

    # Ensure the directory exists (create it if saving later)
    ica_dir.mkdir(parents=True, exist_ok=True)

    # Mark fp1 as bad for pid 40
    if ppid == 'PID 40':
        experiment_raw.info['bads'].append('FP1')
        print(f"[{ppid}] Marked FP1 as bad.")

    # Try to load the ICA-cleaned raw file
    if ica_file.exists():
        try:
            clean_ica_raw = mne.io.read_raw_fif(
                ica_file, preload=True, on_split_missing='warn')
            print(f"Loaded ICA-cleaned file for {ppid} from: {ica_file}")
        except Exception as e:
            print(f"Error reading ICA-cleaned file: {e}")
            # apply ica if file cannot be loaded
            clean_ica_raw = apply_ICA(experiment_raw)
    else:
        print(f"No ICA-cleaned file found at: {ica_file}")
        # apply ica if clean file not found
        clean_ica_raw = apply_ICA(experiment_raw)

    clean_ica_raw.set_annotations(experiment_raw.annotations)
    clean_ica_raw.info['subject_info'] = {
        'pid_str': ppid,
    }
    # Use clean_ica_raw in further steps
    experiment_raw = clean_ica_raw

    # raw plot after ICA
    print("AFTER ICA")
    # experiment_raw.plot(n_channels=30, title=f"Raw Data - {ppid} After ICA")

    # Final band-pass for analysis
    experiment_raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

    # Check EEG signal after ICA
    eeg_data_post_ica = experiment_raw.get_data(picks='eeg')
    print(f"[{ppid}] EEG signal range after ICA (V):",
          np.min(eeg_data_post_ica), "to", np.max(eeg_data_post_ica))

    # Per channel min/max check
    for ch_idx, ch_name in enumerate(experiment_raw.ch_names):
        ch_data = eeg_data_post_ica[ch_idx]
        ch_min = ch_data.min()
        ch_max = ch_data.max()
        if np.abs(ch_min) > 1e-3 or np.abs(ch_max) > 1e-3:
            print(
                f"[{ppid}] AFTER ICA: {ch_name} = min {ch_min:.6f} V, max {ch_max:.6f} V")

    
    print("trial events")
    trial_events = create_trial_events(experiment_raw)

    print("identify epochs")
    crossing_epochs = identify_epochs(
        experiment_raw, trial_events, meta_combined)
    obs_on_off_markers = []
    for epoch in crossing_epochs:
        current_existance = epoch.get('existance_condition')

        # Now loop through the trials inside that epoch
        for e in epoch['trial']:
            obs_on_off_markers.append({
                'label': e['label'],
                'block': e['current_block'],
                'sample': e['sample'],
                'trial_num': e['trial_num'],
                'light': e['light_condition'],
                'forewarn': e['forewarn_condition'],
                'existance': current_existance  
            })

    config.all_obs_on_off_markers.append(obs_on_off_markers)

    #### count trials####
    # ---- PRE-AR report (attempted epochs) ----

    # Align metadata rows to the epochs actually attempted (skip SKIPPED)
    meta_indices = [ep['meta_index'] for ep in crossing_epochs
                    if not ep.get('SKIPPED', False)]
    trial_metadata_aligned = meta_combined.iloc[meta_indices].reset_index(
        drop=True)

    # Choose/construct a full-condition column consistent with  keys
    COND_COL = _ensure_condition_col(trial_metadata_aligned)

    # Full-condition counts BEFORE AutoReject
    counts_pre = trial_metadata_aligned[COND_COL].value_counts().sort_index()
    print("\n[Counts BEFORE AutoReject] (attempted epochs)")
    for cond, n in counts_pre.items():
        print(f"  {cond:<40} {n:>3}")

    # Expectedness & LIGHT breakdown BEFORE AutoReject
    pre_tmp = trial_metadata_aligned.copy()
    pre_tmp['Expectedness'] = _expectedness_series(pre_tmp, COND_COL)
    pre_tmp['LIGHT'] = _light_series(pre_tmp, COND_COL)

    print("\n[BEFORE AR] Expected vs Unexpected (overall)")
    overall_pre = (pre_tmp['Expectedness'].value_counts()
                   .reindex(['EXPECTED', 'UNEXPECTED', 'UNKNOWN'])
                   .fillna(0).astype(int))
    for k, v in overall_pre.items():
        print(f"  {k:<11} {v:>3}")

    print("\n[BEFORE AR] Expected vs Unexpected by LIGHT")
    by_light_pre = (pre_tmp.groupby(['LIGHT', 'Expectedness']).size()
                    .unstack('Expectedness', fill_value=0)
                    .reindex(columns=['EXPECTED', 'UNEXPECTED', 'UNKNOWN'], fill_value=0)
                    .sort_index())
    for light, row in by_light_pre.iterrows():
        exp = int(row.get('EXPECTED', 0))
        unx = int(row.get('UNEXPECTED', 0))
        unk = int(row.get('UNKNOWN', 0))
        line = f"  {light:<7}  EXPECTED={exp:>3}  UNEXPECTED={unx:>3}"
        if unk:
            line += f"  UNKNOWN={unk:>3}"
        print(line)


    # # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
    # # Load or Run AutoReject to Clean Epochs
    # # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =

    # Define the folder where AutoReject outputs are stored
    ar_base = Path("autoreject")

    # Define the file path to the cleaned epochs (.fif file)
    ar_dir = ar_base / ppid
    fif_path = ar_dir / "epochs_clean.fif"
    csv_path = ar_dir / f"kept_indices_{ppid}.csv"

    # Check whether both files exist(i.e., AutoReject was already run)

    if fif_path.exists() and csv_path.exists():
        print(f"Found AutoReject-cleaned files for {ppid}, loading them...")

        # Load the cleaned epochs from disk (with preload to load into memory immediately)
        epochs_clean = mne.read_epochs(fif_path, preload=True)

        # print(f"[{ppid}] Plotting PSD after AutoReject (loaded epochs)...")
        # epochs_clean.plot_psd(fmax=40)

        # Load the kept indices 
        df = pd.read_csv(csv_path)
        # Extract the kept indices as a list of integers (dropping any missing values)
        kept_indicies = df['Kept Epoch Index'].dropna().astype(int).tolist()

        # Safety check: ensure the expected column is present in the file
        if 'Kept Epoch Index' not in df.columns:
            raise ValueError(
                f"'Kept Epoch Index' column not found in {csv_path}")
        # Plot epochs clean
        # epochs_clean.plot_psd(fmax=40)

        # epochs_clean.plot(n_channels=32,
            # title='Cleaned Epochs After AutoReject')

    else:
        # If files don't exist, run AutoReject from scratch
        print(f"fif_path: {fif_path}")
        print(f"csv_path: {csv_path}")
        print(f"No AutoReject files found for {ppid}, running AutoReject...")

        # Align metadata rows to crossing epochs
        meta_indices = [ep['meta_index']
                        for ep in crossing_epochs if not ep.get('SKIPPED', False)]

        print("All LightCondition values in meta_combined:")
        print(meta_combined['LightCondition'].value_counts())

        trial_metadata_aligned = meta_combined.iloc[meta_indices].reset_index(
            drop=True)

        print("LightCondition values in aligned metadata:")
        print(trial_metadata_aligned['LightCondition'].value_counts())

        # Run AutoReject to clean the EEG epochs and get retained trial indices
        epochs_clean, kept_indicies = autoreject(
            experiment_raw, crossing_epochs)

    print(f"[{ppid}] Checking EEG amplitude after AutoReject")

    # shape = (n_epochs, n_channels, n_times)
    eeg_data_ar = epochs_clean.get_data(picks='eeg')
    eeg_data_flat = np.abs(eeg_data_ar).max(
        axis=(0, 2))  # max abs per channel
   # (check if bigger than 300 microvolt)
    for ch_idx, ch_name in enumerate(epochs_clean.ch_names):
        ch_max = eeg_data_flat[ch_idx]
        if ch_max > 3e-4:
            print(
                f"[{ppid}] AFTER AUTOREJECT: High amplitude in {ch_name} = {ch_max:.6f} V")

    print(f"[{ppid}] Global EEG amplitude range after AutoReject: {eeg_data_ar.min():.6f} to {eeg_data_ar.max():.6f} V")
    # print(f"[{ppid}] Global EEG amplitude range after AutoReject: {eeg_data_ar.min()*1e6:.2f} to {eeg_data_ar.max()*1e6:.2f} µV")

    # === Compute signal quality metrics per channel (after AutoReject) ===
    print(f"[{ppid}] Calculating mean amplitude and std dev per channel...")

    # Extract EEG data only
    # shape: (n_epochs, n_channels, n_times)
    eeg_data = epochs_clean.get_data(picks='eeg')
    ch_names = epochs_clean.copy().pick('eeg').ch_names  # ensure only EEG channels

    # Compute mean absolute amplitude and standard deviation
    mean_amp = np.mean(np.abs(eeg_data), axis=(0, 2)) * \
        1e6  # convert to µV for csv
    std_amp = np.std(eeg_data, axis=(0, 2)) * 1e6          # µV

    # Create a DataFrame
    metrics_df = pd.DataFrame({
        'Channel': ch_names,
        'Mean_Amplitude_µV': mean_amp,
        'STD_Amplitude_µV': std_amp
    })

    # # Save the CSV
    signal_metrics_path = ar_dir / f"signal_quality_metrics_{ppid}.csv"
    metrics_df.to_csv(signal_metrics_path, index=False)
    print(f"[{ppid}] Saved signal quality metrics to: {signal_metrics_path}")

    ## Time warp epochs to global median prep time
    aligned_epochs = time_warp_preparation(
        experiment_raw, crossing_epochs, kept_indicies)

    # # plot_title = f"Cleaned Epochs After Manual Rejection — {ppid}"
    # epochs_clean.plot(scalings='auto', title=plot_title)

    # =============================================================================
    #     Save aligned epochs & crossing epochs
    # =============================================================================

    # # Directory name
    dir_name = r"D:\aligned_epochs"

    # Base directory
    base_dir = os.path.join(dir_name, f"{ppid}")
    os.makedirs(base_dir, exist_ok=True)

  # Save crossing_epochs once per participant
    crossing_path = os.path.join(base_dir, f"crossing_epochs_{ppid}.pkl")
    with open(crossing_path, "wb") as f:
        joblib.dump(crossing_epochs, f)

    # Save each condition's aligned epochs in its own subfolder
    for cond, epochs in aligned_epochs.items():

        # Clean folder/file names (remove spaces, special chars)
        cond_safe = cond.replace(" ", "_").replace("/", "_")
        cond_dir = os.path.join(base_dir, cond_safe)
        os.makedirs(cond_dir, exist_ok=True)

        filepath = os.path.join(cond_dir, f"aligned_epochs_{cond_safe}.fif")
        filepath = filepath.strip().replace("\n", "").replace("\r", "")

        print(f"[{ppid}] Saving: {filepath}")
        epochs.save(filepath, overwrite=True)

    print(f"[{ppid}] All aligned epochs and crossing data saved.")

    # ============================
    # Manual Rejection
    # ============================
    if ppid in participants_to_clean:
        print(f"[{ppid}] Running manual epoch rejection now...")
        aligned_epochs = clean_epochs_manually(
            ppid, participants_to_clean[ppid])
    else:
        print(f"[{ppid}] No manual epoch drops specified.")

    for cond, epochs in aligned_epochs.items():
        print(
            f"[{ppid}] Final count for {cond}: {len(epochs)} epochs after manual cleaning")

    """
    Gets a final report on trial numbers that are within aligned_epochs
    """

    final_report = []

    for _, epochs_obj in aligned_epochs.items():
        meta = epochs_obj.metadata

        for _, row in meta.iterrows():
            t_num = row.get('trial_num', None)
            block = row['trial'][0]['current_block'].strip().lower()

            match = meta_combined.loc[
                (meta_combined['ppid'].str.strip().str.lower() == block) &
                (meta_combined['trial_num'] == t_num)
            ]

            m = match.iloc[0]
            light = m['LightCondition'].strip().upper()
            existance = m['ExistanceLevel'].strip().upper()
            forewarn_raw = m['ForewarningLevel'].strip().upper()
            forewarn = 'UNEXPECTED' if forewarn_raw == 'NOTEXPECTED' else forewarn_raw
            cond = f"GLOBAL_{light}_{forewarn}_{existance}"

            final_report.append({
                'condition': cond,
                'trial_num': t_num,
            })

    # Convert to a DataFrame
    df_final = pd.DataFrame(final_report)

    # Define the output directory and ensure it exists
    out_dir = Path("./final_trial_number")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV using the participant ID (ppid)
    csv_filename = out_dir / f"{ppid}.csv"
    df_final.to_csv(csv_filename, index=False)

    print(
        f"Successfully saved {len(df_final)} retained trials to {csv_filename}")

    # === Save final counts that feed into collect_tfrs ===
    summarize_counts_no_pool(ppid, aligned_epochs,
                             out_root=Path(r"D:\final_trial_counts"))

    # ====================
    # collect tfrs
    # ===================
    # Collect the tfr plot for this participant
    collect_tfrs(ppid, aligned_epochs)

    # ======================================
    # collect full channel tfrs 
    # ======================================
    compute_full_channel_tfrs(ppid, aligned_epochs)

    # ===========================================
    # full-channel, NO baseline correction (power in dB)
    # ===================================================
    compute_full_channel_tfrs_no_baseline(
        ppid, aligned_epochs, out_dir=r"D:\tfr_full_rawdb")
