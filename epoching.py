"""Segment continuous EEG into per-trial epochs around obstacle crossings,
extracting baseline, preparation, and reset phases and skipping invalid trials"""

import pandas as pd
import numpy as np
from config import pre_crossing_sec, post_crossing_sec, debug, preparation_buffer, longest_crossing_duration  
import os  
import config
from collections import defaultdict



"""
# cs_offset and ce_offset are sample positions *within the current EEG epoch* (the extracted segment):
#   - cs_offset = number of samples from the start of the epoch to crossing start
#   - ce_offset = number of samples from the start of the epoch to crossing end
"""


def trial_is_absent(meta, trial_num, current_block):
    # Clean the current block name by stripping spaces and converting to lowercase
    current_block_clean = current_block.strip().lower()

    # Check if the cleaned block name exists in the metadata ('ppid' column)
    if not (meta['ppid'].str.strip().str.lower() == current_block_clean).any():
        # If no match found, pause and print info for debugging
        input(
            f"Trial cannot be found in meta file:\n"
            f"Current block: {current_block_clean}\n"
            f"Meta PPID Column (unique):\n{meta['ppid'].str.strip().unique()}"
        )

    # Select the row in metadata where both block name and trial number match
    row = meta.loc[
        (meta['ppid'].str.strip().str.lower() == current_block_clean) &
        (meta['trial_num'] == trial_num)
    ]

    # Return True if a matching row exists and the ExistenceLevel is marked "absent"
    return not row.empty and row.iloc[0]['ExistanceLevel'].strip().lower() == 'absent'


def identify_epochs(raw, trialEvents, meta_combined):
    sfreq = raw.info['sfreq']  # Sampling frequency of EEG data
    ppid = raw.info['subject_info'].get(
        'pid_str', 'unknown_pid')  # Participant ID
    crossing_epochs = []  # Store valid and skipped trial info here
    event_list = []  # For MNE-style event marking 

    for trial in trialEvents:
        SKIPPED = False  # Whether this trial should be excluded
        REASON = "OK"  # Reason for skipping

        trial_num = trial[0]['trial_num']  # Get trial number from first event
        if trial_num == 1:
            SKIPPED = True
            REASON = "Practice Trial"  # Skip practice trial
            # Skip remaining logic for this trial

        # Initialize key timestamps
        start_trial_time = None
        prep_start_sample = None
        cs_sample, ce_sample = None, None  # Crossing start and end 

        # Anchor condition labels to the first event in the trial.
        # These must NOT be inside the event loop the last event
        # can carry the next block's labels at block boundaries.
        current_block = trial[0]['current_block']
        light_condition = trial[0]['light_condition']
        forewarn_condition = trial[0]['forewarn_condition']

        # Loop through all events in the trial
        for event in trial:
            # Get event label (e.g., 'Toe Bounds start')
            label = event['label']
            event_sample = event['sample']
            # Check if this is a walk or return walk marker - marks start of preparation
            if ("Walk" in label or "Return Walk" in label) and prep_start_sample is None:
                prep_start_sample = event_sample

            # Determine whether this trial has an obstacle ('PRESENT') or not ('ABSENT')
            is_absent = trial_is_absent(
                meta_combined, trial_num, current_block)

            # === Identify crossing start and end ===
            if is_absent:
                # If obstacle absent - look for "Foot Mid start" and "Foot Mid end"
                if "Foot" in label and "Mid" in label and "start" in label and cs_sample is None:
                    cs_sample = event_sample
                elif "Foot" in label and "Mid" in label and "end" in label and ce_sample is None:
                    ce_sample = event_sample
            else:
                # If obstacle present - look for "Foot Bounds start" and "Foot Bounds end"
                if "Foot" in label and "Bounds" in label and "start" in label and cs_sample is None:
                    cs_sample = event_sample
                elif "Foot" in label and "Bounds" in label and "end" in label and ce_sample is None:
                    ce_sample = event_sample

        # === Fallbacks: if crossing start/end still not found ===

        if cs_sample is None:
            for e in trial:
                if "Foot" in e['label'] and "Crossing" in e['label'] and cs_sample is None and (
                    ("Mid" in e['label'] if trial_is_absent(
                        meta_combined, trial_num, e['current_block']) else "Bounds" in e['label'])
                ):
                    cs_sample = e['sample']
                    break

        if ce_sample is None:
            for e in trial:
                # Choose "Foot" + "end" label that matches condition (Mid/Bounds)
                if "Foot" in e['label'] and "Crossing" in e['label'] and ce_sample is None and (
                    ("Mid" in e['label'] if trial_is_absent(
                        meta_combined, trial_num, e['current_block']) else "Bounds" in e['label'])
                ):
                    ce_sample = e['sample']
                    break

        # === Skip this trial if key times are still missing ===
        if cs_sample is None or ce_sample is None or prep_start_sample is None:
            if cs_sample is None:
                cs_sample = 0
            if ce_sample is None:
                ce_sample = 0
            if prep_start_sample is None:
                prep_start_sample = 0
            SKIPPED = True
            REASON = "Missing critical event marker"

        if prep_start_sample == cs_sample:
            SKIPPED = True
            REASON = "prep time and cs time are equal"

        prep_duration = (cs_sample - prep_start_sample)/sfreq
        if prep_duration > config.pre_crossing_sec:
            SKIPPED = True
            REASON = f"Preparation duration ({prep_duration:.2f}) is longer than max prep time ({config.pre_crossing_sec})"

        # === prep must occur before crossing ends, walk label shows up after crossing end ===
        if prep_start_sample is not None and cs_sample is not None:
            if prep_start_sample >= cs_sample:
                SKIPPED = True
                REASON = "Preparation time occurs after crossing end"

        # === Skip long crossings ===
        crossing_duration = (ce_sample - cs_sample) / \
            sfreq if (cs_sample and ce_sample) else 0
        if crossing_duration > longest_crossing_duration:
            SKIPPED = True
            REASON = f"Crossing duration too long: {crossing_duration:.2f}s"

        # === Calculate reset duration (from end of this trial to next trial's 'Start Trial') ===
        reset_duration = None
        current_block = trial[0]['current_block']  # Get current trial's block

        for next_trial in trialEvents[trialEvents.index(trial) + 1:]:
            # Skip if next trial is in a different block
            if next_trial[0]['current_block'] != current_block:
                continue

            for e in next_trial:
                if "Start Trial" in e['label']:
                    reset_duration = (e['sample']-ce_sample) / sfreq
                    break
            if reset_duration is not None:
                break

        # Flag trial for skipping if reset is too short
        if reset_duration is not None and reset_duration < post_crossing_sec:
            SKIPPED = True
            REASON = f"Reset phase too short: {reset_duration:.2f}s"

        # === Otherwise: extract EEG data for full epoch ===
        pre_crossing_sample = int(pre_crossing_sec*sfreq)
        baseline_duration_sample = int(config.baseline_duration_sec*sfreq)
        longest_crossing_duration_sample = int(longest_crossing_duration*sfreq)
        post_crossing_sample = int(post_crossing_sec*sfreq)

        epoch_start_sample = cs_sample - pre_crossing_sample - baseline_duration_sample
        epoch_end_sample = epoch_start_sample + baseline_duration_sample + \
            pre_crossing_sample + longest_crossing_duration_sample + post_crossing_sample

        if raw.get_data().shape[1] < epoch_end_sample:
            SKIPPED = True
            REASON = f"Not enough data for epoch"
        # input(epoch_start_sample)

        # Extract the full trial EEG segment
        data, _ = raw[:, epoch_start_sample:epoch_end_sample]

        if data.shape[1] <= 0 and cs_sample is not 0:
            # input(f'cs_sample: {cs_sample}')
            SKIPPED = True
            REASON = f"Data is empty/epoch_start_sample : {epoch_start_sample}/epoch_end_sample : {epoch_end_sample}"

        # === If trial was marked as skipped, save metadata and continue to next trial ===
        if SKIPPED:
            crossing_epochs.append({
                'ppid': ppid,
                'SKIPPED': True,
                'REASON': REASON,
                'trial': trial,
                'trial_num': trial_num
            })
            continue

        # Compute sample offsets for each phase (prep, crossing, reset)

        # Compute how many samples after the start of the epoch the preparation phase starts
        # This gives the offset within the current epoch array 
        prep_start_sample_offset = prep_start_sample - epoch_start_sample

        # Compute the offset of crossing start *relative to the epoch start*
        cs_offset = cs_sample - epoch_start_sample

        # Compute the offset of crossing end relative to the epoch start
        ce_offset = ce_sample - epoch_start_sample

        # Calculate the sample index in raw EEG where the reset phase ends:
        # crossing end + post_crossing duration (usually 1.3s)
        reset_end_sample = ce_sample + post_crossing_sample

        # Compute offset of reset phase end relative to the epoch start
        reset_end_offset = reset_end_sample - epoch_start_sample

        # Extract prep and reset EEG segments from the epoch (already sliced from raw data)

        # Slice from preparation start up to crossing start - this is the preparation phase EEG
        prep_data = data[:, prep_start_sample_offset:cs_offset]

        # Slice from crossing end to reset end - this is the reset phase EEG
        reset_data = data[:, ce_offset:reset_end_offset]

        # -----------------------------------------------
        # Extract baseline: 2 seconds before prep_start_sample
        # -----------------------------------------------

        baseline_start_sample = prep_start_sample - baseline_duration_sample
        baseline_end_sample = prep_start_sample

        # Handle edge cases where baseline would be before recording start
        if baseline_start_sample < 0:
            SKIPPED = True
            REASON = f"Baseline starts before recording"
            crossing_epochs.append({
                'ppid': ppid,
                'SKIPPED': True,
                'REASON': REASON,
                'trial': trial,
                'trial_num': trial_num
            })

        # Slice raw EEG for baseline segment
        baseline_data, _ = raw[
            :, baseline_start_sample:baseline_end_sample]

        # concatenate baseline, prep and reset for autoreject
        combined_data = np.concatenate(
            [baseline_data, prep_data, reset_data], axis=1)

        # this will be stored per epoch so they can be split after autoreject
        baseline_len = baseline_data.shape[1]
        prep_len = prep_data.shape[1]
        reset_len = reset_data.shape[1]

        # Create condition key for computing preparation medians later
        key = f"GLOBAL {light_condition.strip().upper()} {forewarn_condition.strip().upper()} {'ABSENT' if is_absent else 'PRESENT'}"

        # === Save this trial’s full metadata ===
        crossing_epochs.append({
            'ppid': ppid,
            'data': data,
            'prep_data': prep_data,
            'reset_data': reset_data,
            'combined_data': combined_data,
            'prep_len': prep_len,
            'reset_len': reset_len,
            'baseline_data': baseline_data,
            'baseline_len': baseline_len,
            'SKIPPED': False,
            'REASON': REASON,
            'trial': trial,
            'trial_num': trial_num,
            'light_condition': light_condition.strip().upper(),
            'forewarn_condition': forewarn_condition.strip().upper(),
            'existance_condition': 'ABSENT' if is_absent else 'PRESENT',
            'median_key': key,
            'epoch_start_sample': epoch_start_sample,
            'epoch_end_sample': epoch_end_sample,
            'prep_start_sample': prep_start_sample,
            'prep_start_sample_offset': prep_start_sample_offset,
            'crossing_start_sample': cs_sample,
            'crossing_start_sample_offset': cs_offset,
            'crossing_end_sample': ce_sample,
            'crossing_end_sample_offset': ce_offset,
            'reset_start_sample': ce_sample,
            'reset_start_sample_offset': ce_offset,
            'reset_end_sample': reset_end_sample,
            'reset_end_sample_offset': reset_end_offset,
            'meta_index': meta_combined.index[
                (meta_combined['trial_num'] == trial_num) &
                (meta_combined['ppid'].str.strip().str.lower()
                 == current_block.strip().lower())
            ].tolist()[0]
        })

        # Add this trial to event list for MNE
        event_list.append([cs_sample, 0, 1])

    # === Compute condition-wise average preparation durations ===
    cond_prep_times = defaultdict(list)
    for e in crossing_epochs:
        if not e['SKIPPED']:
            key = e['median_key']
            dur = (e['crossing_start_sample'] - e['prep_start_sample']) / sfreq
            cond_prep_times[key].append(dur)

    # Save average duration into config dict
    for cond in cond_prep_times:
        mean_duration = np.mean(cond_prep_times[cond])
        config.condition_preparation_median_times[cond].append(
            mean_duration)  # this is actually means, means of preparation durations per participant per condition.

    
# =============================================================================
#     FINAL: Save skipped trials after all logic (including reset checks)
# =============================================================================

    csv_path = "skipped_trials.csv"
    skipped_epochs = []

    # Collect skipped trials from the current participant
    for e in crossing_epochs:
        if e.get('SKIPPED') == True:
            e_copy = e.copy()
            e_copy.pop('data', None)  # Remove EEG to keep file small
            e_copy.pop('prep_data', None)
            e_copy.pop('reset_data', None)
            e_copy.pop('combined_data', None)
            e_copy.pop('baseline_data', None)
            skipped_epochs.append(e_copy)

    new_skipped = pd.DataFrame(skipped_epochs)

    # Append to the master CSV (once per participant)
    new_skipped.to_csv(csv_path, mode='a', index=False,
                       header=not os.path.exists(csv_path))

    common_length = 0
    for i, e in enumerate(crossing_epochs):
        if e['SKIPPED']:
            continue

        if common_length == 0:
            common_length = e['data'].shape[1]

        if e['data'].shape[1] != common_length:
            print(
                'Shape of crossing epochs are different, printing problematic epoch... \n')
            print(e)
            print(
                f"Epoch end exceeds raw duration by: {(epoch_end_sample - raw.get_data().shape[1])/500}s")
            # input(
            # f"\n common_length : {common_length} | epoch_length : {e['data'].shape[1]} | raw_shape : {raw.get_data().shape}")

    return crossing_epochs
