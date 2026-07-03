
# -*- coding: utf-8 -*-
"""
Build the set of EEG-and-gait matched trials used for the gait analysis
"""

import os
import pandas as pd
import re
from collections import Counter
from helper_functions import (
    extract_light,
    extract_forewarn,
    get_existence_from_timeseries
)


TIME_SERIES_DATA_DIR = r'D:\Gait_Analysis\time_series_data_x_y_added'


def get_all_trials():
    all_trials = []

    # Get all the trial data
    for pid in os.listdir(TIME_SERIES_DATA_DIR):
        pid_dir = os.path.join(TIME_SERIES_DATA_DIR, pid)
        for ts_file in os.listdir(pid_dir):

            
            if "_T1_" in ts_file:
                continue

            ts_file_path = os.path.join(pid_dir, ts_file)
            all_trials.append(ts_file_path)
    return all_trials


def get_valid_eeg_trials():
    # Will contain all paths to all motion tracking files
    all_trials = get_all_trials()

    all_trials_suspicious_removed = remove_suspicious_trials(all_trials)
    valid_eeg_trials = remove_non_eeg_trials(all_trials_suspicious_removed)

    return valid_eeg_trials


def remove_suspicious_trials(all_trials):
    print('remove_suspicious_trials')
    sus_trials_csv_path = r"D:\Gait_Analysis\time_series_suspicious_trials.csv"

    # Turn csv into dataframe
    df = pd.read_csv(
        sus_trials_csv_path,
    )

    for filename in df['filename']:
        for i, _ in enumerate(all_trials):
            if filename in all_trials[i]:
                # print(f'{filename} found in {all_trials[i]}')
                # Remove the trial
                all_trials.remove(all_trials[i])

    return all_trials


def remove_non_eeg_trials(all_trials):
    print('remove_non_eeg_trials')
    print(f'\t all_trials length: {len(all_trials)}')
    EEG_FINAL_TRIAL_NUMBER_DIR = r'D:\Gait_Analysis\eeg_final_trial_number'
    valid_eeg_trials = []

    for pid_file in os.listdir(EEG_FINAL_TRIAL_NUMBER_DIR):
        file_path = os.path.join(EEG_FINAL_TRIAL_NUMBER_DIR, pid_file)
        df = pd.read_csv(file_path)

        pid = pid_file.replace('.csv', '').upper()

        for row in df.itertuples():
            condition = row.condition
            trial_num = row.trial_num
            light = extract_light(condition)
            forewarn = extract_forewarn(condition)
            key = f"{pid} {light} {forewarn}_T{trial_num}_"

            matched = [trial for trial in all_trials if key in trial]

            if len(matched) > 1:
                print(f'WARNING: {len(matched)} matches for key: {key}')
                for m in matched:
                    print(f'\t {m}')

            if len(matched) == 0:
                continue
            else:
                valid_eeg_trials.extend(matched)

    print(f'\t valid_eeg_trials length: {len(valid_eeg_trials)}')

    return valid_eeg_trials


def create_retained_trials_csv_with_obstacle(
    all_trials,
    output_path='eeg final analysis trials with obstacle.csv'
):
    print('create_retained_trials_csv_with_obstacle')

    # Use a Counter to track (pid, light, obstacle) combinations
    stats_counter = Counter()

    for trial in all_trials:
        # Extract PID
        pid_match = re.search(r'\bPID \d+', trial)
        if not pid_match:
            continue
        pid = pid_match.group(0)

        light = extract_light(trial)
        forewarn = str(extract_forewarn(trial)).lower()
        existence = get_existence_from_timeseries(trial)
        obstacle = f"{forewarn}_{existence}"

        stats_counter[(pid, light, obstacle)] += 1

    # Convert the Counter dictionary into a list of dictionaries for Pandas
    rows = []
    for (pid, light, obstacle), count in stats_counter.items():
        rows.append({
            'pid': pid,
            'light': light,
            'obstacle': obstacle,
            'num_trials': count
        })

    # Create DataFrame and Export
    df = pd.DataFrame(rows)

    df = df.sort_values(by=['pid', 'light', 'obstacle'])

    df.to_csv(output_path, index=False)
    print(f'\t CSV created with {len(df)} participant-condition rows.')

    return df


if __name__ == "__main__":
    valid_trials = get_valid_eeg_trials()
    create_retained_trials_csv_with_obstacle(valid_trials)
    print("Done :)")
