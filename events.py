"""Group raw EEG annotations into trials and tag each event with its condition labels"""
import mne
import sys
import csv



def create_trial_events(raw):
    """
    This function processes EEG annotations from a raw .fif file and groups them into trials.
    Each trial starts with a "Start Trial" annotation and includes all subsequent events until the next "Start Trial".
    It attaches metadata like trial number, block name, light condition, and forewarn condition to each event.
    Trials that contain only a single "Start Trial" annotation (and no follow-up events) are discarded.
    The final output is a list of trials, where each trial is a list of event dictionaries.
    """

    # Extracts events and their codes from EEG annotations
    # 'event_id' maps string labels to integer codes, like {"Start Trial": 1}
    events, event_id = mne.events_from_annotations(raw)

    # Reverse the event_id dictionary to map integer codes back to string labels
    code_to_label = {code: label for label, code in event_id.items()}

    trial_events = []      # Stores all completed trials
    current_trial = []     # Accumulates events for the current trial
    trial_num = 1          # Trial number (parsed from "Start Trial" label)
    # Current recording block name (extracted from '.fif' labels)
    current_block = ""
    light_condition = ""   # Light condition metadata (LIGHT / AMBIENT / DARK)
    forewarn_condition = ""  # Forewarning metadata (EXPECTED / UNEXPECTED)

    # Iterate through each extracted event
    for event in events:
        sample = event[0]  # The sample index where the event occurs
        code = event[2]    # The event code (integer)
        # Convert code to human-readable label
        label = code_to_label.get(code, "")

        # If label contains ".fif", it's a block/filename marker
        if ".fif" in label:
            # Extract block name by removing '.fif' extension | PID <number> LIGHT EXPECTED
            current_block = label.replace('.fif', '')

            # Check which light condition is indicated in the label
            if 'LIGHT' in label.upper():
                light_condition = 'LIGHT'
            elif 'AMBIENT' in label.upper():
                light_condition = 'AMBIENT'
            elif 'DARK' in label.upper():
                light_condition = 'DARK'

            # Check which forewarn condition is indicated
            if 'UNEXPECTED' in label.upper():
                forewarn_condition = 'UNEXPECTED'
            elif 'EXPECTED' in label.upper():
                forewarn_condition = 'EXPECTED'

        # If it's a "Start Trial" event, extract the trial number from the label
        if "Start Trial" in label:
            try:
                # Parse trial number from label format: "Start Trial: 5 at 123.45"
                trial_num = int(label.split(':')[1].split('at')[0].strip())
            except Exception:
                # Exit with the label if parsing fails
                sys.exit(label)
                
                trial_num = 0

        # Create a dictionary storing all relevant event information
        ev_dict = {
            "sample": sample,
            "label": label,
            "trial_num": trial_num,
            "current_block": current_block,
            "light_condition": light_condition,
            "forewarn_condition": forewarn_condition
        }

        # If this is a "Start Trial", it's the beginning of a new trial
        if "Start Trial" in label:
            if current_trial:
                # If the current trial has events, save it before starting a new one
                trial_events.append(current_trial)
            # Start a new trial with this "Start Trial" event
            current_trial = [ev_dict]
        else:
            # For all other events, just add them to the ongoing trial
            current_trial.append(ev_dict)

    # After the loop, if a trial is still open, save it
    if current_trial:
        trial_events.append(current_trial)

    # Remove any trial that only contains a single "Start Trial" event
    trial_events = [trial for trial in trial_events if len(trial) > 1]

    # ===  Log label presence per trial ===

    # Open (or create) a CSV file to log label presence information.
    with open("label_presence_check.csv", mode="a", newline="") as f:

        # Create a CSV writer object to write rows into the file
        writer = csv.writer(f)

        # Write the column headers in the first row of the CSV
        writer.writerow([
            "PID",
            "Trial Number",       # The trial number parsed from the "Start Trial" label
            "Has Start",          # Whether the trial contains a "Start Trial" label
            # Whether the trial contains a "Walk" label (forward walking)
            "Has Walk",
            # Whether the trial contains a "Return Walk" label (return phase)
            "Has Return Walk",
            "All Labels"          
        ])
        ppid = raw.info['subject_info'].get('pid_str')

        # Iterate over every trial in final list
        for trial in trial_events:
            # Extract just the labels (e.g., 'Start Trial: 5', 'Walk', etc.)
            labels = [e['label'] for e in trial]

            # Trial number is stored in the first event of each trial
            trial_num = trial[0]['trial_num']

            # Check whether each key label exists in this trial
            has_start = any("Start Trial" in lbl for lbl in labels)
            has_walk = any(
                "Walk" in lbl and not 'Return' in lbl for lbl in labels)
            has_return_walk = any("Return Walk" in lbl for lbl in labels)

            # Write a row for this trial with results of the checks and all labels
            writer.writerow([
                ppid,
                trial_num,
                has_start,
                has_walk,
                has_return_walk
            ])

    # Return the final list of trial event groups

    

    return trial_events
