"""Estimate each participant's leg length from the motion-tracking data"""

import os
import glob
import numpy as np
import pandas as pd

MT_PATH = r"D:\motion_tracking_data"

VALID_PIDS = {
    "PID 3", "PID 4", "PID 5", "PID 6", "PID 7", "PID 8", "PID 9",
    "PID 10", "PID 11", "PID 14", "PID 15", "PID 18", "PID 19",
    "PID 20", "PID 23", "PID 26", "PID 30", "PID 31", "PID 35",
    "PID 50", "PID 57", "PID 58",
    "PID 13", "PID 16", "PID 17", "PID 22", "PID 24", "PID 25", "PID 27",
    "PID 28", "PID 29", "PID 32", "PID 33", "PID 36", "PID 38", "PID 40",
    "PID 41", "PID 43", "PID 45", "PID 46", "PID 49",
    "PID 52", "PID 55"
}

KEEP_MARKERS = [":Skeleton 001", "LThigh",
                "LShin", "LFoot", "RThigh", "RShin", "RFoot"]

results = []
pid_dirs = sorted([d for d in glob.glob(os.path.join(MT_PATH, "PID *"))
                   if os.path.basename(d) in VALID_PIDS])

print(f"Found {len(pid_dirs)} participants\n")

for p, pid_dir in enumerate(pid_dirs):
    pid = os.path.basename(pid_dir)

    # Just need one CSV
    csvs = glob.glob(os.path.join(pid_dir, "*.csv"))
    if not csvs:
        print(f"[{p+1}/{len(pid_dirs)}] {pid} — NO CSVs")
        continue

    
    df = pd.read_csv(csvs[0], on_bad_lines="skip", skiprows=[
                     0], low_memory=False, nrows=5)
    df.drop(df.index[1], inplace=True)

    df.columns = df.iloc[0:3, :].apply(
        lambda x: " ".join(x.dropna().astype(str)), axis=0
    )
    df.drop(df.index[0:3], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.columns = (
        df.columns
        .str.replace("Position X", "Pos[0]")
        .str.replace("Position Y", "Pos[1]")
        .str.replace("Position Z", "Pos[2]")
    )

    
    pattern = "|".join(KEEP_MARKERS)
    pos_cols = [c for c in df.columns if any(
        m in c for m in KEEP_MARKERS) and "Pos[" in c]

    # Get first row values
    row = df[pos_cols].astype(float).iloc[0]

    # Extract 3D positions
    def get_pos(name):
        return np.array([row[[c for c in pos_cols if name in c and "Pos[0]" in c][0]],
                         row[[c for c in pos_cols if name in c and "Pos[1]" in c][0]],
                         row[[c for c in pos_cols if name in c and "Pos[2]" in c][0]]])

    pelvis = get_pos(":Skeleton 001")
    l_thigh = get_pos("LThigh")
    l_shin = get_pos("LShin")
    l_foot = get_pos("LFoot")
    r_thigh = get_pos("RThigh")
    r_shin = get_pos("RShin")
    r_foot = get_pos("RFoot")

    # Leg length = pelvis-to-thigh + thigh-to-shin + shin-to-foot
    l_leg = (np.linalg.norm(pelvis - l_thigh) +
             np.linalg.norm(l_thigh - l_shin) +
             np.linalg.norm(l_shin - l_foot))

    r_leg = (np.linalg.norm(pelvis - r_thigh) +
             np.linalg.norm(r_thigh - r_shin) +
             np.linalg.norm(r_shin - r_foot))

    avg_leg = (l_leg + r_leg) / 2

    results.append({"pid": pid, "leg_length_L": l_leg, "leg_length_R": r_leg,
                    "leg_length_avg": avg_leg})
    print(f"[{p+1}/{len(pid_dirs)}] {pid} — L: {l_leg:.4f} m, R: {r_leg:.4f} m, Avg: {avg_leg:.4f} m")

df_out = pd.DataFrame(results)
print("\n=== SUMMARY ===")
print(df_out.to_string(index=False))

df_out.to_csv(r"D:\Gait_Analysis\participant_leg_lengths.csv", index=False)
print("\nSaved to D:\\Gait_Analysis\\participant_leg_lengths.csv")
