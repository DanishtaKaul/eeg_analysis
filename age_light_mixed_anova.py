# -*- coding: utf-8 -*-
"""Age × Light mixed ANOVA on ROI band power (Frontal-theta, Parietal-alpha, Central-beta) for the PREP segment"""

import os
import numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path
import mne

from cluster_config import (
    base_dir, aligned_epochs_dir,
    sample_pid, sample_condition
)
from extract_prep_reset import load_and_extract_prep_reset


# ================== OUTPUT ==================
OUT_DIR = Path("age_light_anova_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ================== CONFIG ==================
BASE_DIR = Path(base_dir)

LIGHT_LEVELS = ["LIGHT", "DARK"]
OBSTACLE_LEVELS = [
    "EXPECTED_PRESENT",
    "EXPECTED_ABSENT",
    "UNEXPECTED_PRESENT",
    "UNEXPECTED_ABSENT",
]

TARGET_TESTS = [
    {"roi": "FRONTAL",  "band": "THETA", "fmin": 4,  "fmax": 7},
    {"roi": "PARIETAL", "band": "ALPHA", "fmin": 8,  "fmax": 12},
    {"roi": "CENTRAL",  "band": "BETA",  "fmin": 13, "fmax": 30},
]

ROIS = {
    "FRONTAL":  ["FP1", "FPz", "FP2", "F7", "F3", "Fz", "F4", "F8"],
    "CENTRAL":  ["FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"],
    "PARIETAL": ["P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"],
}

participants = [
    "PID 3", "PID 4", "PID 5", "PID 6", "PID 7", "PID 8", "PID 9", "PID 10", "PID 11",
    "PID 13", "PID 14", "PID 15", "PID 17", "PID 18", "PID 19", "PID 20", "PID 22",
    "PID 23", "PID 24", "PID 25", "PID 26", "PID 27", "PID 28", "PID 29", "PID 30",
    "PID 31", "PID 32", "PID 33", "PID 35", "PID 36", "PID 38", "PID 41", "PID 42",
    "PID 43", "PID 45", "PID 46", "PID 49", "PID 50", "PID 52", "PID 55", "PID 57", "PID 58"
]

Young = {
    "PID 3", "PID 4", "PID 5", "PID 6", "PID 7", "PID 8", "PID 9", "PID 10", "PID 11",
    "PID 14", "PID 15", "PID 18", "PID 19", "PID 20", "PID 23", "PID 26", "PID 30",
    "PID 31", "PID 35", "PID 50", "PID 57", "PID 58"
}
Old = set(participants) - Young

# ================== CHANNEL ORDER ==================
cond_safe = sample_condition.replace(" ", "_")
epochs_path = os.path.join(
    aligned_epochs_dir, sample_pid, cond_safe,
    f"aligned_epochs_{cond_safe}.fif"
)
epochs = mne.read_epochs(epochs_path, preload=False, verbose="ERROR")
ch_names = epochs.info["ch_names"]

# ================== HELPERS ==================


def age_group(pid):
    return "Young" if pid in Young else "Old"


def extract_mean(data, roi, fmin, fmax, freqs):
    ch_idx = [ch_names.index(ch) for ch in ROIS[roi]]
    f_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    return float(data[np.ix_(ch_idx, f_idx, np.arange(data.shape[2]))].mean())


# ================== EXTRACTION ==================
rows = []

for pid in participants:
    age = age_group(pid)

    for light in LIGHT_LEVELS:
        prep_obs = []
        freqs_ref = None

        for obst in OBSTACLE_LEVELS:
            cond = f"GLOBAL_{light}_{obst}"
            prep, _, _, _, freqs = load_and_extract_prep_reset(
                pid, cond, base_dir=BASE_DIR, prep_window="last"
            )

            freqs_ref = freqs if freqs_ref is None else freqs_ref
            prep_obs.append(prep)

        prep_mean = np.mean(np.stack(prep_obs), axis=0)

        for test in TARGET_TESTS:
            mean_power = extract_mean(
                prep_mean,
                test["roi"],
                test["fmin"],
                test["fmax"],
                freqs_ref
            )

            rows.append({
                "pid": pid,
                "age": age,
                "light": light,
                "roi": test["roi"],
                "band": test["band"],
                "mean_power": mean_power
            })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "mean_power_TARGET_PREP.csv", index=False)

# ================== ANOVAS ==================
anova_rows = []

for test in TARGET_TESTS:
    roi = test["roi"]
    band = test["band"]

    sub = df.query("roi == @roi and band == @band")

    aov = pg.mixed_anova(
        data=sub,
        dv="mean_power",
        between="age",
        within="light",
        subject="pid"
    )

    for _, r in aov.iterrows():
        anova_rows.append({
            "roi": roi,
            "band": band,
            "effect": r["Source"],
            "df1": r["DF1"],
            "df2": r["DF2"],
            "F": r["F"],
            "p_unc": r["p-unc"],
            "np2": r["np2"]
        })

anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(OUT_DIR / "anova_results_TARGET_PREP.csv", index=False)


# ================== SIGNIFICANT RESULTS WITH MEANS ==================
ALPHA = 0.05

sig = anova_df.query("p_unc < @ALPHA").copy()

means = (
    df
    .groupby(["roi", "band", "age", "light"])["mean_power"]
    .mean()
    .reset_index()
)

sig_rows = []

for _, r in sig.iterrows():
    roi = r["roi"]
    band = r["band"]
    effect = r["effect"]

    entry = {
        "roi": roi,
        "band": band,
        "effect": effect,
        "F": r["F"],
        "p_unc": r["p_unc"],
        "np2": r["np2"],
    }

    # ---------- AGE main effect ----------
    if effect == "age":
        entry["Young_mean"] = means.query(
            "roi==@roi and band==@band and age=='Young'"
        )["mean_power"].mean()

        entry["Old_mean"] = means.query(
            "roi==@roi and band==@band and age=='Old'"
        )["mean_power"].mean()

    # ---------- LIGHT main effect ----------
    elif effect == "light":
        entry["LIGHT_mean"] = means.query(
            "roi==@roi and band==@band and light=='LIGHT'"
        )["mean_power"].mean()

        entry["DARK_mean"] = means.query(
            "roi==@roi and band==@band and light=='DARK'"
        )["mean_power"].mean()

    # ---------- INTERACTION ----------
    elif effect == "Interaction":
        for a in ["Young", "Old"]:
            for l in ["LIGHT", "DARK"]:
                entry[f"{a}_{l}_mean"] = means.query(
                    "roi==@roi and band==@band and age==@a and light==@l"
                )["mean_power"].mean()

    sig_rows.append(entry)

sig_df = pd.DataFrame(sig_rows)

sig_df.to_csv(
    OUT_DIR / "significant_TARGET_PREP_with_means.csv",
    index=False
)

print("[SAVED] significant_TARGET_PREP_with_means.csv")
