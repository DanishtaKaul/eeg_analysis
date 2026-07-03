"""Young vs Old comparison of leg length and height"""

import pandas as pd
from scipy import stats

legs = pd.read_csv(r"D:\Gait_Analysis\participant_leg_lengths.csv")
heights = pd.read_excel(
    r"D:\Gait_Analysis\Demographic Data & Physical Activity Levels.xlsx")
heights["pid"] = "PID " + heights["Participant ID"].astype(str)
heights["height_m"] = heights["Height"].astype(
    str).str.replace("cm", "").astype(float) / 100

Young = {
    "PID 3", "PID 4", "PID 5", "PID 6", "PID 7", "PID 8", "PID 9",
    "PID 10", "PID 11", "PID 14", "PID 15", "PID 18", "PID 19",
    "PID 20", "PID 23", "PID 26", "PID 30", "PID 31", "PID 35",
    "PID 50", "PID 57", "PID 58"
}

Old = {
    "PID 13", "PID 17", "PID 22", "PID 24", "PID 25", "PID 27",
    "PID 28", "PID 29", "PID 32", "PID 33", "PID 36", "PID 38",
    "PID 41", "PID 43", "PID 45", "PID 46", "PID 49",
    "PID 52", "PID 55"
}

eeg_pids = Young | Old

legs["age"] = legs["pid"].apply(
    lambda x: "young" if x in Young else ("old" if x in Old else "exclude"))
df = legs.merge(heights[["pid", "height_m"]], on="pid")
df = df[df["pid"].isin(eeg_pids)]

print(f"Young N = {len(df[df['age'] == 'young'])}")
print(f"Old N = {len(df[df['age'] == 'old'])}")

young_legs = df[df["age"] == "young"]["leg_length_avg"]
old_legs = df[df["age"] == "old"]["leg_length_avg"]
young_height = df[df["age"] == "young"]["height_m"]
old_height = df[df["age"] == "old"]["height_m"]

print("\n=== NORMALITY (Shapiro-Wilk) ===")
print(
    f"Leg length - Young: W = {stats.shapiro(young_legs).statistic:.3f}, p = {stats.shapiro(young_legs).pvalue:.4f}")
print(
    f"Leg length - Old:   W = {stats.shapiro(old_legs).statistic:.3f}, p = {stats.shapiro(old_legs).pvalue:.4f}")
print(
    f"Height - Young:     W = {stats.shapiro(young_height).statistic:.3f}, p = {stats.shapiro(young_height).pvalue:.4f}")
print(
    f"Height - Old:       W = {stats.shapiro(old_height).statistic:.3f}, p = {stats.shapiro(old_height).pvalue:.4f}")

# Leg length - Welch's t-test
result_leg = stats.ttest_ind(young_legs, old_legs, equal_var=False)
print(f"\n=== LEG LENGTH (Welch's t-test) ===")
print(
    f"Young: N = {len(young_legs)}, mean = {young_legs.mean():.4f} m, SD = {young_legs.std():.4f}")
print(
    f"Old:   N = {len(old_legs)}, mean = {old_legs.mean():.4f} m, SD = {old_legs.std():.4f}")
print(f"t = {result_leg.statistic:.3f}, df = {result_leg.df:.2f}, p = {result_leg.pvalue:.4f}")

# Height - Mann-Whitney U (normality violated for old group)
u_stat, p_mann = stats.mannwhitneyu(
    young_height, old_height, alternative='two-sided')
print(f"\n=== HEIGHT (Mann-Whitney U) ===")
print(f"Young: N = {len(young_height)}, mean = {young_height.mean():.4f} m, SD = {young_height.std():.4f}, median = {young_height.median():.4f}")
print(f"Old:   N = {len(old_height)}, mean = {old_height.mean():.4f} m, SD = {old_height.std():.4f}, median = {old_height.median():.4f}")
print(f"U = {u_stat:.1f}, p = {p_mann:.4f}")
