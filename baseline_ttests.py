# -*- coding: utf-8 -*-

"""Baseline t-tests: paired t-tests across Light conditions (Dark/Light/Ambient)
and a Young vs Old Welch t-test on frontal-ROI baseline alpha"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as tdist
from scipy.stats import ttest_rel


# ==== Light baseline paired t-tests on alpha_db (within-subject) ====

LIGHT_CSV = r"D:\cluster_outputs\baseline_light_main_band_power_seed42.csv"


def paired_stats(a, b, alpha=0.05):
    import numpy as np
    from scipy import stats
    # a, b are 1D arrays (per-participant)
    t, p = stats.ttest_rel(a, b)
    d = a - b
    n = d.size
    df = n - 1                                   
    dz = d.mean() / d.std(ddof=1)                
    se = d.std(ddof=1) / np.sqrt(n)
    from scipy.stats import t as tdist
    tcrit = tdist.ppf(1 - alpha/2, df=df)
    lo, hi = d.mean() - tcrit*se, d.mean() + tcrit*se
    return dict(
        n=int(n), df=int(df),                    
        t=float(t), p=float(p),
        mean_diff=float(d.mean()),
        ci_low=float(lo), ci_high=float(hi),
        cohen_dz=float(dz)
    )


try:
    dfL = pd.read_csv(LIGHT_CSV)
    
    dfL = dfL[dfL["PID"].astype(str).str.match(r"PID\s+\d+")].copy()

    # pivot to per-participant wide table: one alpha value per light level
    alpha_w = dfL.pivot_table(
        index="PID", columns="light", values="alpha_db_uv")

    # Require all three conditions present
    alpha_w = alpha_w[["LIGHT", "AMBIENT", "DARK"]].dropna()

    # Descriptives (grand means)
    gm = alpha_w.mean()
    print("\n=== Baseline alpha (dB) grand means by Light ===")
    print(f"  LIGHT   : {gm['LIGHT']:.3f} dB")
    print(f"  AMBIENT : {gm['AMBIENT']:.3f} dB")
    print(f"  DARK    : {gm['DARK']:.3f} dB")

    # Paired t-tests (within-subject)
    res_DL = paired_stats(alpha_w["DARK"].values,
                          alpha_w["LIGHT"].values)    # Dark - Light
    res_DA = paired_stats(alpha_w["DARK"].values,
                          alpha_w["AMBIENT"].values)  # Dark - Ambient
    res_LA = paired_stats(alpha_w["LIGHT"].values,
                          alpha_w["AMBIENT"].values)  # Light - Ambient

    print("\n=== Paired t-tests on baseline alpha (dB) — Light conditions ===")
    print(f"Dark - Light   : ={res_DL['mean_diff']:.3f} dB "
          f"[{res_DL['ci_low']:.3f}, {res_DL['ci_high']:.3f}], "
          f"t({res_DL['n']-1})={res_DL['t']:.3f}, p={res_DL['p']:.4f}, dz={res_DL['cohen_dz']:.3f}")
    print(f"Dark - Ambient : ={res_DA['mean_diff']:.3f} dB "
          f"[{res_DA['ci_low']:.3f}, {res_DA['ci_high']:.3f}], "
          f"t({res_DA['n']-1})={res_DA['t']:.3f}, p={res_DA['p']:.4f}, dz={res_DA['cohen_dz']:.3f}")
    print(f"Light - Ambient: ={res_LA['mean_diff']:.3f} dB "
          f"[{res_LA['ci_low']:.3f}, {res_LA['ci_high']:.3f}], "
          f"t({res_LA['n']-1})={res_LA['t']:.3f}, p={res_LA['p']:.4f}, dz={res_LA['cohen_dz']:.3f}")

    # save a small CSV 
    out_dir_L = Path(LIGHT_CSV).parent / "ttest"
    out_dir_L.mkdir(exist_ok=True)
    pd.DataFrame([
        {"contrast": "Dark - Light",   **res_DL},
        {"contrast": "Dark - Ambient", **res_DA},
        {"contrast": "Light - Ambient", **res_LA},
        {"LIGHT_mean_dB": gm["LIGHT"],
            "AMBIENT_mean_dB": gm["AMBIENT"], "DARK_mean_dB": gm["DARK"]}
    ]).to_csv(out_dir_L / f"baseline_light_paired_ttests_from_{Path(LIGHT_CSV).stem}.csv",
              index=False)
    print(
        f"\nSaved Light t-tests - {out_dir_L / ('baseline_light_paired_ttests_from_' + Path(LIGHT_CSV).stem + '.csv')}")

except Exception as e:
    print(f"\n[WARN] Skipped Light paired t-tests: {e}")



##### frontal alpha baseline main effect of age #####
# Young vs Old Welch t-test on frontal-ROI baseline alpha
# (input: baseline_alpha_FRONTALROI_age_main_seed42.csv, from the frontal ROI script).


# ---- CONFIG ----
CSV = r"D:\cluster_outputs\baseline_alpha_FRONTALROI_age_main_seed42.csv"  
BANDS = ["alpha_db_uv"]   # only alpha needed for ROI baseline
Y_LABEL = "Young"
O_LABEL = "Old"
ALPHA = 0.05
# ---------------

csv_path = Path(CSV)
if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path)

# keep only participant-level averages from  ROI CSV 
if "condition" in df.columns:
    df = df[df["condition"] == "MEAN_ACROSS_CONDITIONS"]

# Keep only participant rows (drop group means) and only the two groups
df = df[df["group"].isin([Y_LABEL, O_LABEL])].copy()
df = df[~df["PID"].astype(str).str.startswith("GROUP_MEAN")]

# Strict checks on group presence and sizes (before band filtering)
present_groups = set(df["group"].unique().tolist())
expected_groups = {Y_LABEL, O_LABEL}
if present_groups != expected_groups:
    raise ValueError(
        f"Expected groups {expected_groups}, found {present_groups}")


def welch_stats(y, o):
    """Welch t-test for Young vs Old, no confidence interval."""
    y = np.asarray(y, float)
    o = np.asarray(o, float)

    ny, no = y.size, o.size
    if ny < 2 or no < 2:
        raise ValueError(
            f"Need at least 2 per group (Young n={ny}, Old n={no}).")

    # Welch t-test
    t_stat, p_two = stats.ttest_ind(y, o, equal_var=False)

    # Welch-Satterthwaite df
    vy, vo = np.var(y, ddof=1), np.var(o, ddof=1)
    df_w = (vy/ny + vo/no)**2 / (
        (vy**2)/((ny**2)*(ny-1)) + (vo**2)/((no**2)*(no-1))
    )

    return dict(
        n_young=ny,
        n_old=no,
        mean_young=float(y.mean()),
        sd_young=float(y.std(ddof=1)),
        mean_old=float(o.mean()),
        sd_old=float(o.std(ddof=1)),
        diff_young_minus_old=float(y.mean() - o.mean()),
        t=float(t_stat),
        df=float(df_w),
        p_two=float(p_two),
    )


rows = []
for band in BANDS:
    # 1) Column must exist
    if band not in df.columns:
        raise KeyError(
            f"Column '{band}' not found in CSV. Available columns: {', '.join(df.columns)}")

    # 2) No NaNs allowed — list offending PIDs if any
    df_band = df[["PID", "group", band]].copy()
    missing = df_band[df_band[band].isna()]
    if not missing.empty:
        offenders = ", ".join(missing["PID"].astype(str).tolist())
        raise ValueError(
            f"Found NaN in '{band}' for PID(s): {offenders}. Fix your CSV first.")

    # 3) Build groups (strict)
    y_vals = df_band.loc[df_band["group"] == Y_LABEL, band].to_numpy()
    o_vals = df_band.loc[df_band["group"] == O_LABEL, band].to_numpy()
    if y_vals.size < 2 or o_vals.size < 2:
        raise ValueError(
            f"Not enough data for '{band}': Young n={y_vals.size}, Old n={o_vals.size} (need ≥2 each)")

    res = welch_stats(y_vals, o_vals)
    rows.append({"band": band, **res})

out = pd.DataFrame(rows)

# Console print
print("\n=== Young vs Old — Welch t-tests (dB) ===")
for _, r in out.iterrows():
    band = r["band"]
    print(f"\n[{band}]  n: {int(r['n_young'])} vs {int(r['n_old'])}")
    print(f"  Young mean±SD: {r['mean_young']:.3f} ± {r['sd_young']:.3f} dB")
    print(f"  Old   mean±SD: {r['mean_old']:.3f} ± {r['sd_old']:.3f} dB")
    print(f"  Welch t({r['df']:.1f})={r['t']:.3f}, p={r['p_two']:.4f}")
    print(f"  Diff (Y-O): {r['diff_young_minus_old']:.3f} dB")

# Save to subfolder under cluster_outputs
out_dir = csv_path.parent / "ttest"
out_dir.mkdir(exist_ok=True)
out_csv = out_dir / f"baseline_ttests_from_{csv_path.stem}.csv"
out.round(6).to_csv(out_csv, index=False)
print(f"\nSaved results - {out_csv}")
