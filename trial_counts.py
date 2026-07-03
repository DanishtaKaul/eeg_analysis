# -*- coding: utf-8 -*-
"""Count retained trials per participant by condition and flag low-count cells"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd

# ------------------------------
# Canonical levels / configuration
# ------------------------------
LIGHTS: List[str] = ['Light', 'Ambient', 'Dark']
EXPECTS: List[str] = ['Expected', 'Unexpected']
EXISTS: List[str] = ['Present', 'Absent']

# ---- Design maxima ----
DESIGN_MAX_TOTAL_PER_LIGHT: int = 80                     # EP+UP+EA+UA per Light
DESIGN_MAX_PER_LIGHT_OBS4: Dict[str, int] = {            # per Light, per obstacle type
    'EP': 20, 'UP': 20, 'EA': 20, 'UA': 20
}
DESIGN_MAX_OVERALL_OBS4: Dict[str, int] = {              # across all lights (default 3*per-light)
    k: 3*v for k, v in DESIGN_MAX_PER_LIGHT_OBS4.items()
}

# Derived thresholds (half-rule)
THRESH_TOTAL_PER_LIGHT: float = DESIGN_MAX_TOTAL_PER_LIGHT / 2
THRESH_PER_LIGHT_OBS4: Dict[str, float] = {
    k: v/2 for k, v in DESIGN_MAX_PER_LIGHT_OBS4.items()}
THRESH_OVERALL_OBS4: Dict[str, float] = {
    k: v/2 for k, v in DESIGN_MAX_OVERALL_OBS4.items()}


# ------------------------------
# Robust token parsing for keys
# ------------------------------
def _parse_tokens(label: str) -> Tuple[str, str, str]:
    """
    Map a condition label to (Light, Expectation, Existence).
    Accepts variations like FOREWARN/NOTFOREWARN, NOTEXPECTED, etc.
    """
    s = str(label).upper()

    # Light
    if 'LIGHT' in s:
        L = 'Light'
    elif 'AMBIENT' in s:
        L = 'Ambient'
    elif 'DARK' in s:
        L = 'Dark'
    else:
        L = 'UNKNOWN'

    # Expectation (FOREWARN - Expected; NOTFOREWARN/NOTEXPECTED - Unexpected)
    if any(t in s for t in ['UNEXPECTED', 'NOTEXPECTED', 'NOT_FOREWARN', 'NOTFOREWARN', 'NOFOREWARN']):
        E = 'Unexpected'
    elif any(t in s for t in ['EXPECTED', 'FOREWARN', 'FOREWARNED']):
        E = 'Expected'
    else:
        E = 'UNKNOWN'

    # Existence
    if 'ABSENT' in s:
        X = 'Absent'
    elif 'PRESENT' in s:
        X = 'Present'
    else:
        X = 'UNKNOWN'

    return L, E, X


# ------------------------------
# Main entry point
# ------------------------------
def summarize_counts_no_pool(
    ppid: str,
    aligned_epochs: dict,
    out_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Count trials per participant AFTER manual rejection, write CSVs, and return tidy DataFrame.
    """
    if out_root is None:
        out_root = Path(r"D:\final_trial_counts")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Build base-12 counts from aligned_epochs keys ----
    rows = []
    unknown_hits = 0
    for key, epochs in aligned_epochs.items():
        L, E, X = _parse_tokens(key)
        if 'UNKNOWN' in (L, E, X):
            unknown_hits += 1
        rows.append(dict(light=L, expectation=E,
                    existence=X, n=int(len(epochs))))
    df12 = pd.DataFrame(rows)

    # Complete the grid
    grid = pd.MultiIndex.from_product([LIGHTS, EXPECTS, EXISTS],
                                      names=['light', 'expectation', 'existence']).to_frame(index=False)
    df12 = grid.merge(
        df12, on=['light', 'expectation', 'existence'], how='left').fillna({'n': 0})
    df12['n'] = df12['n'].astype(int)

    # ---- View: by_light (collapse obstacle types) ----
    by_light = df12.groupby('light', as_index=False)['n'].sum()
    by_light['view'] = 'by_light'
    by_light['expectation'] = 'ALL'
    by_light['existence'] = 'ALL'
    by_light['obstacle4'] = 'ALL'

    # ---- View: obstacle4_overall (EP/UP/EA/UA across ALL lights) ----
    def o4(expect: str, exist: str, tag: str) -> dict:
        n = int(df12[(df12['expectation'] == expect) &
                (df12['existence'] == exist)]['n'].sum())
        return dict(view='obstacle4_overall', light='ALL',
                    expectation=expect, existence=exist, obstacle4=tag, n=n)

    obstacle4 = pd.DataFrame([
        o4('Expected',   'Present', 'EP'),
        o4('Unexpected', 'Present', 'UP'),
        o4('Expected',   'Absent',  'EA'),
        o4('Unexpected', 'Absent',  'UA'),
    ])

    # ---- View: by_light_obstacle4 (Light × EP/UP/EA/UA) ----
    perL_rows = []
    for L in LIGHTS:
        for (E, X, tag) in [
            ('Expected',   'Present', 'EP'),
            ('Unexpected', 'Present', 'UP'),
            ('Expected',   'Absent',  'EA'),
            ('Unexpected', 'Absent',  'UA'),
        ]:
            n = int(df12[(df12['light'] == L) & (df12['expectation'] == E) & (
                df12['existence'] == X)]['n'].sum())
            perL_rows.append(dict(view='by_light_obstacle4', light=L,
                                  expectation=E, existence=X, obstacle4=tag, n=n))
    perL = pd.DataFrame(perL_rows)

    # ---- View: base12 (12 cells) + obstacle4 tag per row ----
    base12 = df12.copy()
    base12['view'] = 'base12'
    base12['obstacle4'] = base12.apply(
        lambda r: ('EP' if (r.expectation == 'Expected' and r.existence == 'Present') else
                   'UP' if (r.expectation == 'Unexpected' and r.existence == 'Present') else
                   'EA' if (r.expectation == 'Expected' and r.existence == 'Absent') else
                   'UA'),
        axis=1
    )

    # ---- Stitch tidy and save ----
    tidy = pd.concat([
        by_light[['view', 'light', 'expectation',
                  'existence', 'obstacle4', 'n']],
        obstacle4[['view', 'light', 'expectation',
                   'existence', 'obstacle4', 'n']],
        perL[['view', 'light', 'expectation', 'existence', 'obstacle4', 'n']],
        base12[['view', 'light', 'expectation', 'existence', 'obstacle4', 'n']],
    ], ignore_index=True)
    tidy.insert(0, 'pid', ppid)

    # Paths
    tidy_path = out_root / f"counts_tidy_{ppid}.csv"
    base12_path = out_root / f"pivot_base12_{ppid}.csv"
    onefile_path = out_root / f"obstacle4_by_light_and_all_{ppid}.csv"
    bylight_path = out_root / f"pivot_by_light_{ppid}.csv"
    flags_path = out_root / f"flags_halfrule_{ppid}.csv"

    # Save tidy
    tidy.to_csv(tidy_path, index=False)
    print(f"[{ppid}] wrote {tidy_path}")

    # ---- obstacle4_by_light_and_all: EP/UP/EA/UA per light + ALL row ----
    per_light_table = (
        perL.pivot(index='light', columns='obstacle4', values='n')
            .reindex(index=LIGHTS, columns=['EP', 'UP', 'EA', 'UA'])
            .fillna(0).astype(int)
    )
    overall_row = (
        obstacle4.set_index('obstacle4')['n']
                 .reindex(['EP', 'UP', 'EA', 'UA'])
                 .fillna(0).astype(int)
    )
    per_light_with_all = per_light_table.copy()
    per_light_with_all.loc['ALL'] = overall_row
    per_light_with_all.reset_index().rename(
        columns={'index': 'light'}).to_csv(onefile_path, index=False)
    print(f"[{ppid}] wrote {onefile_path}")

    # ---- pivot_base12 (12-cell matrix) ----
    (df12.set_index(['light', 'expectation', 'existence'])['n']
        .unstack(['expectation', 'existence'])
        .reindex(index=LIGHTS)
     ).to_csv(base12_path)
    print(f"[{ppid}] wrote {base12_path}")

    # ---- pivot_by_light (totals per light; EP+UP+EA+UA) ----
    by_light_totals = (
        df12.groupby('light')['n'].sum()
            .reindex(LIGHTS)
            .rename('Total')
            .reset_index()
    )
    by_light_totals.to_csv(bylight_path, index=False)
    print(f"[{ppid}] wrote {bylight_path}")

    # ---- Half-rule flags for totals + EP/UP/EA/UA (per light and overall) ----
    flag_rows = []

    # (1) Per-light totals: flag if < half of DESIGN_MAX_TOTAL_PER_LIGHT
    for _, r in by_light_totals.iterrows():  # columns: light, Total
        lvl = r['light']
        n = int(r['Total'])
        flag_rows.append(dict(
            pid=ppid, category='per_light_total', level=lvl, cell='ALL',
            n=n, threshold=int(THRESH_TOTAL_PER_LIGHT), ok=(n >= THRESH_TOTAL_PER_LIGHT)
        ))

    # (2) Per-light obstacle types (EP/UP/EA/UA): flag if < half of DESIGN_MAX_PER_LIGHT_OBS4
    for L in LIGHTS:
        for cell in ['EP', 'UP', 'EA', 'UA']:
            n = int(per_light_with_all.loc[L, cell]
                    ) if L in per_light_with_all.index else 0
            thr = THRESH_PER_LIGHT_OBS4[cell]
            flag_rows.append(dict(
                pid=ppid, category='per_light_obstacle4', level=L, cell=cell,
                n=n, threshold=int(thr), ok=(n >= thr)
            ))

    # (3) Overall (across lights) obstacle types: flag if < half of DESIGN_MAX_OVERALL_OBS4
    for cell in ['EP', 'UP', 'EA', 'UA']:
        n = int(per_light_with_all.loc['ALL', cell])
        thr = THRESH_OVERALL_OBS4[cell]
        flag_rows.append(dict(
            pid=ppid, category='overall_obstacle4', level='ALL', cell=cell,
            n=n, threshold=int(thr), ok=(n >= thr)
        ))

    flags_df = pd.DataFrame(flag_rows)
    flags_df.to_csv(flags_path, index=False)
    print(f"[{ppid}] wrote {flags_path}")

    # surface any UNKNOWN parses
    if unknown_hits:
        print(f"[{ppid}] WARNING: {unknown_hits} aligned_epochs keys mapped to 'UNKNOWN' tokens. Check labels.")

    # console summary of failures
    fails = flags_df[~flags_df['ok']]
    if len(fails):
        print(f"[{ppid}] Half-rule FAILS:")
        for _, r in fails.iterrows():
            print(
                f"  - {r['category']} | {r['level']} | {r['cell']}: n={r['n']} < {r['threshold']}")

    return tidy
