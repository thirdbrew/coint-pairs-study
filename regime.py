"""
regime -- did the edge change over the sample, or was it flat all along?

WHY THIS RUNS AT ALL. Two reasons, and the second is the sharper one.

1. The study registered a PRIOR it never tested. Rad, Low & Faff (2016) report
   that cointegration pair selection decayed sharply after 2009. This sample is
   entirely post-2009, so the prior predicts a flat null throughout rather than a
   decay visible inside the window -- but "predicts" is not "checked", and the
   difference between an edge that was never there and one that died mid-sample
   is exactly what a full-sample Sharpe hides.

2. `prereg.grade()` said so. When stage B passed it printed: "a pass is a licence
   to look harder, not to deploy. Check regime dependence (breaks.stability_report)
   before acting." The one positive Sharpe in this entire study is rolling OLS's
   +0.4620, and a positive number that lives in a single regime is not an edge --
   it is a story about one period. It gets the same treatment as the null.

DETECT FIRST, THEN TEST. `detect_breaks` locates break dates from the series
before anyone has an opinion about where they should be. Choosing the split after
seeing the data makes "the edge died in year X" unfalsifiable, because some date
always exists that makes a strategy look regime-dependent.

READ THE POWER TABLE BEFORE QUOTING A NULL FROM THIS. Measured in breaks.py over
25-30 synthetic runs:

    pre-Sharpe -> post-Sharpe    detected
       2.54    ->  0.00            100%
       1.59    ->  0.48             43%
       0.95    ->  0.79             13%
       0.79    ->  0.79 (none)      10%   <- false-positive rate

This detector finds a COLLAPSE reliably and a HALVING less than half the time, and
even a break it does find can be located two years from where it happened. So
"no breaks detected" means "no LARGE break detected" and is NOT evidence of
stability. The asymmetry is the useful part: a break it does find had to be big.

Writes reports/grade/regime.json.
"""

import json
import os

import numpy as np
import pandas as pd

import breaks as B
import stats_lib as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports", "grade")

SERIES = {
    "static_ols (stage A, graded)": "static_ols",
    "rolling_ols (stage C winner)": "rolling_ols",
}


def analyse(name, r):
    """Detect, test, attribute -- in that order."""
    print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
    print(f"  full sample: Sharpe {S.sharpe(r):+.4f}  over {len(r):,} days")

    brk = B.detect_breaks(r)
    dates = [pd.Timestamp(d) for d in brk] if len(brk) else []
    print(f"  breaks detected: {len(dates)}"
          + (f"  {[f'{d:%Y-%m}' for d in dates]}" if dates else ""))

    tests = []
    for d in dates:
        pos = int(r.index.searchsorted(d))
        try:
            t = B.chow_test(r.to_numpy(float), pos)
            tests.append({"date": f"{d:%Y-%m-%d}", "fstat": float(t["fstat"]),
                          "pvalue": float(t["pvalue"])})
            print(f"    Chow at {d:%Y-%m-%d}: F={t['fstat']:.3f}  p={t['pvalue']:.4f}")
        except Exception as e:
            print(f"    Chow at {d:%Y-%m-%d}: failed ({type(e).__name__})")

    attrib = []
    if dates:
        att = B.regime_attribution(r, dates)
        print("    per-regime:")
        for _, row in pd.DataFrame(att).iterrows():
            d = row.to_dict()
            attrib.append({k: (float(v) if isinstance(v, (int, float, np.floating))
                               else str(v)) for k, v in d.items()})
            print("      " + "  ".join(f"{k}={v}" for k, v in d.items()))

    # A fixed calendar halving, reported ALONGSIDE the detected split and never
    # instead of it. It is not a chosen break -- it is the midpoint, declared in
    # advance by arithmetic, so it cannot be tuned to flatter anything.
    mid = len(r) // 2
    h1, h2 = r.iloc[:mid], r.iloc[mid:]
    print(f"  fixed midpoint split ({r.index[mid]:%Y-%m}):")
    print(f"    first half  Sharpe {S.sharpe(h1):+.4f}   ({len(h1):,} days)")
    print(f"    second half Sharpe {S.sharpe(h2):+.4f}   ({len(h2):,} days)")

    return {
        "full_sharpe": float(S.sharpe(r)),
        "n_days": int(len(r)),
        "n_breaks": len(dates),
        "break_dates": [f"{d:%Y-%m-%d}" for d in dates],
        "chow_tests": tests,
        "attribution": attrib,
        "midpoint_date": f"{r.index[mid]:%Y-%m-%d}",
        "first_half_sharpe": float(S.sharpe(h1)),
        "second_half_sharpe": float(S.sharpe(h2)),
    }


def main():
    path = os.path.join(OUT, "estimator_returns.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    out = {}
    for label, col in SERIES.items():
        out[col] = analyse(label, df[col].dropna())

    print(f"\n{'=' * 66}")
    print("HOW TO READ A NULL HERE: breaks.py finds a Sharpe COLLAPSE 100% of the")
    print("time and a HALVING only 43%, with a 10% false-positive rate. 'No breaks")
    print("detected' therefore means 'no large break detected' and is NOT evidence")
    print("of stability. A break it DOES find is worth taking seriously.")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "regime.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote reports/grade/regime.json")


if __name__ == "__main__":
    main()
