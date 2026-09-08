"""
grade_C -- stage C. Three hedge-ratio estimators, compared by Model Confidence Set.

THE POINT OF DOING IT THIS WAY. Running three estimators and reporting the best
is a mechanism hunt, and the parent repo has four consecutive pre-registered
failures on record from that reflex. Declaring the candidate set up front and
letting the Model Confidence Set return the subset that cannot be distinguished
from the best is a model comparison. Same three estimators, opposite epistemics,
and the difference is entirely in whether the choice is made before or after the
numbers are seen.

MCS answers "which of these can I not rule out?", which is the honest question
when the differences are small. It can return all three -- and given stage A
graded 0/3 with a Sharpe CI containing zero, that is the registered expectation.

TWO SETTINGS ARE NOT PREFERENCES, both from the docstring at stats_lib.py:308:
  criterion="sharpe"  -- arch's MCS eliminates by expected LOSS, so the default
                         "mean" builds the set on mean return. A registration
                         that says "MCS on Sharpe" is not satisfied by it.
  reps=20000          -- at reps=1000 the Monte Carlo stderr on a p near 0.10 is
                         ~0.010, which is the entire distance between 0.09 and
                         0.11. A verdict taken there is a coin flip in a suit.

Bar frozen before this ran: pairs-trading blob ec59631b.
"""

import json
import os

import numpy as np
import pandas as pd

import formation as F
import hedge_ratio as H
import stats_lib as S
import trade as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports", "grade")

REPS = 20000
SIZE = 0.05


def main():
    panel = F.download()
    os.makedirs(OUT, exist_ok=True)

    series = {}
    for name, fn in H.ESTIMATORS.items():
        print(f"\n== {name} ==", flush=True)
        beta_fn = None if name == "static_ols" else fn
        rets, _ = T.run_all(panel, cost_bps=10.0, beta_fn=beta_fn, verbose=False)
        series[name] = rets
        print(f"  days {len(rets)} | total {rets.sum():+.2%} | "
              f"Sharpe {S.sharpe(rets):+.4f}", flush=True)

    df = pd.DataFrame(series).dropna()
    print(f"\naligned panel: {df.shape[0]} days x {df.shape[1]} estimators")

    print(f"\n== Model Confidence Set (criterion='sharpe', reps={REPS:,}, "
          f"size={SIZE}) ==")
    mcs = S.model_confidence_set(df, size=SIZE, reps=REPS, criterion="sharpe")
    print(f"  included: {mcs['included']}")
    print(f"  excluded: {mcs['excluded']}")
    for k, v in sorted(mcs["pvalues"].items()):
        print(f"    {k:<12} p = {v:.4f}   (MC stderr {S.mc_stderr(v, REPS):.4f})")

    excluded = 1.0 if "static_ols" in mcs["excluded"] else 0.0

    results = {
        "static_ols_excluded": excluded,
        "mcs_included": mcs["included"],
        "mcs_excluded": mcs["excluded"],
        "mcs_pvalues": mcs["pvalues"],
        "mcs_criterion": mcs["criterion"],
        "mcs_reps": mcs["reps"],
        "mcs_size": mcs["size"],
        "sharpes": {k: float(S.sharpe(v)) for k, v in series.items()},
        "sharpe_cis": {k: [float(x) for x in S.sharpe_ci(v)[1:]]
                       for k, v in series.items()},
        "total_returns": {k: float(v.sum()) for k, v in series.items()},
        "days": int(df.shape[0]),
    }
    path = os.path.join(OUT, "stage_C_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    df.to_csv(os.path.join(OUT, "estimator_returns.csv"), encoding="utf-8")

    print("\n  Sharpe by estimator (95% CI):")
    for k in series:
        lo, hi = results["sharpe_cis"][k]
        print(f"    {k:<12} {results['sharpes'][k]:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    print(f"\n  static_ols_excluded {excluded:.0f}   bar >= 1.0")
    print(f"\nwrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
