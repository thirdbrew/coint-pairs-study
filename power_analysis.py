"""
power_analysis -- what could this study have detected, and what does it rule out?

WHY A NULL RESULT NEEDS THIS. "No edge found" is worth nothing on its own: a test
with no power finds nothing by construction. The question a reader should ask,
and that this file answers, is what edge the study COULD have seen and what its
data actually excludes.

Three separate quantities, routinely confused with each other:

  MINIMUM DETECTABLE SHARPE   the true Sharpe this sample could distinguish from
                              zero at conventional power. A function of T only.

  UPPER CONFIDENCE BOUND      the largest true Sharpe consistent with what was
                              observed. This is what a null RULES OUT, and it is
                              the number that decides whether the null is
                              informative or merely quiet.

  MINIMUM BACKTEST LENGTH     how much data a search of this size needs before
                              its winner means anything (Bailey-Lopez de Prado,
                              MinBTL ~ 2 ln(N) / SR^2). A function of the SEARCH,
                              not of the sample -- and the one that bites here.

The three can disagree, and saying so is the point. A study can be underpowered
for its own registered bar and still exclude that bar, if the observed value
lands far enough below it.

Writes reports/grade/power_analysis.json.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import stats_lib as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports", "grade")

REGISTERED_BAR = 0.50
ALPHA = 0.05
POWER = 0.80


def min_detectable_sharpe(n_days, alpha=ALPHA, power=POWER,
                          periods=S.TRADING_DAYS):
    """Smallest true annualised Sharpe distinguishable from zero, one-sided.

    sd of an annualised Sharpe estimate is ~sqrt(periods / T) near SR=0, so the
    detectable effect is (z_{1-alpha} + z_{power}) times that.
    """
    se = np.sqrt(periods / n_days)
    return float((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * se), float(se)


def main():
    path = os.path.join(OUT, "returns_10bps.csv")
    rets = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    with open(os.path.join(OUT, "stage_A_results.json"), encoding="utf-8") as fh:
        res = json.load(fh)

    n = len(rets)
    observed = float(S.sharpe(rets))
    _, lo, hi = S.sharpe_ci(rets)
    n_trials = res["n_trials"]
    sr_star = res["sr_star_annualised"]

    mds, se = min_detectable_sharpe(n)
    minbtl_1 = float(S.min_backtest_length(n_trials, target_sharpe=1.0))
    minbtl_bar = float(S.min_backtest_length(n_trials, target_sharpe=REGISTERED_BAR))
    minbtl_star = float(S.min_backtest_length(n_trials, target_sharpe=sr_star))
    years = n / S.TRADING_DAYS

    print("=== the sample ===")
    print(f"  trading days                 {n:,}  ({years:.1f} years)")
    print(f"  observed annualised Sharpe   {observed:+.4f}")
    print(f"  95% CI (stationary bootstrap) [{lo:+.4f}, {hi:+.4f}]")
    print(f"  se of an annualised Sharpe   {se:.4f}")

    print("\n=== what it could have detected ===")
    print(f"  minimum detectable Sharpe (alpha={ALPHA}, power={POWER:.0%})  "
          f"{mds:+.4f}")
    if mds > REGISTERED_BAR:
        print(f"  -> ABOVE the registered bar of {REGISTERED_BAR:.2f}. This sample "
              f"could NOT reliably\n     detect an edge exactly at its own bar. "
              f"State this, do not bury it.")
    else:
        print(f"  -> below the registered bar of {REGISTERED_BAR:.2f}; the sample "
              f"was adequate for it.")

    print("\n=== what it actually rules out ===")
    print(f"  upper 95% confidence bound   {hi:+.4f}")
    if hi < REGISTERED_BAR:
        print(f"  -> EXCLUDES the registered bar of {REGISTERED_BAR:.2f}. The null "
              f"is INFORMATIVE:\n     whatever edge exists here, it is smaller than "
              f"the bar this study registered.")
    else:
        print(f"  -> does NOT exclude {REGISTERED_BAR:.2f}. The null is quiet, not "
              f"informative.")

    print("\n=== what the SEARCH demands (Bailey-Lopez de Prado MinBTL) ===")
    print(f"  trials in the search         {n_trials:,}")
    print(f"  years needed to validate SR=1.00   {minbtl_1:6.1f}")
    print(f"  years needed to validate SR={REGISTERED_BAR:.2f}   {minbtl_bar:6.1f}")
    print(f"  years available                    {years:6.1f}")
    print(f"  -> a search this large needs {minbtl_1:.0f} years before a Sharpe of "
          f"1.0 is\n     distinguishable from the best of {n_trials:,} coin flips. "
          f"This study has {years:.1f}.")
    print(f"  This is the binding constraint, and it is a property of the SEARCH,")
    print(f"  not of the sample length. Testing fewer pairs would not fix the")
    print(f"  strategy; it would lower the bar the strategy has to clear.")

    out = {
        "days": n, "years": years,
        "observed_sharpe": observed,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "sharpe_se": se,
        "min_detectable_sharpe": mds,
        "alpha": ALPHA, "power": POWER,
        "registered_bar": REGISTERED_BAR,
        "underpowered_for_own_bar": bool(mds > REGISTERED_BAR),
        "upper_bound_excludes_bar": bool(hi < REGISTERED_BAR),
        "n_trials": n_trials,
        "minbtl_years_sr1": minbtl_1,
        "minbtl_years_at_bar": minbtl_bar,
        "minbtl_years_at_sr_star": minbtl_star,
        "sr_star": sr_star,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "power_analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote reports/grade/power_analysis.json")


if __name__ == "__main__":
    main()
