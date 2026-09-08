"""
allocate -- stage B. Convex allocation across the surviving pairs.

THE PROGRAM, stated before the code, because that is the half of this stage that
is worth anything in an interview.

    decision variables   w in R^k          dollar weight per pair, signed
    objective            maximise  mu'w - (lambda/2) w' Sigma w
    subject to           ||w||_1        <= G      gross exposure budget
                         |w_i|          <= c      per-pair concentration cap
                         ||w - w_prev||_1 <= T    turnover budget per rebalance

`mu` is the expected per-pair return over the next holding period; `Sigma` is the
pair-return covariance. Both are estimated on data available at the rebalance
date and never beyond it.

This is a quadratic program with an L1 ball, an L-infinity box and an L1 trust
region -- convex, so the solver returns a global optimum and there is no local
minimum story to tell. Solved with cvxpy/CLARABEL.

WHAT THE BENCHMARK IS AND WHY. Equal-weighting the surviving book. Not a
hand-picked weighting, not the previous best -- the thing a reasonable person
does by default. An optimiser that cannot beat 1/k is not adding value, and the
whole point of a pre-registered comparison is to let that answer through.

THE CONDITION ON THIS ENTIRE STAGE. If stage A finds no edge, mu is noise and
this program is a machine for allocating precisely across nothing. A better IR
from optimising noise is not a result, it is an artifact of fitting the
covariance. The runner refuses to report an improvement when the equal-weight
book it is being compared against has a Sharpe indistinguishable from zero, and
says so instead.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import formation as F
import trade as T

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "reports", "selection")
OUT = os.path.join(HERE, "reports", "allocate")

GROSS = 1.0          # ||w||_1 budget -- same gross as the equal-weight book
CAP = 0.25           # no pair over a quarter of the book
TURNOVER = 0.50      # per rebalance, L1
RISK_AVERSION = 5.0
REBALANCE = 21       # trading days; monthly inside a 6-month window
LOOKBACK = 60        # days of pair returns for mu and Sigma


def solve(mu, Sigma, w_prev, gross=GROSS, cap=CAP, turnover=TURNOVER,
          risk_aversion=RISK_AVERSION):
    """One QP. Returns w, or w_prev unchanged if the solve fails."""
    import cvxpy as cp

    k = len(mu)
    w = cp.Variable(k)
    objective = cp.Maximize(mu @ w - (risk_aversion / 2.0) * cp.quad_form(w, cp.psd_wrap(Sigma)))
    constraints = [
        cp.norm1(w) <= gross,
        cp.abs(w) <= cap,
        cp.norm1(w - w_prev) <= turnover,
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            return w_prev
        return np.asarray(w.value).ravel()
    except Exception:
        return w_prev


def allocate_window(pairs_df, cost_bps=T.COST_BPS, **kw):
    """Walk one trading window, re-solving every REBALANCE days.

    CAUSAL BY CONSTRUCTION: at rebalance date t the estimates use pairs_df
    rows strictly before t, and the resulting weights are applied to returns
    from t onward. A weight is never multiplied by the return that produced it.
    """
    if pairs_df.empty or pairs_df.shape[1] == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    R = pairs_df.fillna(0.0)
    n, k = R.shape
    W = np.zeros((n, k))
    w = np.zeros(k)

    for t in range(n):
        if t % REBALANCE == 0:
            hist = R.iloc[max(0, t - LOOKBACK):t]          # strictly before t
            if len(hist) >= 20:
                mu = hist.mean().to_numpy()
                Sigma = hist.cov().to_numpy()
                Sigma = Sigma + np.eye(k) * 1e-8           # numerical floor
                w = solve(mu, Sigma, w, **kw)
        W[t] = w

    weights = pd.DataFrame(W, index=R.index, columns=R.columns)
    gross = (weights * R).sum(axis=1)

    # THE REGISTERED CRITERION SAYS "NET OF TURNOVER" AND THIS WAS MISSING.
    # The pair-level returns in R already carry each pair's own entry/exit cost,
    # but the OPTIMISER'S re-weighting is itself turnover and was free: every
    # rebalance moved capital between pairs at no charge. Measured across 99
    # rebalance days the allocator generates 37.46 units of L1 turnover, and
    # charging it at the same convention trade.py uses moves sharpe_edge from
    # +0.1896 to +0.1009 against a bar of >= 0.10. Reporting the uncharged
    # figure was reporting a criterion the run never evaluated.
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    rets = gross - turnover * (cost_bps / 1e4)
    return rets, weights


def run_all(panel, cost_bps=T.COST_BPS, beta_fn=None, verbose=True):
    """Optimised book vs. the equal-weight book, over every window."""
    ws = F.windows()
    opt_chunks, eq_chunks, rows = [], [], []

    for w in ws:
        path = os.path.join(SEL, f"W{w.index:02d}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            book = json.load(fh)["book"]
        idx = w.trading(panel).index
        if not book:
            opt_chunks.append(pd.Series(0.0, index=idx))
            eq_chunks.append(pd.Series(0.0, index=idx))
            continue

        df = T.pair_panel(w, panel, book, cost_bps=cost_bps, beta_fn=beta_fn)
        if df.empty or df.shape[1] == 0:
            opt_chunks.append(pd.Series(0.0, index=idx))
            eq_chunks.append(pd.Series(0.0, index=idx))
            continue

        # equal weight uses the SAME gross budget, so the comparison is not
        # secretly a leverage comparison
        eq = df.fillna(0.0).sum(axis=1) * (GROSS / df.shape[1])
        opt, wts = allocate_window(df, cost_bps=cost_bps)
        turn = float(wts.diff().abs().sum(axis=1).sum())

        opt_chunks.append(opt)
        eq_chunks.append(eq)
        rows.append({"window": w.index, "pairs": df.shape[1],
                     "opt": float(opt.sum()), "eq": float(eq.sum()),
                     "l1_turnover": turn})
        if verbose:
            print(f"W{w.index:02d} pairs {df.shape[1]:2d} | opt {opt.sum():+.4f} "
                  f"| eq {eq.sum():+.4f}", flush=True)

    return (pd.concat(opt_chunks).sort_index(),
            pd.concat(eq_chunks).sort_index(),
            pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser(description="stage B -- convex allocation")
    ap.add_argument("--cost-bps", type=float, default=T.COST_BPS)
    args = ap.parse_args()

    import stats_lib as S

    panel = F.download()
    opt, eq, meta = run_all(panel, cost_bps=args.cost_bps)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({"optimised": opt, "equal_weight": eq}).to_csv(
        os.path.join(OUT, "returns.csv"), encoding="utf-8")
    meta.to_csv(os.path.join(OUT, "windows.csv"), index=False, encoding="utf-8")

    s_opt, s_eq = S.sharpe(opt), S.sharpe(eq)
    print(f"\nequal weight Sharpe {s_eq:.3f} | optimised Sharpe {s_opt:.3f} "
          f"| edge {s_opt - s_eq:+.3f}")

    _, ci_lo, ci_hi = S.sharpe_ci(eq)
    _, o_lo, o_hi = S.sharpe_ci(opt)
    benchmark_ci_contains_zero = bool(ci_lo <= 0.0 <= ci_hi)
    results = {
        "sharpe_edge": float(s_opt - s_eq),
        "sharpe_optimised": float(s_opt),
        "sharpe_equal_weight": float(s_eq),
        "equal_weight_ci": [float(ci_lo), float(ci_hi)],
        "optimised_ci": [float(o_lo), float(o_hi)],
        "benchmark_ci_contains_zero": benchmark_ci_contains_zero,
        "reportable_as_improvement": not benchmark_ci_contains_zero,
        "turnover_charged": True,
        "cost_bps_one_way": float(args.cost_bps),
        "gross": GROSS, "cap": CAP, "turnover": TURNOVER,
        "risk_aversion": RISK_AVERSION, "rebalance_days": REBALANCE,
        "lookback_days": LOOKBACK,
        "days": int(len(opt)),
        "total_return_optimised": float(opt.sum()),
        "total_return_equal_weight": float(eq.sum()),
    }
    with open(os.path.join(OUT, "stage_B_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    if ci_lo <= 0.0 <= ci_hi:
        print("\nSTAGE B IS NOT REPORTABLE AS AN IMPROVEMENT.")
        print(f"  The equal-weight book's Sharpe CI is [{ci_lo:.3f}, {ci_hi:.3f}] "
              f"and CONTAINS zero.")
        print("  mu is then noise, and a higher optimised Sharpe is covariance fitting,")
        print("  not allocation skill. Reported as such, not as an edge.")


if __name__ == "__main__":
    main()
