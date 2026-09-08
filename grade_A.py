"""
grade_A -- compute stage A's registered metrics and the cost curve.

Writes reports/grade/stage_A_results.json. It does NOT grade: the frozen bar
lives in the pairs-trading repo (blob 73543a53, pushed), because that repo has
the remote that makes "frozen" mean something. prereg.grade() is called from
there against this file, so the verdict sits next to the bar rather than next to
the code that produced the number.

THE THREE REGISTERED METRICS

  sharpe_net  annualised, net of 10 bps round-trip, return on committed capital.

  dsr         deflated Sharpe probability at n_trials = 2,465,737.
              deflated_sharpe needs the VARIANCE OF SHARPE ACROSS TRIALS, and
              2.47M pairs cannot all be traded. Estimating that variance from
              the pairs we SELECTED would be circular -- they were chosen for
              looking good. So it is estimated from a random sample of pairs
              drawn from the same windows and traded under identical rules:
              the null distribution of per-pair Sharpe for this protocol on
              this universe. That is the quantity the formula wants.

  spa_p       Hansen SPA against a ZERO-return benchmark, criterion="mean",
              reading the CONSISTENT p-value (Hansen's recommended one of the
              three).

              Two choices there, both forced. Benchmarking on SPY would grade a
              market-neutral book against a criterion blind to what it is, so
              the benchmark is zero. And criterion="sharpe" CANNOT take a
              zero-variance benchmark -- sharpe_losses raises
              "zero-variance series cannot be Sharpe-scaled" -- which is correct
              behaviour, since the Sharpe of a constant is undefined. Against a
              zero benchmark the question is whether mean return clears zero
              after accounting for the search, and that is criterion="mean".

Also reported, and registered as report-regardless: the cost curve and the
BREAKEVEN COST IN BPS, which on the registered prior is the number that decides
whether any of this is tradable.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd

import formation as F
import stats_lib as S
import trade as T

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "reports", "selection")
OUT = os.path.join(HERE, "reports", "grade")

N_TRIALS = 2_465_737
COST_GRID = [0, 2, 5, 10, 15, 20, 30, 40, 50]
N_NULL_PAIRS = 2000
N_NULL_BOOKS = 500
SEED = 42


def null_pair_pool(panel, n=N_NULL_PAIRS, seed=SEED, cost_bps=T.COST_BPS):
    """Return series for randomly drawn pairs, keyed by window.

    Random, NOT selected -- what the protocol produces from pairs it had no
    reason to like. Raw material for the trial distribution, not the trial
    distribution itself (see null_book_sharpes).
    """
    rng = random.Random(seed)
    pool = {}
    per_window = max(1, n // len(F.windows()))

    for w in F.windows():
        names = F.members(w, panel)
        if len(names) < 2:
            continue
        series = []
        for _ in range(per_window):
            a, b = rng.sample(names, 2)
            s = T.pair_returns(w, panel, a, b, cost_bps=cost_bps)
            if s is None or s.std() == 0 or not np.isfinite(s.std()):
                continue
            series.append(s)
        if series:
            pool[w.index] = series
    return pool


def null_book_sharpes(pool, n_books=N_NULL_BOOKS, book_size=20, seed=SEED):
    """Sharpe of randomly assembled BOOKS -- the trial distribution, horizon-matched.

    WHY NOT PER-PAIR SHARPE. The first version fed deflated_sharpe the Sharpes of
    single pairs over a median of 125 trading days, while the graded candidate is
    a 20-pair book over 3,642 days. Bailey-Lopez de Prado wants the dispersion of
    trial Sharpes AT THE CANDIDATE'S SAMPLE LENGTH, and Sharpe estimation noise
    scales as sqrt(periods_per_year / T). At T=125 that noise alone is
    sqrt(252/125) = 1.4199 -- almost exactly the 1.4451 the old version measured,
    whose skew (+0.04) and excess kurtosis (-0.56) confirm it was short rather
    than fat-tailed. It measured the horizon, not the search, and inflated SR* by
    roughly 5x: SR* = 1.4451 x 5.0434 = 7.288 is essentially sqrt(252/T) x 5.04.

    A trial here is shaped like the candidate: draw `book_size` random pairs per
    window from the pool, equal-weight them exactly as run_all does, and
    concatenate all 29 windows into one full-length series.
    """
    rng = random.Random(seed + 1)
    out = []
    for _ in range(n_books):
        chunks = []
        for series in pool.values():
            k = min(book_size, len(series))
            picked = rng.sample(series, k)
            chunks.append(pd.concat(picked, axis=1).sum(axis=1) / book_size)
        if not chunks:
            continue
        sr = S.sharpe(pd.concat(chunks).sort_index())
        if np.isfinite(sr):
            out.append(float(sr))
    return np.asarray(out)


def cost_curve(panel, grid=COST_GRID):
    rows = []
    for c in grid:
        r, _ = T.run_all(panel, cost_bps=float(c), verbose=False)
        rows.append({"cost_bps": c, "sharpe": float(S.sharpe(r)),
                     "total_return": float(r.sum())})
    return pd.DataFrame(rows)


def breakeven(curve):
    """Linear interpolation of the cost at which net Sharpe crosses zero."""
    c, s = curve["cost_bps"].to_numpy(float), curve["sharpe"].to_numpy(float)
    for i in range(len(c) - 1):
        if (s[i] > 0) != (s[i + 1] > 0):
            return float(c[i] + (c[i + 1] - c[i]) * s[i] / (s[i] - s[i + 1]))
    return float("inf") if s[-1] > 0 else float("-inf")


def main():
    ap = argparse.ArgumentParser(description="stage A -- compute registered metrics")
    ap.add_argument("--null-pairs", type=int, default=N_NULL_PAIRS)
    args = ap.parse_args()

    panel = F.download()
    os.makedirs(OUT, exist_ok=True)

    print("== base case: 10 bps round-trip ==")
    rets, meta = T.run_all(panel, cost_bps=10.0, verbose=True)
    sharpe_net = float(S.sharpe(rets))
    _, ci_lo, ci_hi = S.sharpe_ci(rets)

    print("\n== selection summary ==")
    sel = pd.read_csv(os.path.join(SEL, "summary.csv"))
    print(f"windows {len(sel)} | tests {sel.tested.sum():,} | "
          f"sig@0.05 {sel.sig_raw.sum():,} vs noise {sel.expected_fp.sum():,.0f} "
          f"(ratio {sel.sig_raw.sum() / sel.expected_fp.sum():.2f}x) | "
          f"FDR survivors {sel.sig_fdr.sum():,} | empty books {(sel.book_size == 0).sum()}")

    print(f"\n== trial Sharpe distribution ({args.null_pairs} random pairs "
          f"-> {N_NULL_BOOKS} random BOOKS, horizon-matched) ==")
    pool = null_pair_pool(panel, n=args.null_pairs)
    per_pair = np.asarray([float(S.sharpe(x)) for v in pool.values() for x in v])
    trial = null_book_sharpes(pool, n_books=N_NULL_BOOKS)
    print(f"  per-pair  (WRONG horizon, ~125d): n={len(per_pair)}  "
          f"sd {per_pair.std(ddof=1):.4f}   [sqrt(252/125) = 1.4199]")
    print(f"  per-book  (matched, 3642d):       n={len(trial)}  "
          f"sd {trial.std(ddof=1):.4f}  mean {trial.mean():+.4f}")

    # WHICH TRIAL DISPERSION TO USE, AND WHY THE EMPIRICAL ONE IS NOT IT.
    #
    # The 500 random books each draw 20 pairs from a pool of only ~69 per window,
    # so they SHARE most of their constituents. Their Sharpes are far more alike
    # than independent trials would be, and the measured sd comes out at 0.1096 --
    # just 0.417x the sd an annualised Sharpe estimate has at this sample length
    # from estimation noise alone, sqrt(252/3642) = 0.2630. Overlap-induced
    # correlation biases it DOWN, which would understate SR* and make the
    # deflation bar too easy to clear.
    #
    # So the graded figure uses the LARGER of the two. Erring conservative on a
    # deflation bar is the defensible direction -- the same reasoning that keeps
    # n_trials at the registered 2,465,737 against an actual 2,458,394. The
    # theoretical value also reproduces an independent adversarial estimate
    # (1.350 vs 1.327 here), which the empirical one does not.
    sd_empirical = float(np.std(trial, ddof=1))
    sd_theoretical = float(np.sqrt(S.TRADING_DAYS / len(rets)))
    sd_used = max(sd_empirical, sd_theoretical)

    dsr = float(S.deflated_sharpe(rets, n_trials=N_TRIALS,
                                  sharpe_variance=sd_used ** 2))
    sr_star = float(S.expected_max_sharpe(
        N_TRIALS, sd_used ** 2 / S.TRADING_DAYS)) * np.sqrt(S.TRADING_DAYS)
    print(f"  trial sd: empirical {sd_empirical:.4f} (overlap-biased LOW) vs "
          f"theoretical {sd_theoretical:.4f} -> using {sd_used:.4f}")

    print("\n== Hansen SPA vs zero-return benchmark ==")
    zero = pd.Series(0.0, index=rets.index)
    spa = S.spa_test(zero, pd.DataFrame({"coint_pairs": rets}), reps=20000,
                     criterion="mean")
    spa_p = float(spa["consistent"])
    print(f"SPA p = {spa_p:.4f}  (MC stderr {S.mc_stderr(spa_p, 20000):.4f})")

    print("\n== cost curve ==")
    curve = cost_curve(panel)
    print(curve.to_string(index=False))
    be = breakeven(curve)
    print(f"\nbreakeven cost: {be:.2f} bps round-trip")

    results = {
        "sharpe_net": sharpe_net,
        "dsr": dsr,
        "spa_p": spa_p,
        "spa_pvalues": spa["pvalues"],
        "spa_criterion": spa["criterion"],
        "spa_mc_stderr": float(S.mc_stderr(spa_p, 20000)),
        "sharpe_ci_lo": float(ci_lo),
        "sharpe_ci_hi": float(ci_hi),
        "n_trials": N_TRIALS,
        "sr_star_annualised": sr_star,
        "trial_sharpe_n": int(len(trial)),
        "trial_sharpe_sd": float(trial.std(ddof=1)),
        "trial_sd_empirical": sd_empirical,
        "trial_sd_theoretical": sd_theoretical,
        "trial_sd_used": sd_used,
        "trial_kind": "random 20-pair books, horizon-matched to the candidate",
        "trial_per_pair_sd_wrong_horizon": float(per_pair.std(ddof=1)),
        "trial_per_pair_n": int(len(per_pair)),
        "breakeven_bps": be,
        "days": int(len(rets)),
        "total_return": float(rets.sum()),
        "tests": int(sel.tested.sum()),
        "sig_raw": int(sel.sig_raw.sum()),
        "expected_fp": float(sel.expected_fp.sum()),
        "noise_ratio": float(sel.sig_raw.sum() / sel.expected_fp.sum()),
        "fdr_survivors": int(sel.sig_fdr.sum()),
        "empty_books": int((sel.book_size == 0).sum()),
        "cost_curve": curve.to_dict(orient="records"),
    }
    path = os.path.join(OUT, "stage_A_results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    rets.to_csv(os.path.join(OUT, "returns_10bps.csv"), encoding="utf-8")

    print(f"\nwrote {os.path.relpath(path, HERE)}")
    print(f"\n  sharpe_net {sharpe_net:+.4f}   bar >= 0.50")
    print(f"  dsr        {dsr:.4f}   bar >= 0.95   (SR* = {sr_star:.3f} annualised)")
    print(f"  spa_p      {spa_p:.4f}   bar <= 0.05")
    print("\nGrade from the pairs-trading repo, where the bar is frozen:")
    print('  python -c "import json,prereg; '
          "print(prereg.grade('coint_A_selection', "
          "json.load(open(r'%s'))) ['report'])\"" % path)


if __name__ == "__main__":
    main()
