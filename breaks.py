"""
breaks -- structural-break detection: find the regime change, don't pick it.
============================================================================

WHY THIS EXISTS
---------------
Failed Improvements §5 names this gap in its own words:

    "The both-halves requirement is a crude structural-break test -- the formal version is a
     Chow test on the coefficient across a suspected break date."

That entry describes a candidate that PASSED its pre-registered bar (Sharpe 1.16 vs 0.96,
positive edge in both halves) and was declined because an attribution table showed the edge
lived in 2015-2020 and had largely stopped by 2021. Two things were missing:

1. A formal test, rather than an eyeball on a table. `chow_test` is that.
2. A way to find the break WITHOUT choosing it. This matters more than it sounds. If the
   analyst picks the split date after seeing the data, the split is another researcher
   degree of freedom, and "the edge died in 2021" becomes unfalsifiable -- you can always
   find a date that makes a strategy look regime-dependent. `detect_breaks` locates break
   dates from the series itself, before anybody has an opinion about where they should be.

The honest ordering is: detect first, then test the detected date, then attribute. Not:
notice a bad stretch, then test that stretch.

WHAT A BREAK MEANS FOR A DECISION
---------------------------------
A detected break is not automatically disqualifying -- markets do change, and a strategy
that works in one regime may be worth holding if you can name the regime and say why it
persists. What it disqualifies is the claim that the FULL-SAMPLE number describes what you
should expect next. `regime_attribution` prints the per-regime numbers so that claim has to
be made explicitly rather than by averaging.

MEASURED CALIBRATION (2026-08-19, 25-30 synthetic runs of 2520 daily observations)
---------------------------------------------------------------------------------
Detection on raw returns, model="l2", min_size=252, penalty = 1.0 * var * ln(n):

    pre-Sharpe  post-Sharpe   detected   median err (business days)
       2.54        0.00          100%          45
       1.59        0.48           43%          90
       1.27        0.63           27%         188
       0.95        0.79           13%         280
       0.79        0.79 (none)    10%   <- false-positive rate

Reproduce with `python breaks.py`; those figures are that script's output, not hand-copied.

Those medians hide a long tail. Re-measured over 40 seeds at the largest effect size
(2.54 -> 0.00), where detection succeeds 38 times out of 40: median error 78 business days,
90th percentile 240, maximum 520. So even a break this detector finds easily can be located
two years from where it actually happened.

READ THAT SECOND COLUMN BEFORE QUOTING A NULL RESULT. This detector reliably finds a
collapse (Sharpe 2.5 -> 0). It finds a HALVING (1.59 -> 0.48) less than half the time.
"No breaks detected" therefore means "no large break detected", and is NOT evidence that
an edge is stable -- with ten years of daily data nothing could give you that. The
asymmetry is the useful part: a break this thing DOES find is worth taking seriously,
because it had to be big to show up at all.

Even when a break is found, the date carries months of uncertainty, so treat it as "around
here" and never as a precise event date to attach a story to.

Uses `ruptures` (deepcharles/ruptures) for detection and statsmodels for the Chow test.
Run `python breaks.py` to re-check the detector's false/true-positive calibration.
"""

import numpy as np
import pandas as pd


# Binary segmentation, not PELT. Measured on this machine at n=2520 (a 15y daily backtest,
# the primary use case here): Pelt+l2 took 914s, Binseg+l2 took 0.050s -- PELT's pruning
# degenerates on smooth, autocorrelated series, and an "rbf" cost builds an n x n kernel on
# top of that. A 15-minute detector does not get run, and a detector that does not get run
# finds no breaks at all.
DEFAULT_MODEL = "l2"
MAX_BREAKS = 5

# Penalty multiplier on a BIC-shaped term (var * ln n) per breakpoint. Calibrated on 2026-08-19
# against 25-30 synthetic runs of 2520 daily observations; see MEASURED CALIBRATION in the
# module docstring. A detector that fires on everything is as useless as one that fires on
# nothing, and the first kind is more dangerous because it looks diligent.
PENALTY_MULT = 1.0

# DETECT ON RAW RETURNS, NOT ON A ROLLING SHARPE. This was got wrong first time and the error
# is worth recording: overlapping 252-day windows induce ~252 lags of autocorrelation, so the
# rolling series wanders smoothly and looks piecewise-constant even when the underlying is
# IID. Measured false-positive rate on a STATIONARY series was 1.00 -- a break found every
# single time -- and it stayed at 0.72 even after raising the penalty 64x. Raw returns carry
# no induced dependence and separate cleanly.


def detect_breaks(series, n_breaks=None, penalty=None, model=DEFAULT_MODEL, min_size=60,
                  max_breaks=MAX_BREAKS):
    """Find structural break dates in a series without being told where to look.

    `series`   RAW returns. Not a rolling Sharpe -- see the note above PENALTY_MULT: the
               smoothing induces ~window lags of autocorrelation and drove the measured
               false-positive rate to 1.00 on a stationary series.
    `n_breaks` force exactly k breaks. Leave None to select k automatically.
    `penalty`  cost charged per breakpoint during selection. None -> PENALTY_MULT * var * ln n.
    `min_size` minimum segment length; stops it slicing off a fortnight and calling it a
               regime.

    Selection fits Binseg for k = 0..max_breaks and keeps the k minimising
    cost(k) + penalty*k. That is PELT's criterion evaluated over a small candidate set,
    which is all that is needed when a 15-year backtest plausibly holds 0-5 regimes.

    Returns a list of break dates.
    """
    import ruptures as rpt

    s = pd.Series(series).dropna()
    if len(s) < 3 * min_size:
        return []
    x = s.values.reshape(-1, 1).astype(float)

    if n_breaks:
        algo = rpt.Binseg(model=model, min_size=min_size).fit(x)
        idx = algo.predict(n_bkps=n_breaks)
        return [s.index[i] for i in idx if 0 < i < len(s)]

    pen = penalty if penalty is not None else PENALTY_MULT * float(np.var(x)) * np.log(len(s))
    algo = rpt.Binseg(model=model, min_size=min_size).fit(x)
    k_max = min(max_breaks, max(0, len(s) // min_size - 1))

    best_k, best_score, best_bkps = 0, None, [len(s)]
    for k in range(0, k_max + 1):
        try:
            bkps = algo.predict(n_bkps=k) if k else [len(s)]
        except Exception:
            break
        score = algo.cost.sum_of_costs(bkps) + pen * k
        if best_score is None or score < best_score:
            best_k, best_score, best_bkps = k, score, bkps

    # ruptures returns segment END points and always includes len(s); those are not breaks.
    return [s.index[i] for i in best_bkps if 0 < i < len(s)]


def chow_test(y, break_point, x=None, min_obs=30):
    """Chow test for a coefficient shift at `break_point`.

    With `x` omitted this tests a MEAN shift (regress on a constant): "did the average
    return change?"
    With `x` given -- pass the benchmark's returns -- it tests an ALPHA AND BETA shift:
    "did the strategy's relationship to the market change?" That second form is what the
    Failed Improvements note means by "a Chow test on the coefficient", and it is the one
    to use when asking whether a diversifier stopped diversifying.

    Returns dict(fstat, pvalue, df_num, df_den, n_before, n_after).
    A small p-value says the coefficients before and after are not the same numbers.
    """
    import statsmodels.api as sm

    y = pd.Series(y).dropna()
    if x is not None:
        x = pd.Series(x).reindex(y.index)
        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        y, X = df["y"], sm.add_constant(df[["x"]])
    else:
        X = pd.DataFrame({"const": np.ones(len(y))}, index=y.index)

    mask = y.index < break_point
    n1, n2 = int(mask.sum()), int((~mask).sum())
    k = X.shape[1]
    # `n > k + 1` alone is not enough: with a constant-only regression (k=1) that admits a
    # 3-observation "regime" and returns a confident p-value for it. Require a real sample
    # on BOTH sides, because a break near either end is where this fails quietly.
    if min(n1, n2) < max(k + 2, min_obs):
        return dict(fstat=np.nan, pvalue=np.nan, df_num=k, df_den=np.nan,
                    n_before=n1, n_after=n2,
                    note=f"too few observations on one side ({min(n1, n2)} < {max(k + 2, min_obs)})")

    rss_pool = sm.OLS(y, X).fit().ssr
    rss1 = sm.OLS(y[mask], X[mask]).fit().ssr
    rss2 = sm.OLS(y[~mask], X[~mask]).fit().ssr

    df_den = n1 + n2 - 2 * k
    denom = (rss1 + rss2) / df_den
    if denom <= 0:
        return dict(fstat=np.nan, pvalue=np.nan, df_num=k, df_den=df_den,
                    n_before=n1, n_after=n2)
    f = ((rss_pool - (rss1 + rss2)) / k) / denom
    from scipy import stats as st
    return dict(fstat=float(f), pvalue=float(1 - st.f.cdf(f, k, df_den)),
                df_num=k, df_den=int(df_den), n_before=n1, n_after=n2)


def regime_attribution(returns, break_dates, periods_per_year=252):
    """Per-regime Sharpe / CAGR / share of total return, split at `break_dates`.

    This is the table crypto_trend_test.py now always prints, generalised. Its purpose is to
    stop a full-sample number standing in for a claim about the future when the sample
    contains regimes that differ.
    """
    import stats_lib as sl

    r = pd.Series(returns).dropna()
    edges = [r.index[0]] + sorted(break_dates) + [r.index[-1]]
    rows = []
    total_growth = float((1 + r).prod())
    for a, b in zip(edges[:-1], edges[1:]):
        seg = r.loc[a:b]
        if len(seg) < 20:
            continue
        growth = float((1 + seg).prod())
        yrs = len(seg) / periods_per_year
        rows.append(dict(
            start=str(pd.Timestamp(a).date()), end=str(pd.Timestamp(b).date()),
            days=len(seg), years=round(yrs, 2),
            sharpe=round(sl.sharpe(seg, periods_per_year), 2),
            cagr=round(growth ** (1 / yrs) - 1, 4) if yrs > 0 else np.nan,
            total_return=round(growth - 1, 4),
            share_of_growth=round((growth - 1) / (total_growth - 1), 3)
            if abs(total_growth - 1) > 1e-9 else np.nan))
    return pd.DataFrame(rows)


def stability_report(returns, benchmark=None, rolling_window=252, periods_per_year=252):
    """The whole battery: detect breaks, test each formally, attribute per regime.

    Detection runs on the ROLLING Sharpe rather than raw returns -- a change in edge is a
    level shift there, and level shifts are what change-point detection is good at.

    Returns dict(breaks, tests, attribution, report) with `report` an ASCII block a lab can
    print straight into its output.
    """
    r = pd.Series(returns).dropna()
    lines = []
    if len(r) < 3 * periods_per_year:
        return dict(breaks=[], tests=[], attribution=pd.DataFrame(),
                    report="  (series too short for break detection)")

    # Raw returns, minimum one-year regimes. See the note on PENALTY_MULT for why this is
    # not run on a rolling Sharpe.
    brk = detect_breaks(r, min_size=periods_per_year)

    lines.append("  break detection on raw returns (dates found, not chosen):")
    if not brk:
        lines.append("    none found -- but see the power note: detection is reliable only")
        lines.append("    for LARGE shifts. This is not evidence the edge is stable.")
    else:
        lines.append(f"    {len(brk)} break(s): "
                     f"{', '.join(str(pd.Timestamp(b).date()) for b in brk)}")

    tests = []
    for b in brk:
        t = chow_test(r, b, x=benchmark)
        t["date"] = str(pd.Timestamp(b).date())
        tests.append(t)
        kind = "alpha/beta vs benchmark" if benchmark is not None else "mean return"
        if np.isfinite(t.get("pvalue", np.nan)):
            verdict = "SHIFT CONFIRMED" if t["pvalue"] < 0.05 else "not significant"
            lines.append(f"    Chow test at {t['date']} ({kind}): "
                         f"F={t['fstat']:.2f} p={t['pvalue']:.4f}  {verdict}")
        else:
            lines.append(f"    Chow test at {t['date']}: {t.get('note', 'not computable')}")

    att = regime_attribution(r, brk, periods_per_year)
    if not att.empty and len(att) > 1:
        lines.append("  per-regime attribution:")
        lines.append(f"    {'period':<26}{'yrs':>6}{'Sharpe':>8}{'CAGR':>9}"
                     f"{'share of growth':>17}")
        for _, row in att.iterrows():
            lines.append(f"    {row['start']} -> {row['end']}{row['years']:>6.1f}"
                         f"{row['sharpe']:>8.2f}{row['cagr']:>8.1%}"
                         f"{row['share_of_growth']:>17.1%}")
        top = att.loc[att["share_of_growth"].idxmax()]
        if float(top["share_of_growth"]) > 0.70:
            lines.append(f"  [!] {top['share_of_growth']:.0%} of all growth came from ONE "
                         f"regime ({top['start']} -> {top['end']}).")
            lines.append("      The full-sample number does not describe what to expect next.")

    return dict(breaks=brk, tests=tests, attribution=att, report="\n".join(lines))


if __name__ == "__main__":
    # Re-measure the detector's false/true-positive rates. Run this after ANY change to
    # PENALTY_MULT, DEFAULT_MODEL, or the choice of input series -- the numbers in the module
    # docstring are only worth quoting if they were produced by the code as it stands.
    import warnings
    warnings.filterwarnings("ignore")
    N, TRIALS = 2520, 30
    IDX = pd.date_range("2016-01-01", periods=N, freq="B")
    print(f"detect_breaks calibration: model={DEFAULT_MODEL} penalty_mult={PENALTY_MULT} "
          f"min_size=252, {TRIALS} trials of {N} daily observations")
    print(f"{'pre-Sharpe':>11}{'post-Sharpe':>12}{'detected':>10}{'med err (bdays)':>17}")
    for mu0, mu1 in [(0.0016, 0.0000), (0.0010, 0.0003), (0.0008, 0.0004),
                     (0.0006, 0.0005), (0.0005, 0.0005)]:
        hits, errs = 0, []
        for s in range(TRIALS):
            rs = np.random.RandomState(400 + s)
            y = pd.Series(np.r_[rs.normal(mu0, 0.01, N // 2),
                                rs.normal(mu1, 0.01, N - N // 2)], index=IDX)
            b = detect_breaks(y, min_size=252)
            if b:
                hits += 1
                errs.append(min(abs(y.index.get_loc(d) - N // 2) for d in b))
        tag = "  <- false positives" if mu0 == mu1 else ""
        print(f"{mu0 / 0.01 * np.sqrt(252):>11.2f}{mu1 / 0.01 * np.sqrt(252):>12.2f}"
              f"{hits / TRIALS:>10.2f}{(np.median(errs) if errs else float('nan')):>17.0f}{tag}")
    print("\nA null result from this detector means 'no LARGE break', never 'the edge is stable'.")
