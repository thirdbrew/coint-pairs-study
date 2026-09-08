"""
stats_lib -- the multiple-testing and confidence machinery this project has been
approximating by hand.
================================================================================

WHY THIS EXISTS
---------------
Trading/Failed Improvements.md records five principled attempts to beat the incumbent, each
judged against a pre-registered bar. The fifth PASSED its bar and was declined anyway, on
the strength of an attribution table showing the edge lived in one regime. The write-up
names what was missing:

    "The statistical reason these all failed -- selection under multiple testing, and why
     pre-registration is the retail-scale substitute for a deflated Sharpe."

Pre-registration is a good discipline and it should stay. But it is a substitute for
something that exists, is cheap, and is implemented here:

  * A Sharpe of 1.16 vs an incumbent 0.96 has an error bar. On ~10y of daily data the
    standard error on a Sharpe is roughly 0.3, so that gap is inside noise. `sharpe_ci`
    says so in one call instead of via a diagnostic table.
  * Sixteen lab scripts, each sweeping variants, is a search. `spa_test` prices that search:
    it answers "is the BEST of these genuinely better than the benchmark, given that I
    looked at all of them?"
  * `mcs` (Model Confidence Set) returns the set of strategies that cannot be told apart
    from the best. For the crypto slot it returns {incumbent, crypto} -- the same verdict
    the attribution table reached, as a number, before the diagnostics.
  * `deflated_sharpe` is the number the doctrine calls the real version of its trial-count
    clause.

CONVENTIONS THAT MATTER
-----------------------
* Everything takes RETURNS (higher = better). arch's tests are written in terms of LOSSES
  (lower = better) and the conversion is done inside. Getting that sign backwards silently
  inverts every conclusion, which is exactly why it is not left to the caller.
* Sharpe maths is done on PER-PERIOD returns and annualised only for display. The PSR/DSR
  formulas are defined on per-period Sharpe; feeding them an annualised one is a common and
  invisible error.
* Nothing here decides anything. These return numbers and intervals; the pre-registered bar
  still lives in the lab script, and is still written before the run.

References: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014); Hansen, "A Test for
Superior Predictive Ability" (2005); White, "A Reality Check for Data Snooping" (2000);
Hansen, Lunde & Nason, "The Model Confidence Set" (2011). Implementation via `arch`
(bashtage/arch), which ships all three multiple-comparison procedures.
"""

import numpy as np
import pandas as pd
from scipy import stats as _st

EULER = 0.5772156649015329
TRADING_DAYS = 252

# A constant return series has std() == 3.5e-18, not 0.0 -- floating-point residue, not zero.
# Testing `std == 0` therefore misses it, and mean/std comes back as ~2.9e15: a Sharpe of
# three quadrillion that no downstream check would question. Compare against a tolerance.
_EPS = 1e-12


# --------------------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------------------
def excess(returns, risk_free=0.0, periods_per_year=TRADING_DAYS):
    """Subtract a risk-free rate, returning a PER-PERIOD excess series.

    `risk_free` is either
      * a scalar ANNUALISED rate (0.015 == 1.5%/yr), divided by periods_per_year here, or
      * a per-period Series already in the same units as `returns` (e.g. swing_lib's
        daily_cash_rate applied to ^IRX), aligned on the index.

    A scalar is the convenient form and a realised series is the honest one: cash paid
    0.01%/yr in 2011 and 5.3%/yr in 2024, and a flat average smears that across the window.
    """
    r = pd.Series(returns).astype(float)
    if isinstance(risk_free, (pd.Series, pd.DataFrame, np.ndarray, list)):
        rf = pd.Series(risk_free).astype(float).reindex(r.index)
        if rf.isna().any():
            raise ValueError("risk_free series does not cover the return index "
                             "(%d gaps)" % int(rf.isna().sum()))
        return r - rf
    return r - float(risk_free) / periods_per_year


def sharpe(returns, periods_per_year=TRADING_DAYS, annualised=True, risk_free=0.0):
    """Sharpe ratio of `returns` in excess of `risk_free`.

    READ THE DEFAULT. `risk_free=0.0` is excess-of-ZERO, which is what every other lab in
    this repo computes and is kept as the default so nothing silently changes underneath
    them. It is a return-to-vol ratio, NOT a Sharpe ratio against cash, and the difference
    is not cosmetic when the series being compared run at different volatilities: the
    omitted rf drag is proportional to vol, so excess-of-zero systematically flatters the
    low-vol series. On 2007-2026, cash (^IRX) averaged ~1.46%/yr; that is 0.18 of Sharpe on
    an 8%-vol mix and 0.07 on a 20%-vol one, i.e. an 0.11 handicap handed to the bond-heavy
    end of any slate purely by the choice of default.

    Pass `risk_free` -- ideally the realised cash series, not a flat guess -- whenever the
    number is going to be compared ACROSS series with different vol.
    """
    r = excess(pd.Series(returns).dropna(), risk_free, periods_per_year).dropna()
    if len(r) < 2 or r.std() < _EPS:
        return np.nan
    s = r.mean() / r.std()
    return s * np.sqrt(periods_per_year) if annualised else s


def sharpe_stderr(returns, periods_per_year=TRADING_DAYS):
    """Analytic SE of the annualised Sharpe, adjusted for skew and kurtosis (Lo 2002).

    Useful as a sanity check on the bootstrap: if the two disagree badly, the return series
    has structure (autocorrelation, regime) that the analytic formula does not see -- and
    the bootstrap is the one to trust.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 10 or r.std() < _EPS:
        return np.nan
    s = r.mean() / r.std()                       # per-period
    g3, g4 = _st.skew(r), _st.kurtosis(r, fisher=False)
    var = (1 + 0.5 * s ** 2 - g3 * s + (g4 - 3) / 4 * s ** 2) / n
    return np.sqrt(max(var, 0)) * np.sqrt(periods_per_year)


def sharpe_ci(returns, periods_per_year=TRADING_DAYS, reps=1000, block_size=None,
              alpha=0.05, seed=42):
    """Block-bootstrap confidence interval for the annualised Sharpe.

    Uses the STATIONARY bootstrap: daily returns are autocorrelated and vol-clustered, so an
    IID bootstrap would resample away the very dependence that makes a Sharpe uncertain and
    hand back an interval that is too narrow -- flattering, and wrong in the dangerous
    direction.

    Returns (point, low, high). If the incumbent's point estimate sits inside a candidate's
    interval, the candidate has not been shown to be better, whatever the bar said.
    """
    from arch.bootstrap import StationaryBootstrap
    r = pd.Series(returns).dropna()
    if len(r) < 30:
        return (np.nan, np.nan, np.nan)
    if block_size is None:
        # Politis-White rule of thumb; ~n^(1/3) is the usual default and is plenty here.
        block_size = max(2, int(round(len(r) ** (1 / 3))))
    bs = StationaryBootstrap(block_size, r.values, seed=seed)

    def _f(x):
        x = np.asarray(x).ravel()
        sd = x.std(ddof=1)
        return np.array([x.mean() / sd * np.sqrt(periods_per_year) if sd > _EPS else 0.0])

    ci = bs.conf_int(_f, reps=reps, method="basic", size=1 - alpha)
    return (sharpe(r, periods_per_year), float(ci[0, 0]), float(ci[1, 0]))


# --------------------------------------------------------------------------------------
# multiple testing across a search
# --------------------------------------------------------------------------------------
def _losses(returns_df):
    """arch's comparison tests are written on losses (lower = better). Returns invert."""
    return -pd.DataFrame(returns_df).dropna()


CRITERIA = ("mean", "sharpe")


def sharpe_losses(returns_df, risk_free=0.0, periods_per_year=TRADING_DAYS):
    """Per-period losses whose MEAN is the negative annualised Sharpe, up to sqrt(T).

    WHY THIS EXISTS. arch's SPA/MCS/StepM/RealityCheck all rank models by EXPECTED LOSS.
    Hand them loss = -return and they compare MEAN RETURN -- a perfectly well-formed answer
    to a question nobody asked when the hypothesis was written on Sharpe. The two are not
    monotone in each other: on simple_vs_policy_2026_08 the highest-Sharpe mix is the
    LOWEST-CAGR series in the pool, so a mean-return test cannot see the effect at all.

    The fix is a scale, not a new procedure. For series i define

        loss_i(t) = -(r_i(t) - rf(t)) / sd_i        sd_i = sample std of the excess series

    Then mean(-loss_i) * sqrt(periods_per_year) == the annualised Sharpe of series i,
    exactly (test_sharpe_losses_reproduce_the_sharpe pins it to 1e-12). Every model is
    divided by its OWN sd, so a mean-loss comparison across models is a Sharpe comparison.

    THE ONE THING THIS DOES NOT DO. sd_i is estimated once on the full sample and treated as
    known inside the bootstrap, so the resulting p-values price the uncertainty in the MEAN
    of the scaled series and not the uncertainty in the scaling itself. That is the standard
    device and it is mildly anti-conservative; it is not a substitute for a bootstrap that
    re-estimates sd inside every replicate.
    """
    df = pd.DataFrame(returns_df).dropna()
    ex = df.apply(lambda c: excess(c, risk_free, periods_per_year))
    sd = ex.std(ddof=1)
    if (sd < _EPS).any():
        bad = list(sd.index[sd < _EPS])
        raise ValueError("zero-variance series cannot be Sharpe-scaled: %s" % bad)
    return -(ex / sd)


def _criterion_losses(returns_df, criterion, risk_free, periods_per_year=TRADING_DAYS):
    """Dispatch to the loss function the CALLER named. No silent default of substance."""
    if criterion not in CRITERIA:
        raise ValueError("criterion must be one of %r, got %r" % (CRITERIA, criterion))
    if criterion == "mean":
        if not np.isscalar(risk_free) or float(risk_free) != 0.0:
            # A constant subtracted from every series shifts all mean losses equally and
            # cannot change a mean-return ranking, so silently accepting rf here would
            # imply an adjustment that does not happen.
            return _losses(pd.DataFrame(returns_df).apply(
                lambda c: excess(c, risk_free, periods_per_year)))
        return _losses(returns_df)
    return sharpe_losses(returns_df, risk_free=risk_free,
                         periods_per_year=periods_per_year)


def spa_test(benchmark_returns, candidate_returns, reps=1000, block_size=10, seed=42,
             criterion="mean", risk_free=0.0, periods_per_year=TRADING_DAYS):
    """Hansen's Superior Predictive Ability test -- prices the whole search at once.

    H0: NO candidate beats the benchmark once you account for having looked at all of them.

    This is the formal version of the trial-count clause in Failed Improvements. Feed it
    every variant that was ever tried, not just the survivor -- the correction is only valid
    if the count is honest, which is precisely the discipline pre-registration was standing
    in for.

    `criterion` NAMES WHAT "BEATS" MEANS and it is not a detail:
      "mean"   loss = -excess return   -> superiority in MEAN RETURN (the historical default)
      "sharpe" loss = -excess/sd       -> superiority in SHARPE, via sharpe_losses()
    A hypothesis written on Sharpe and tested at criterion="mean" gets a well-formed p-value
    for the wrong question, and the two can point opposite ways. The choice is echoed back
    in the result dict so an artifact records which one was run.

    Returns a dict with arch's three p-values:
      lower      most liberal  (assumes all candidates are relevant)
      consistent the one to quote
      upper      most conservative
    A `consistent` p-value above 0.10 means: the best thing you found is what a search this
    wide produces from noise.
    """
    from arch.bootstrap import SPA
    bench = pd.Series(benchmark_returns).dropna()
    cand = pd.DataFrame(candidate_returns).dropna()
    idx = bench.index.intersection(cand.index)
    bench, cand = bench.loc[idx], cand.loc[idx]
    # Scale benchmark and candidates TOGETHER so each column is divided by its own sd once,
    # on the same aligned rows. Scaling them in separate calls would be identical here but
    # is one refactor away from not being.
    both = _criterion_losses(pd.concat([bench.rename("__bench__"), cand], axis=1),
                             criterion, risk_free, periods_per_year)
    spa = SPA(both["__bench__"].values, both.drop(columns=["__bench__"]).values,
              reps=reps, block_size=block_size, seed=seed)
    spa.compute()
    p = spa.pvalues
    return dict(pvalues={k: float(p[k]) for k in p.index},
                consistent=float(p["consistent"]),
                criterion=criterion, reps=int(reps), block_size=int(block_size),
                seed=int(seed),
                n_candidates=cand.shape[1], n_obs=len(idx),
                verdict=("no candidate beats the benchmark once the search is priced in"
                         if float(p["consistent"]) > 0.10 else
                         "at least one candidate survives the multiple-testing correction"))


def reality_check(benchmark_returns, candidate_returns, reps=1000, block_size=10, seed=42,
                  criterion="mean", risk_free=0.0, periods_per_year=TRADING_DAYS):
    """White's Reality Check -- SPA's predecessor, kept for cross-checking.

    SPA is strictly better (it drops poor candidates from the null rather than letting them
    drag the critical value around), so quote SPA. If the two disagree wildly, the cause is
    usually a few very bad candidates in the set, and that is worth knowing.

    `criterion` is as in spa_test.
    """
    from arch.bootstrap import RealityCheck
    bench = pd.Series(benchmark_returns).dropna()
    cand = pd.DataFrame(candidate_returns).dropna()
    idx = bench.index.intersection(cand.index)
    both = _criterion_losses(pd.concat([bench.loc[idx].rename("__bench__"),
                                        cand.loc[idx]], axis=1),
                             criterion, risk_free, periods_per_year)
    rc = RealityCheck(both["__bench__"].values, both.drop(columns=["__bench__"]).values,
                      reps=reps, block_size=block_size, seed=seed)
    rc.compute()
    return float(rc.pvalues["consistent"])


def superior_models(benchmark_returns, candidate_returns, size=0.05, reps=1000,
                    block_size=10, seed=42, criterion="mean", risk_free=0.0,
                    periods_per_year=TRADING_DAYS):
    """StepM: WHICH candidates beat the benchmark, controlling the family-wise error rate.

    SPA answers "did anything win?"; this answers "name them", without the false-discovery
    inflation you get from running k separate t-tests and keeping the winners.

    `criterion` is as in spa_test.
    """
    from arch.bootstrap import StepM
    bench = pd.Series(benchmark_returns).dropna()
    cand = pd.DataFrame(candidate_returns).dropna()
    idx = bench.index.intersection(cand.index)
    both = _criterion_losses(pd.concat([bench.loc[idx].rename("__bench__"),
                                        cand.loc[idx]], axis=1),
                             criterion, risk_free, periods_per_year)
    sm = StepM(both["__bench__"].values, both.drop(columns=["__bench__"]),
               size=size, reps=reps, block_size=block_size, seed=seed)
    sm.compute()
    return list(sm.superior_models)


def model_confidence_set(returns_df, size=0.05, reps=1000, block_size=10, seed=42,
                         criterion="mean", risk_free=0.0,
                         periods_per_year=TRADING_DAYS):
    """Model Confidence Set: the strategies that cannot be distinguished from the best.

    The most useful of the three here, because it matches how this project actually decides.
    A candidate that merely lands in the same confidence set as the incumbent has not earned
    a switch -- which is the crypto-slot verdict, reached without needing the regime table.

    `criterion` NAMES WHAT "BEST" MEANS -- see spa_test. arch's MCS eliminates by EXPECTED
    LOSS, so criterion="mean" builds the set on mean return and criterion="sharpe" builds it
    on Sharpe. A registration that says "MCS on Sharpe" is not satisfied by the default.

    THE MEMBERSHIP BINARY IS A MONTE CARLO ESTIMATE, NOT AN ANALYTIC ONE. The MCS p-values
    come out of a block bootstrap, so a p sitting near `size` flips in and out of the set
    with the seed. At reps=1000 the Monte Carlo standard error on a p near 0.10 is ~0.010,
    which is the whole distance between 0.09 and 0.11: a decision taken there is a coin
    flip dressed as a result. `mc_stderr` sizes it, and reps needs to be large enough that
    the error is small against the gap to `size` -- 20000 is cheap (~3s on 9 series x 4839
    bars) and gets the error to ~0.002.

    Returns dict(included, excluded, pvalues, criterion, reps, block_size, seed).
    """
    from arch.bootstrap import MCS
    losses = _criterion_losses(returns_df, criterion, risk_free, periods_per_year)
    mcs = MCS(losses, size=size, reps=reps, block_size=block_size, seed=seed)
    mcs.compute()
    return dict(included=sorted(mcs.included), excluded=sorted(mcs.excluded),
                pvalues={k: float(v) for k, v in mcs.pvalues.iloc[:, 0].items()},
                criterion=criterion, size=float(size), reps=int(reps),
                block_size=int(block_size), seed=int(seed))


def block_bootstrap_stat(func, data, reps=2000, block_size=10, seed=42, alpha=0.05):
    """Stationary-bootstrap sampling distribution of ANY statistic of a return panel.

    `func(array)` takes an (n_obs, n_cols) ndarray of one resampled panel and returns one
    float. ROWS are resampled jointly, so the cross-sectional correlation between columns is
    preserved and a paired statistic (A minus B on the same days) stays paired.

    Exists because point estimates with no interval keep appearing next to bootstrapped
    p-values in the same table, which reads as though both carry uncertainty. Max drawdown
    is the worst offender: it is an ORDER STATISTIC OF ONE REALISED PATH, usually decided by
    a single episode, and it is the least stable number in a metrics block.

    CHOOSE block_size FOR THE STATISTIC, NOT OUT OF HABIT. A mean needs only enough block to
    carry the autocorrelation; a DRAWDOWN is a path statistic, and blocks of 10 days cannot
    reassemble a 14-month 2008-style decline, so a short block understates deep drawdowns
    and biases the resampled distribution toward zero. Run it at more than one block length
    and report both rather than picking the flattering one.

    Returns dict(point, mean, lo, hi, prob_le_zero, reps, block_size, seed).
    """
    from arch.bootstrap import StationaryBootstrap
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows = np.arange(arr.shape[0])
    bs = StationaryBootstrap(block_size, rows, seed=seed)
    draws = np.empty(int(reps), dtype=float)
    for k, (pos, _kw) in enumerate(bs.bootstrap(int(reps))):
        draws[k] = float(func(arr[np.asarray(pos[0]).ravel()]))
    point = float(func(arr))
    return dict(point=point, mean=float(draws.mean()),
                lo=float(np.percentile(draws, 100 * alpha / 2.0)),
                hi=float(np.percentile(draws, 100 * (1.0 - alpha / 2.0))),
                prob_le_zero=float((draws <= 0.0).mean()),
                reps=int(reps), block_size=int(block_size), seed=int(seed))


def mc_stderr(p, reps):
    """Monte Carlo standard error of a bootstrap p-value estimated from `reps` draws.

    sqrt(p(1-p)/reps). Quote it next to any p that is being compared against a fixed bar:
    if |p - bar| is not several times this, the verdict is simulation noise and the honest
    move is more reps, not a decision.
    """
    p = float(p)
    if reps <= 0 or not 0.0 <= p <= 1.0:
        return np.nan
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / reps))


# --------------------------------------------------------------------------------------
# deflating a Sharpe for the size of the search
# --------------------------------------------------------------------------------------
def probabilistic_sharpe(returns, benchmark_sr=0.0, periods_per_year=TRADING_DAYS):
    """PSR: P(true Sharpe > benchmark_sr), correcting for skew, fat tails and sample length.

    `benchmark_sr` is annualised for convenience; it is converted internally.
    A PSR below ~0.95 means the observed Sharpe is not distinguishable from the benchmark's.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 10 or r.std() < _EPS:
        return np.nan
    sr = r.mean() / r.std()                              # per-period
    sr_star = benchmark_sr / np.sqrt(periods_per_year)   # match units
    g3, g4 = _st.skew(r), _st.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(_st.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials, sharpe_variance):
    """E[max Sharpe] under the null that every trial has zero true edge.

    This is the bar a winner must clear to be interesting: the best of N coin-flips is not
    zero, and the more variants a lab sweeps the higher it climbs. Per-period units.
    """
    if n_trials < 2:
        return 0.0
    v = np.sqrt(max(sharpe_variance, 0))
    a = _st.norm.ppf(1 - 1.0 / n_trials)
    b = _st.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(v * ((1 - EULER) * a + EULER * b))


def deflated_sharpe(returns, n_trials, sharpe_variance=None, all_trial_sharpes=None,
                    periods_per_year=TRADING_DAYS):
    """DSR: PSR measured against the Sharpe a search of this size produces from luck alone.

    Pass EITHER `sharpe_variance` (variance of the annualised Sharpes across all trials) or
    `all_trial_sharpes` (the Sharpes themselves, annualised) and it will compute it.

    `n_trials` must be the honest count of everything tried -- every variant in every sweep,
    including the ones abandoned before they were written up. Understating it is the same
    error as not correcting at all, only harder to notice later.

    A DSR below 0.95 is the formal version of "a lone marginal pass ships nothing".
    """
    if sharpe_variance is None:
        if all_trial_sharpes is None:
            raise ValueError("pass sharpe_variance or all_trial_sharpes")
        ann = np.asarray(all_trial_sharpes, dtype=float)
        ann = ann[np.isfinite(ann)]
        if len(ann) < 2:
            raise ValueError("need >= 2 trial Sharpes to estimate their variance")
        sharpe_variance = float(np.var(ann, ddof=1)) / periods_per_year   # -> per-period
    else:
        sharpe_variance = float(sharpe_variance) / periods_per_year
    sr_star = expected_max_sharpe(n_trials, sharpe_variance)     # per-period
    return probabilistic_sharpe(returns, benchmark_sr=sr_star * np.sqrt(periods_per_year),
                                periods_per_year=periods_per_year)


def min_backtest_length(n_trials, target_sharpe=1.0, periods_per_year=TRADING_DAYS):
    """Roughly how many YEARS of data a search of N trials needs before a Sharpe means much.

    MinBTL ~ 2*ln(N) / SR^2. It is an approximation, and its job is to be sobering: at 50
    trials and a target Sharpe of 1.0 you need ~8 years before the winner is distinguishable
    from the best of fifty coin flips.
    """
    if n_trials < 2 or target_sharpe <= 0:
        return np.nan
    return float(2.0 * np.log(n_trials) / (target_sharpe ** 2))


# --------------------------------------------------------------------------------------
# a single report a lab can print
# --------------------------------------------------------------------------------------
def compare(incumbent, candidate, n_trials=None, labels=("incumbent", "candidate"),
            periods_per_year=TRADING_DAYS, reps=1000):
    """The standard read: two return series, with error bars and a multiple-testing price.

    Prints ASCII only (the PC console is cp1252) and returns the numbers as a dict so a lab
    can assert on them against its pre-registered bar.
    """
    a, b = pd.Series(incumbent).dropna(), pd.Series(candidate).dropna()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    sa, la, ha = sharpe_ci(a, periods_per_year, reps=reps)
    sb, lb, hb = sharpe_ci(b, periods_per_year, reps=reps)
    overlap = not (lb > ha or la > hb)
    out = dict(n_obs=len(idx),
               incumbent=dict(sharpe=sa, lo=la, hi=ha),
               candidate=dict(sharpe=sb, lo=lb, hi=hb),
               edge=sb - sa, intervals_overlap=bool(overlap))
    lines = [
        f"  {labels[0]:<14} Sharpe {sa:5.2f}   95% CI [{la:5.2f}, {ha:5.2f}]",
        f"  {labels[1]:<14} Sharpe {sb:5.2f}   95% CI [{lb:5.2f}, {hb:5.2f}]",
        f"  edge {sb - sa:+.2f}",
    ]
    if overlap:
        lines.append("  [!] the intervals OVERLAP -- this edge is not distinguishable from noise")
    if n_trials:
        dsr = deflated_sharpe(b, n_trials, all_trial_sharpes=[sa, sb],
                              periods_per_year=periods_per_year)
        mb = min_backtest_length(n_trials, target_sharpe=max(sb, 0.1))
        out["deflated_sharpe"] = dsr
        out["min_backtest_years"] = mb
        lines.append(f"  deflated Sharpe (N={n_trials} trials): {dsr:.3f}"
                     f"   {'PASS' if dsr and dsr > 0.95 else 'below the 0.95 bar'}")
        lines.append(f"  a search this wide needs ~{mb:.1f}y of data; you have "
                     f"{len(idx) / periods_per_year:.1f}y")
    out["report"] = "\n".join(lines)
    return out
