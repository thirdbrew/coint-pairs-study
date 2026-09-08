# Cointegration Statistical Arbitrage Under Multiple-Testing Correction

A point-in-time study of pairs trading on the S&P 500, 2011–2026, with the selection
step treated as the multiple-testing problem it actually is.

**Full write-up: [`reports/paper.md`](reports/paper.md)**

---

## The finding in three numbers

| | |
|---|---|
| Pairs a naive cointegration screen calls "significant" | **173,594** |
| Pairs chance alone predicts at α=0.05 | **122,920** |
| Expected max Sharpe from a 2.46M-pair search, **from luck alone** | **1.33** |

![Observed vs expected cointegrated pairs per window](reports/figures/fig1_noise_ratio.png)

After Benjamini–Hochberg FDR control, **1,446 pairs survive** — 0.83% of the naive count —
and **9 of 29 windows hold nothing tradable**. Traded out-of-sample with a static hedge
ratio the surviving book returns Sharpe **−0.25**, against a pre-registered bar of ≥0.50.

**At zero cost — free trading — the Sharpe is still −0.19.** The cost curve never crosses
zero. This edge is not destroyed by transaction costs; it is not there gross.

## The number that frames the whole thing

After searching 2.46 million pairs, **validating a Sharpe of 0.50 would take 118 years of
data.** Not 118 years of this strategy — 118 years to distinguish it from the best of 2.46
million coin flips. The sample is 14.5.

| Bailey–López de Prado minimum backtest length, N=2,465,737 | years |
|---|---|
| to validate a Sharpe of 1.00 | 29.4 |
| to validate a Sharpe of 0.50 | 117.7 |
| **available** | **14.5** |

That is a property of the *search*, not of the sample — with a consequence worth stating
plainly: **testing fewer pairs would not improve the strategy. It would lower the bar the
strategy has to clear.**

And the null is characterised in both directions, because reporting one would be motivated
selection: the minimum detectable Sharpe here is **+0.654**, *above* the registered bar of
0.50 — so the sample was underpowered for its own bar. The upper 95% bound is **+0.296**,
which *excludes* 0.50 — so the null is informative anyway.

## Pre-registration

Three hypotheses were registered, frozen and **pushed to a remote before any code ran**.
`prereg.grade()` refuses to score a registration that is not pushed — a bar that exists
only on one disk can be rewritten after the result is known, which is not a bar.

| stage | question | frozen | verdict |
|---|---|---|---|
| **A** selection | does cointegration selection retain edge after correction? | `73543a53` | **FAIL 0/3** |
| **C** estimation | is a time-varying hedge ratio distinguishable from static OLS? | `ec59631b` | **PASS 1/1** |
| **B** allocation | does convex allocation beat equal weighting? | `aa565e04` | **PASS 1/1**, and void |

Stage B's pass is void *by its own registration*, written before the number existed: the
equal-weight benchmark's Sharpe CI contains zero, so the optimiser improved on an estimate
of noise. That is covariance fitting, not skill.

## What the three stages do

```
A  SELECTION      2,458,393 Engle-Granger tests across 29 formation windows
   statistics     -> Benjamini-Hochberg FDR at q=0.10
        |            how many "cointegrated" pairs are real vs. expected noise?
        v
C  ESTIMATION     static OLS vs 60-day rolling OLS vs Kalman random-walk beta
   econometrics   -> Model Confidence Set on Sharpe picks the surviving subset
        |            (the set is declared UP FRONT -- comparison, not hunt)
        v
B  ALLOCATION     max mu'w - (lambda/2) w'Sigma w
   optimization      s.t. ||w||_1 <= 1, |w_i| <= 0.25, ||w - w_prev||_1 <= 0.5
                   -> convex program (cvxpy/CLARABEL) vs equal weighting
```

Two multiplicities, two corrections, applied at the stage each belongs to: **FDR** on
selection, **Model Confidence Set** on model choice, **deflated Sharpe** on performance.

## Four bugs, none found by code review

An independent adversarial review of the finished study **reversed one of its published
conclusions**. That is recorded here rather than quietly patched, because it is the most
instructive part of the project.

1. **The trading path leaked the future** — and it flipped Stage C's registered verdict
   from FAIL to PASS. The z-score was normalised with the *mean of the trading-window
   beta*, so day 0's signal depended on day 125's prices. It hid because for static OLS
   that mean is a constant and nothing leaks: **Stage A was causally clean by luck of
   estimator choice, not by design.** And the causality probe tested the estimator
   *functions*, never the path built on them. Both are now fixed and probed end-to-end,
   with a meta-test at the path level.
2. **SR\* was a restatement of the trial horizon.** Trial Sharpes were measured on single
   pairs over ~125 days and applied to a 3,642-day book. Reported 7.29; correct 1.33.
3. **25 tickers returned a different company's prices** — `FB`, `EMC`, `S` (Sprint →
   SentinelOne), `BEAM`, `SE` and 20 more, retired symbols later reassigned. Found by
   chasing a window that returned +72.7% on two pairs. *The first fix was worse than the
   bug:* an absolute price floor deleted **NVDA**, whose back-adjusted 2011 close is $0.36.
4. **The Kalman filter's variance parameters saw the future**, caught by the
   estimator-level probe.

Both probe classes now pin **both directions** — a probe that passes everything is not a
probe.

## Layout

```
formation.py         price panel, 29-window GGR schedule, ticker-reuse filter
selection.py         stage A: Engle-Granger + Benjamini-Hochberg
trade.py             book -> daily returns; costs, delisting, committed capital
hedge_ratio.py       stage C: three estimators
allocate.py          stage B: the convex program, net of turnover
grade_A.py           registered metrics, trial distribution, cost curve
grade_C.py           Model Confidence Set
mechanism_C.py       the absorption table
figures.py           the four charts
test_lookahead.py    causality probes, estimator AND path level (+ meta-tests)
test_data_quality.py ticker-reuse filter, pinned both ways
universe.py  stats_lib.py  prereg.py     vendored, unmodified
```

## Running it

```bash
pip install numpy pandas scipy statsmodels arch yfinance joblib cvxpy
python formation.py
python selection.py      # ~40 min on 16 cores
python grade_A.py ; python grade_C.py ; python mechanism_C.py ; python allocate.py
python test_lookahead.py ; python test_data_quality.py
```

## Honest limits

Point-in-time membership fixes **selection, not prices**. 190 of 829 union members
(22.9%) cannot be priced; coverage is **77.1%**, and the loss is **time-varying** — early
windows lose more, so the study is better-powered after ~2018. A clean fix needs CRSP or
Norgate.

The result is a **non-detection, not a disproof**: the Sharpe CI is [−0.72, +0.30] and
contains zero. It is a statement about GGR-style cointegration selection on US large caps
2011–2026 — not about pairs trading in general.

Stage C's winner, rolling OLS, has its own CI touching zero, and sits on a book whose
selection stage failed its bar. The Model Confidence Set rules static OLS *out*; it does
not establish that anything is *in*.

The strategy is long/short and therefore untradable in a cash account. This is research.
No position in it was ever taken.

## Attribution

Point-in-time S&P 500 membership from [fja05680/sp500](https://github.com/fja05680/sp500)
(MIT), vendored at `data/sp500_ticker_start_end.csv`.
