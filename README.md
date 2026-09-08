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
| Expected max Sharpe from a 2.46M-pair search, **from luck alone** | **7.29** |

![Observed vs expected cointegrated pairs per window](reports/figures/fig1_noise_ratio.png)

After Benjamini–Hochberg FDR control, **1,446 pairs survive** — 0.83% of the naive count —
and **9 of 29 windows hold nothing tradable**. Traded out-of-sample the surviving book
returns Sharpe **−0.25**, against a pre-registered bar of ≥0.50.

**At 0 bps — free trading — the Sharpe is still −0.19.** The cost curve never crosses
zero. This edge is not destroyed by transaction costs; it is not there gross.

## Pre-registration

Three hypotheses were registered, frozen and **pushed to a remote before any code ran**.
`prereg.grade()` refuses to score a registration that is not pushed — a bar that exists
only on one disk can be rewritten after the result is known, which is not a bar.

| stage | question | frozen | verdict |
|---|---|---|---|
| **A** selection | does cointegration selection retain edge after correction? | `73543a53` | **FAIL 0/3** |
| **C** estimation | is a time-varying hedge ratio distinguishable from static OLS? | `ec59631b` | **FAIL 0/1** |
| **B** allocation | does convex allocation beat equal weighting? | `aa565e04` | **PASS 1/1**, and void |

Stage B's pass is void *by its own registration*, written before the number existed:
the equal-weight benchmark's Sharpe CI contains zero, so the optimiser improved on an
estimate of noise. That is covariance fitting, not skill.

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

## Three bugs, none found by code review

1. **25 tickers returned a different company's prices.** Symbols retired from the index
   and later reassigned — `FB`, `EMC`, `S` (Sprint → SentinelOne), `BEAM`, `SE` and 20
   more — come back from yfinance as the *new* company under the *old* symbol, with no
   error and no gap. Found by chasing a window that returned +72.7% on two pairs.
   *The first fix was worse than the bug:* an absolute price floor deleted **NVDA**, whose
   legitimate back-adjusted 2011 close is $0.36.
2. **The Kalman filter leaked the future.** Variance parameters were fit by MLE over
   formation + trading, so the states were causal *given* parameters that had seen the
   future. Perturbing only the last trading day moved β on earlier days by 3.8e-02.
3. **Two API assumptions that would have thrown** — `spa_test`'s return shape, and a
   Sharpe criterion applied to a zero-variance benchmark.

Both classes are now pinned by tests, in **both directions**: `test_lookahead.py` includes
a meta-test asserting the probe catches a deliberately non-causal estimator, because a
probe that passes everything is not a probe.

## Layout

```
formation.py         price panel, 29-window GGR schedule, ticker-reuse filter
selection.py         stage A: Engle-Granger + Benjamini-Hochberg
trade.py             book -> daily returns; costs, delisting, committed capital
hedge_ratio.py       stage C: three estimators, all causal
allocate.py          stage B: the convex program
grade_A.py           registered metrics, trial-Sharpe distribution, cost curve
grade_C.py           Model Confidence Set
test_lookahead.py    causality probe (+ meta-test)
test_data_quality.py ticker-reuse filter, pinned both ways
universe.py  stats_lib.py  prereg.py     vendored, unmodified
```

## Running it

```bash
pip install numpy pandas scipy statsmodels arch yfinance joblib cvxpy
python formation.py    # panel + coverage + data-quality audit
python selection.py    # ~20 min on 16 cores
python grade_A.py ; python grade_C.py ; python allocate.py
python test_lookahead.py ; python test_data_quality.py
```

## Honest limits

Point-in-time membership fixes **selection, not prices**. 190 of 829 union members
(22.9%) cannot be priced by yfinance at all; coverage is **77.1%**, and the loss is
**time-varying** — early windows lose more, so the study is better-powered after ~2018.
A clean fix needs CRSP or Norgate.

The result is a **non-detection, not a disproof**: the Sharpe CI is [−0.72, +0.30] and
contains zero. It is a statement about GGR-style cointegration selection on US large caps
2011–2026 — not about pairs trading in general.

The strategy is long/short and therefore untradable in a cash account. This is research.
No position in it was ever taken.

## Attribution

Point-in-time S&P 500 membership from [fja05680/sp500](https://github.com/fja05680/sp500)
(MIT), vendored at `data/sp500_ticker_start_end.csv`.
