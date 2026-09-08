# Cointegration Statistical Arbitrage Under Multiple-Testing Correction

**A point-in-time study of the S&P 500, 2011–2026.**

---

## Summary

A naive cointegration screen over the S&P 500 identifies **173,594 "significant" pairs**.
Under the null of no cointegration anywhere, chance alone predicts **122,920** of them.

That ratio — **1.41x** — is the entire finding. After Benjamini–Hochberg FDR control,
**1,446 pairs survive**, 0.83% of the naive count, and **9 of 29 windows hold nothing
tradable at all**. Traded out-of-sample under the Gatev–Goetzmann–Rouwenhorst protocol,
the surviving book returns an annualised Sharpe of **−0.25** against a pre-registered
bar of ≥0.50.

The number that puts it in perspective: a search of **2,465,737 pairs** has an expected
maximum Sharpe of **7.29 from luck alone**.

Three hypotheses were pre-registered, frozen and pushed to a remote before any code ran.
Two failed. The one that passed was pre-registered as meaningless if it did.

| stage | question | verdict |
|---|---|---|
| **A** — selection | does cointegration selection retain edge after correction? | **FAIL 0/3** |
| **C** — estimation | is a time-varying hedge ratio distinguishable from static OLS? | **FAIL 0/1** |
| **B** — allocation | does convex allocation beat equal weighting? | **PASS 1/1**, and void |

---

## 1. The question

Pairs trading is the most-replicated strategy in retail quantitative finance, and the
selection step is a multiple-testing problem that is almost never treated as one. Testing
every pair in the index means running >100,000 simultaneous hypothesis tests per
formation window. At α=0.05, tens of thousands of "cointegrated" pairs are guaranteed by
construction.

The question here is not whether pairs trading works. It is **what is left when you
account for having looked 2.46 million times.**

**Registered prior, written before the run:** Rad, Low & Faff (2016) find both distance
and cointegration pair selection decayed sharply post-2009. The expected result was
little to no edge. This study was built so that outcome would be a measurement rather
than a disappointment.

## 2. Data and protocol

| | |
|---|---|
| **Universe** | Point-in-time S&P 500 membership. 829 tickers ever a member, 2011–2026 |
| **Protocol** | Gatev–Goetzmann–Rouwenhorst: 12-month formation, 6-month trading, non-overlapping, rolled 6 months. **29 windows**, 2012-01 → 2026-06 |
| **Selection** | Engle–Granger on log prices, every eligible pair, formation window only |
| **Correction** | Benjamini–Hochberg FDR at q=0.10; book capped at top 20 by test statistic |
| **Trading** | Enter \|z\|>2, exit \|z\|<0.5, force-close at window end. **Thresholds fixed at GGR values, never tuned** |
| **Construction** | Dollar-neutral, hedge-ratio normalised by (1+\|β\|), return on committed capital |
| **Costs** | 10 bps round-trip base case, swept 0–50 bps |
| **Benchmark** | Zero-return series. The book is market-neutral; an SPY criterion would be blind to what it is |

Point-in-time membership matters more than it looks. Today's index has 503 members;
**829 were members at some point in the window.** Screening 15 years of history against
today's list pre-filters the universe to companies that survived — the classic
survivorship bias, and it flatters every result.

Thresholds are not tuned because tuning them is a further search that would have to enter
`n_trials`. The protocol is used exactly as published.

## 3. Stage A — selection

**Bar frozen and pushed before the run** (`73543a53`): `sharpe_net ≥ 0.50`,
`dsr ≥ 0.95`, `spa_p ≤ 0.05`, at `n_trials = 2,465,737`.

### 3.1 How much of the screen is noise

| | count |
|---|---|
| Engle–Granger tests | **2,458,393** |
| "Significant" at α=0.05 | **173,594** |
| Expected from chance alone | **122,920** |
| Ratio | **1.41x** |
| Surviving Benjamini–Hochberg at q=0.10 | **1,446** (0.83% of naive) |
| Windows with an empty book | **9 of 29** |
| Windows finding *fewer* than chance predicts | **4** |

The median window runs at 1.35x noise. Four windows come in **below** 1.0 — the screen
found fewer cointegrated pairs than random data would have produced.

FDR was chosen over Bonferroni deliberately. Bonferroni controls the probability of even
one false positive, which at this scale means a per-test threshold near 4e-7 and an
essentially empty book. FDR controls the expected *proportion* of the selected set that
is false, which is the question actually being asked: of the pairs I chose to trade, what
share are noise?

### 3.2 What the surviving book earns

```
cost (bps round-trip)     annualised Sharpe     total return
        0                     -0.1851             -19.76%
        2                     -0.1980             -21.14%
        5                     -0.2173             -23.20%
       10                     -0.2496             -26.64%
       20                     -0.3140             -33.51%
       50                     -0.5059             -54.14%
```

**Breakeven cost: −∞. The curve never crosses zero.**

This is the load-bearing detail. A common and comfortable finding is that a strategy is
real gross and destroyed by transaction costs. **This one is not there gross.** At zero
cost — free trading, infinite liquidity — the Sharpe is still −0.1851 over 3,642 trading
days.

### 3.3 The grade

```
PRE-REGISTERED RESULT -- coint_A_selection
  bar is PUSHED: on origin/master as 73543a53

  metric                bar      observed   verdict
  sharpe_net         >= 0.5      -0.2496    FAIL
  dsr                >= 0.95      0.0000    FAIL
  spa_p              <= 0.05      0.8294    FAIL

  VERDICT: FAIL (0/3 criteria met)
```

**SR\* = 7.288.** That is the annualised Sharpe a search of 2,465,737 trials produces
from luck alone, computed from the trial-Sharpe distribution of 1,694 randomly drawn
pairs traded under identical rules. The random sample matters: estimating trial variance
from the *selected* pairs would be circular, since they were chosen for looking good.

Against SR\* = 7.29, the deflated Sharpe probability is 1.4e-179.

Hansen SPA returns p = 0.8294 against a zero-return benchmark, with a Monte Carlo
standard error of 0.0027 — that p is precise, not simulation noise.

**Stated honestly:** the 95% CI on Sharpe is **[−0.72, +0.30] and contains zero.** The
correct claim is *no edge detected, point estimate negative* — not *it loses money*. This
study cannot distinguish the strategy from zero; it can rule out the bar.

## 4. Stage C — estimation

**Bar frozen before the run** (`ec59631b`): static OLS lands in the MCS *excluded* set.

Three hedge-ratio estimators, declared as a set up front and compared by Model Confidence
Set on Sharpe (`criterion="sharpe"`, `reps=20000`, `size=0.05`).

Declaring the set in advance is what makes this a **model comparison** rather than a
mechanism hunt. Running three estimators and reporting the winner is the latter.

```
estimator      Sharpe    95% CI                 MCS p     MC stderr   verdict
static_ols    -0.2496   [-0.7229, +0.2958]     1.0000     0.0000     INCLUDED
rolling_ols   -0.5136   [-0.9884, -0.0013]     0.3456     0.0034     INCLUDED
kalman        -1.1279   [-1.6213, -0.6342]     0.0087     0.0007     EXCLUDED
```

**The Model Confidence Set eliminates the Kalman filter** — the most sophisticated
estimator — at p=0.0087. Performance degrades monotonically with estimator
sophistication. The registered claim (that static OLS would be excluded) fails 0/1;
the substantive result is its mirror image.

### 4.1 A mechanism, tested rather than asserted

The natural explanation is that an adaptive β *tracks* the spread, absorbing the very
divergence the strategy exists to trade. That has a testable implication: adaptive
estimators should spend less time beyond the entry threshold.

```
estimator      mean |z|   days |z|>2   sd(beta_t)
static_ols       2.547       50.3%       0.0000
rolling_ols      1.545       28.5%       0.4505
kalman           1.699       31.3%       0.0235
```

**Absorption is confirmed** — adaptive β cuts time beyond the entry threshold from 50%
to ~30%.

**It does not explain the ranking.** Kalman varies its β *least* of the two adaptive
estimators (sd 0.0235 vs 0.4505) and still performs worst by a wide margin. The tidy
story accounts for adaptive-vs-static, not for Kalman-vs-rolling. **That remains
unexplained and is recorded as open** rather than furnished with a second story.

## 5. Stage B — allocation

**Bar frozen before the run** (`aa565e04`): `sharpe_edge ≥ 0.10`.

The program, stated before the code:

```
decision variables   w ∈ R^k              signed dollar weight per pair
objective            max  μ'w − (λ/2)·w'Σw
subject to           ‖w‖₁          ≤ 1.0     gross exposure budget
                     |w_i|         ≤ 0.25    per-pair concentration cap
                     ‖w − w_prev‖₁ ≤ 0.50    turnover trust region
```

A quadratic program with an L1 ball, an L∞ box and an L1 trust region — convex, so the
solver returns a global optimum. Re-solved every 21 trading days on 60-day trailing
estimates of μ and Σ, using only data strictly before the rebalance date. Solved with
cvxpy/CLARABEL. Benchmarked against equal weighting **at the same gross budget**, so the
comparison cannot quietly become a leverage comparison.

```
equal weight Sharpe  -0.2496
optimised Sharpe     -0.0599
sharpe_edge          +0.1896     bar >= 0.10     PASS 1/1
```

### 5.1 The pass is void, and was pre-registered as void

The registration for stage B, written before the run, says:

> *"A PASS here on a no-edge book would be evidence of overfitting the covariance, not
> evidence of skill, and must be reported that way if it happens."*

It happened. The equal-weight benchmark's own Sharpe CI is [−0.723, +0.296] and contains
zero, so μ is an estimate of noise. An optimiser that improves Sharpe by allocating
precisely across noise has fitted the covariance matrix, not found skill. `allocate.py`
carries a guard that refuses to report the improvement as an edge and prints exactly that.

**This is the most useful result in the study.** It is the case where a numerical pass
means nothing, and the only thing preventing it from being reported as a +0.19 Sharpe
improvement is a sentence written before the number existed.

## 6. What broke, and how it was caught

Three defects. **None was found by code review.** All three would have produced
confident, wrong numbers.

### 6.1 Twenty-five tickers returned a different company's prices

Found by chasing an outlier: one window returned **+72.7% on a two-pair book**. The cause
was `PARA`, whose series splices a ~101,500 foreign listing onto Paramount's $3, fitting
β=6.470 and α=−70.132.

The general case is worse than the instance. **25 symbols were retired from the index and
later reassigned to a different company**, and yfinance serves the new company's prices
under the old symbol with no error and no gap:

| symbol | index membership | prices actually returned |
|---|---|---|
| `FB` | 2013-12 → 2022-06 (left as META) | 2025-06 onward |
| `EMC` | 1996-03 → 2016-09 (Dell) | 2023-05 onward |
| `S` | 1996-01 → 2013-07 (Sprint) | 2021-06 onward (SentinelOne) |
| `BEAM` | 1996-01 → 2014-05 (spirits) | 2020-02 onward (Beam Therapeutics) |
| `SE` | 2007-01 → 2017-02 (Spectra Energy) | 2017-10 onward (Sea Limited) |

plus `STI` `SHLD` `SPLS` `APC` `CAM` `NE` `MMI` `NSM` `INFO` `TE` `POM` `ADT` `LB` `LIFE`
`PCL` `MHS` `MI` `SBNY`, and two corrupted stubs.

So the 190 unpriceable delisted names are only the **visible** half of the survivorship
problem. **Missing data announces itself. Wrong data does not.**

**The first fix was worse than the bug.** It used an absolute price floor and rejected
**NVDA** — `auto_adjust` back-adjusts splits, so NVIDIA's legitimate 2011 close in this
panel is **$0.36** — along with AIV, FSLR and SEDG. An absolute floor on back-adjusted
history cannot distinguish a real company from a corrupted stub.

Replaced with two tests, neither an arbitrary threshold:

1. **Membership overlap** — the series must carry prices during a spell when the ticker
   was actually in the index.
2. **Terminal price** — back-adjustment scales history down and leaves the last price
   alone, so the endpoint is honest. NVDA ends at $194.97 and passes; `COL` ends at $0.05
   and `CPWR` at $0.01, and they do not.

`test_data_quality.py` pins **both directions**: a filter is only as good as the false
positives it does not produce.

**Note the direction.** The contaminated pair *made* money. Removing it made every number
in this study worse. The filter is specified on data provenance alone and was written
before any graded metric was read.

### 6.2 The Kalman filter leaked future information

The first `kalman_beta` estimated its variance parameters by maximum likelihood over
formation **and** trading data concatenated. The filtered states were causal *given* those
parameters — but the parameters had seen the future.

Caught by a probe, not by reading the code: perturbing only the **last** trading day moved
β on **earlier** days by 3.8e-02. Fixed by fitting on the formation window and calling
`.filter(params)` with them held fixed.

`test_lookahead.py` now asserts the untouched prefix is bit-identical at five cut points
for all three estimators, **and** includes a meta-test confirming the probe catches a
deliberately non-causal estimator. A probe that passes everything is not a probe.

### 6.3 Two API assumptions that would have thrown

`spa_test` returns `pvalues{lower, consistent, upper}`, not a scalar `pvalue`. And
`criterion="sharpe"` cannot accept a zero-variance benchmark — correctly, since the Sharpe
of a constant is undefined. Against a zero benchmark the question is whether mean return
clears zero after the search, so `criterion="mean"` is the right test.

## 7. Limitations

**Stated first because it is the one that cannot be fixed here.** Point-in-time membership
fixes *selection*, not *prices*. yfinance will not serve history for a delisted, acquired
or renamed company, so **190 of 829 union members (22.9%) cannot be priced**. Coverage is
**639/829 = 77.1%**. The residual bias is "point-in-time members minus the ones we cannot
price" — strictly smaller than survivorship bias, but not zero. A genuinely clean fix
needs CRSP or Norgate.

The coverage loss is **time-varying**, and that is a finding rather than bookkeeping.
Early windows lose proportionally more names, because 2011-era members are
disproportionately the ones since delisted: W00 has 330 eligible names against W28's 486.
The study is structurally better-powered after ~2018.

Further limits:

- **One protocol, one sample.** A null here is a statement about GGR-style cointegration
  selection on the US large-cap universe 2011–2026. It is not a claim about pairs trading
  in general, other universes, intraday horizons, or other selection methods.
- **The result is a non-detection, not a disproof.** CI [−0.72, +0.30] contains zero.
- **Long/short, and therefore not tradable in a cash account.** This is research. No
  position in it was ever taken.
- **Kalman-vs-rolling ordering is unexplained** (§4.1).

## 8. Reproducing

```bash
python formation.py      # download panel, coverage report, data-quality audit
python selection.py      # stage A: 2.46M Engle-Granger tests, ~20 min on 16 cores
python grade_A.py        # registered metrics + cost curve
python grade_C.py        # three estimators, Model Confidence Set
python allocate.py       # convex allocation vs equal weight
python test_lookahead.py
python test_data_quality.py
```

Bars were frozen with `prereg.freeze()`, which commits, pushes, and verifies the bytes
landed before returning. `prereg.grade()` raises `NotFrozen` on any registration that is
not pushed — a bar that exists only on one disk can still be rewritten after the result is
known, which is not a bar.

Frozen blobs: **A** `73543a53` · **C** `ec59631b` · **B** `aa565e04`.

---

## References

Gatev, Goetzmann & Rouwenhorst (2006), *Pairs Trading: Performance of a Relative-Value
Arbitrage Rule*, Review of Financial Studies 19(3).

Rad, Low & Faff (2016), *The Profitability of Pairs Trading Strategies: Distance,
Cointegration, and Copula Methods*, Quantitative Finance 16(10).

Hansen (2005), *A Test for Superior Predictive Ability*, Journal of Business & Economic
Statistics 23(4).

Hansen, Lunde & Nason (2011), *The Model Confidence Set*, Econometrica 79(2).

Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio
Management 40(5).

Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*, JRSS-B 57(1).
