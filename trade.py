"""
trade -- turn a selected book into a daily return series.

CONSTRUCTION. Each pair is held dollar-neutral and hedge-ratio weighted:

    w_a = pos / (1 + |beta|)          w_b = -pos * beta / (1 + |beta|)

so gross exposure is 1.0 per open pair by construction and the pair return is
w_a*r_a + w_b*r_b. Normalising by (1 + |beta|) is what stops a pair with
beta=4 from quietly carrying five times the risk of a pair with beta=1 and
dominating an "equally weighted" book.

RETURN ON COMMITTED CAPITAL. The portfolio return is the mean across all k
pairs in the book, not across the ones that happen to be open. A pair sitting
flat contributes 0 rather than being dropped from the denominator. Dropping it
would report the return of capital that was working while ignoring capital that
was committed and idle, which is how pairs studies inflate Sharpe. GGR use the
same convention.

SIGN. z = (spread - mu) / sigma on the FORMATION window's mu and sigma.
z > +entry means the spread is unusually wide: A is rich against B, so short
the spread (pos = -1, short A / long B). z < -entry is the mirror.

THE DELISTING RULE IS A LOOKAHEAD CONTROL, NOT A DETAIL. When a leg stops
pricing mid-trade -- acquisition, delisting, ticker change -- the position is
force-closed at the last valid price and stays flat for the rest of the window.
The tempting alternative, dropping the pair from the sample, deletes exactly
the outcomes that hurt (the acquired name that gapped away from its partner)
and is a survivorship leak that flatters the result.

COSTS are charged on realised turnover, |w_t - w_{t-1}| summed across both
legs, at cost_bps per unit of one-way turnover. Entering and exiting a pair
therefore pays roughly the full round trip.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import formation as F

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "reports", "selection")
OUT = os.path.join(HERE, "reports", "trade")

ENTRY_Z = 2.0       # GGR convention -- NOT tuned, see the spec's hard constraints
EXIT_Z = 0.5
COST_BPS = 10.0     # round-trip base case; the cost curve sweeps this


def ols_beta(la, lb):
    """Hedge ratio from formation-window log prices. Intercept included."""
    ok = np.isfinite(la) & np.isfinite(lb)
    if ok.sum() < 2:
        return np.nan, np.nan
    X = np.column_stack([np.ones(ok.sum()), lb[ok]])
    coef, *_ = np.linalg.lstsq(X, la[ok], rcond=None)
    return float(coef[1]), float(coef[0])


def pair_returns(window, panel, a, b, cost_bps=COST_BPS,
                 entry=ENTRY_Z, exit_=EXIT_Z, beta_fn=None):
    """Daily return series for one pair over its trading window.

    `beta_fn(form_la, form_lb, trade_la, trade_lb)` overrides the static OLS
    hedge ratio and returns a beta series aligned to the trading window --
    the seam stage C plugs rolling-OLS and Kalman estimators into.
    """
    form = window.formation(panel)
    trade = window.trading(panel)
    if a not in panel.columns or b not in panel.columns or len(trade) == 0:
        return None

    f_la, f_lb = np.log(form[a].to_numpy(float)), np.log(form[b].to_numpy(float))
    t_la, t_lb = np.log(trade[a].to_numpy(float)), np.log(trade[b].to_numpy(float))

    if beta_fn is None:
        beta, alpha = ols_beta(f_la, f_lb)
        if not np.isfinite(beta):
            return None
        beta_t = np.full(len(t_la), beta)
        alpha_t = np.full(len(t_la), alpha)
    else:
        beta_t, alpha_t = beta_fn(f_la, f_lb, t_la, t_lb)
        beta, alpha = float(np.nanmean(beta_t)), float(np.nanmean(alpha_t))

    f_spread = f_la - (alpha + beta * f_lb)
    mu, sigma = np.nanmean(f_spread), np.nanstd(f_spread, ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        return None

    t_spread = t_la - (alpha_t + beta_t * t_lb)
    z = (t_spread - mu) / sigma

    ra = np.diff(t_la, prepend=np.nan)
    rb = np.diff(t_lb, prepend=np.nan)
    alive = np.isfinite(t_la) & np.isfinite(t_lb)

    n = len(z)
    pos = np.zeros(n)
    dead = False
    cur = 0.0
    for k in range(n):
        if dead or not alive[k]:
            # leg stopped pricing: force-close at the last valid price and stay flat
            dead = True
            cur = 0.0
        elif cur == 0.0:
            if np.isfinite(z[k]) and abs(z[k]) > entry:
                cur = -np.sign(z[k])
        elif np.isfinite(z[k]) and abs(z[k]) < exit_:
            cur = 0.0
        pos[k] = cur
    pos[-1] = 0.0                      # force-close at window end

    denom = 1.0 + np.abs(beta_t)
    w_a = pos / denom
    w_b = -pos * beta_t / denom

    held_a = np.concatenate([[0.0], w_a[:-1]])
    held_b = np.concatenate([[0.0], w_b[:-1]])
    gross = np.nan_to_num(held_a * ra) + np.nan_to_num(held_b * rb)

    turnover = (np.abs(np.diff(w_a, prepend=0.0))
                + np.abs(np.diff(w_b, prepend=0.0)))
    net = gross - turnover * (cost_bps / 1e4)

    return pd.Series(net, index=trade.index, name=f"{a}/{b}")


def pair_panel(window, panel, book, cost_bps=COST_BPS, beta_fn=None):
    """Book -> DataFrame of per-pair daily returns, one column per tradable pair.

    This is the unit stage B allocates over. run_window collapses it with equal
    weights; allocate.py solves for the weights instead. Both consume the same
    object, so the comparison is like-for-like by construction rather than by
    two code paths that are meant to agree.
    """
    series = []
    for entry in book:
        s = pair_returns(window, panel, entry["a"], entry["b"],
                         cost_bps=cost_bps, beta_fn=beta_fn)
        if s is not None:
            series.append(s)
    if not series:
        return pd.DataFrame(index=window.trading(panel).index)
    return pd.concat(series, axis=1)


def run_window(window, panel, book, cost_bps=COST_BPS, beta_fn=None):
    """Book -> one daily portfolio return series (return on committed capital)."""
    df = pair_panel(window, panel, book, cost_bps=cost_bps, beta_fn=beta_fn)
    if df.empty or df.shape[1] == 0:
        return pd.Series(0.0, index=window.trading(panel).index), 0
    # divide by the BOOK size, not the number that survived -- committed capital
    return df.sum(axis=1) / len(book), df.shape[1]


def run_all(panel, cost_bps=COST_BPS, beta_fn=None, verbose=True):
    """Every window, concatenated into one continuous daily series."""
    ws = F.windows()
    chunks, meta = [], []
    for w in ws:
        path = os.path.join(SEL, f"W{w.index:02d}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            sel = json.load(fh)
        book = sel["book"]
        if not book:
            idx = w.trading(panel).index
            chunks.append(pd.Series(0.0, index=idx))
            meta.append({"window": w.index, "book": 0, "traded": 0})
            continue
        r, traded = run_window(w, panel, book, cost_bps=cost_bps, beta_fn=beta_fn)
        chunks.append(r)
        meta.append({"window": w.index, "book": len(book), "traded": traded})
        if verbose:
            print(f"W{w.index:02d} book {len(book):2d} traded {traded:2d} "
                  f"| ret {r.sum():+.4f} over {len(r)} days", flush=True)
    if not chunks:
        raise RuntimeError("no selection output found -- run selection.py first")
    return pd.concat(chunks).sort_index(), pd.DataFrame(meta)


def main():
    ap = argparse.ArgumentParser(description="stage A -- trade the selected book")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = ap.parse_args()

    panel = F.download()
    rets, meta = run_all(panel, cost_bps=args.cost_bps)
    os.makedirs(OUT, exist_ok=True)
    rets.to_csv(os.path.join(OUT, f"returns_{args.cost_bps:g}bps.csv"), encoding="utf-8")
    meta.to_csv(os.path.join(OUT, "windows.csv"), index=False, encoding="utf-8")

    import stats_lib as S
    print(f"\ncost {args.cost_bps:g} bps | days {len(rets)} | "
          f"total {rets.sum():+.2%} | Sharpe {S.sharpe(rets):.3f}")


if __name__ == "__main__":
    main()
