"""
mechanism_C -- does an adaptive hedge ratio absorb the divergence it should trade?

The claim to test: a beta that tracks the spread partially cancels the very
divergence the strategy exists to trade, so adaptive estimators should spend
less time beyond the entry threshold than a static one.

THIS FILE EXISTS BECAUSE THE FIRST VERSION OF THIS TABLE WAS WRONG. It built z
from `np.nanmean(beta_t)` over the TRADING window -- the same leak that was in
trade.pair_returns -- so every figure in it was computed from data the strategy
could not have had. The normalisation here mirrors the fixed path exactly: the
estimator's own beta as of the LAST FORMATION DAY, from formation data only.

Writes reports/grade/stage_C_mechanism.json.
"""

import json
import os

import numpy as np

import formation as F
import hedge_ratio as H
import trade as T

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "reports", "selection")
OUT = os.path.join(HERE, "reports", "grade")


def _books():
    out = {}
    for w in F.windows():
        p = os.path.join(SEL, f"W{w.index:02d}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[w.index] = json.load(fh)["book"]
    return out


def measure(panel, books, name, fn):
    """Mean |z|, fraction of days beyond the entry threshold, and sd(beta_t)."""
    zabs, beyond, bsd = [], [], []
    for w in F.windows():
        for e in books.get(w.index, []):
            a, b = e["a"], e["b"]
            if a not in panel.columns or b not in panel.columns:
                continue
            form, tr = w.formation(panel), w.trading(panel)
            if len(tr) == 0:
                continue
            f_la = np.log(form[a].to_numpy(float))
            f_lb = np.log(form[b].to_numpy(float))
            t_la = np.log(tr[a].to_numpy(float))
            t_lb = np.log(tr[b].to_numpy(float))

            if name == "static_ols":
                beta, alpha = T.ols_beta(f_la, f_lb)
                if not np.isfinite(beta):
                    continue
                beta_t = np.full(len(t_la), beta)
                alpha_t = np.full(len(t_la), alpha)
            else:
                beta_t, alpha_t = fn(f_la, f_lb, t_la, t_lb)
                # formation-end beta, formation data only -- mirrors trade.py
                nb, na = fn(f_la[:-1], f_lb[:-1], f_la[-1:], f_lb[-1:])
                beta, alpha = float(nb[-1]), float(na[-1])
                if not np.isfinite(beta) or not np.isfinite(alpha):
                    continue

            f_spread = f_la - (alpha + beta * f_lb)
            mu, sigma = np.nanmean(f_spread), np.nanstd(f_spread, ddof=1)
            if not np.isfinite(sigma) or sigma <= 0:
                continue

            z = (t_la - (alpha_t + beta_t * t_lb) - mu) / sigma
            zabs.append(float(np.nanmean(np.abs(z))))
            beyond.append(float(np.nanmean(np.abs(z) > T.ENTRY_Z)))
            bsd.append(float(np.nanstd(beta_t)))

    return {
        "mean_abs_z": float(np.nanmean(zabs)),
        "frac_beyond_entry": float(np.nanmean(beyond)),
        "mean_beta_sd": float(np.nanmean(bsd)),
        "pairs": len(zabs),
    }


def main():
    panel = F.download()
    books = _books()
    os.makedirs(OUT, exist_ok=True)

    res = {}
    print(f"{'estimator':14s} {'mean|z|':>9s} {'days |z|>2':>12s} {'sd(beta_t)':>12s}")
    for name, fn in H.ESTIMATORS.items():
        r = measure(panel, books, name, fn)
        res[name] = r
        print(f"{name:14s} {r['mean_abs_z']:9.3f} "
              f"{r['frac_beyond_entry'] * 100:11.1f}% {r['mean_beta_sd']:12.4f}")

    with open(os.path.join(OUT, "stage_C_mechanism.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("\nwrote reports/grade/stage_C_mechanism.json")


if __name__ == "__main__":
    main()
