"""
selection -- stage A. Engle-Granger on every pair, then FDR control.

THE NUMBER THIS FILE EXISTS TO PRODUCE. Each window tests ~125,000 pairs at
alpha=0.05. Under the null of no cointegration anywhere, that alone yields
about 6,250 "significant" pairs. Any selection method that reports its hit
count without that denominator is reporting noise as a finding. So this module
always emits three counts together:

    tested          how many hypotheses were run
    sig_raw         how many cleared alpha=0.05 uncorrected
    expected_fp     alpha * tested -- what pure noise would have given you
    sig_fdr         how many survive Benjamini-Hochberg at q=0.10

WHY BENJAMINI-HOCHBERG AND NOT BONFERRONI. Bonferroni controls the probability
of even one false positive, which at 125,000 tests means a per-test threshold
of 4e-7 and essentially guarantees an empty book. FDR controls the expected
PROPORTION of the selected set that is false, which is the question actually
being asked here: of the pairs I chose to trade, what share are noise? A book
that is 10% false positives is tradable; the Bonferroni framing throws away
the whole book to avoid one.

FORMATION WINDOW ONLY. Every test in here reads window.formation(panel) and
nothing else. The trading window does not exist yet at selection time.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

import formation as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports", "selection")

ALPHA = 0.05        # per-test level, for the noise-denominator comparison
Q_FDR = 0.10        # Benjamini-Hochberg false discovery rate
BOOK_SIZE = 20      # top-k by test statistic, GGR convention
MIN_OVERLAP = 200   # trading days both legs must share inside the formation window


def _test_chunk(logp, pairs):
    """Engle-Granger over one chunk of (i, j) index pairs. Returns (t, p) arrays."""
    out_t = np.full(len(pairs), np.nan)
    out_p = np.full(len(pairs), np.nan)
    for k, (i, j) in enumerate(pairs):
        a, b = logp[:, i], logp[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < MIN_OVERLAP:
            continue
        try:
            t, p, _ = coint(a[ok], b[ok])
            out_t[k], out_p[k] = t, p
        except Exception:
            pass          # singular / degenerate pair stays NaN and is dropped
    return out_t, out_p


def select(window, panel, n_jobs=-1, chunk=2000, verbose=True):
    """Run stage A for one window. Returns a dict; writes nothing."""
    names = F.members(window, panel)
    form = window.formation(panel)[names]
    logp = np.log(form.to_numpy(dtype=float))

    n = len(names)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if verbose:
        print(f"W{window.index:02d} {window.formation_start:%Y-%m-%d}: "
              f"{n} names, {len(pairs):,} pairs", flush=True)

    chunks = [pairs[k:k + chunk] for k in range(0, len(pairs), chunk)]
    res = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_test_chunk)(logp, c) for c in chunks)
    tstat = np.concatenate([r[0] for r in res])
    pval = np.concatenate([r[1] for r in res])

    valid = np.isfinite(pval)
    tested = int(valid.sum())
    p_ok = pval[valid]
    t_ok = tstat[valid]
    idx_ok = [pairs[k] for k in np.flatnonzero(valid)]

    sig_raw = int((p_ok <= ALPHA).sum())
    expected_fp = ALPHA * tested

    reject, _, _, _ = multipletests(p_ok, alpha=Q_FDR, method="fdr_bh")
    sig_fdr = int(reject.sum())

    # Book: strongest rejections among the FDR survivors. More negative t is
    # stronger evidence, so rank ascending.
    surv = np.flatnonzero(reject)
    order = surv[np.argsort(t_ok[surv])][:BOOK_SIZE]
    book = [{"a": names[idx_ok[k][0]], "b": names[idx_ok[k][1]],
             "tstat": float(t_ok[k]), "pvalue": float(p_ok[k])} for k in order]

    return {
        "window": window.index,
        "formation_start": f"{window.formation_start:%Y-%m-%d}",
        "formation_end": f"{window.formation_end:%Y-%m-%d}",
        "trading_end": f"{window.trading_end:%Y-%m-%d}",
        "names": n,
        "pairs_enumerated": len(pairs),
        "tested": tested,
        "alpha": ALPHA,
        "sig_raw": sig_raw,
        "expected_fp": expected_fp,
        "excess_over_noise": sig_raw - expected_fp,
        "q_fdr": Q_FDR,
        "sig_fdr": sig_fdr,
        "book_size": len(book),
        "book": book,
    }


def main():
    ap = argparse.ArgumentParser(description="stage A -- cointegration selection")
    ap.add_argument("--windows", default="all", help="'all' or comma-separated indices")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--no-price-filter", action="store_true",
                    help="membership overlap only -- sensitivity check on the "
                         "one rule that cannot be made causal")
    ap.add_argument("--out", default=None, help="override output dir")
    args = ap.parse_args()

    if args.no_price_filter:
        F.PRICE_FILTER = False
        print("PRICE_FILTER DISABLED -- membership overlap only")
    out_dir = args.out or OUT

    panel = F.download()
    ws = F.windows()
    if args.windows != "all":
        want = {int(x) for x in args.windows.split(",")}
        ws = [w for w in ws if w.index in want]

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for w in ws:
        r = select(w, panel, n_jobs=args.jobs)
        with open(os.path.join(out_dir, f"W{w.index:02d}.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=2)
        print(f"   tested {r['tested']:,} | sig@0.05 {r['sig_raw']:,} "
              f"| noise expects {r['expected_fp']:,.0f} "
              f"| FDR q=0.10 keeps {r['sig_fdr']:,} | book {r['book_size']}",
              flush=True)
        rows.append(r)

    if rows:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "book"} for r in rows])
        df.to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding="utf-8")
        print(f"\n{len(rows)} windows | tests {df.tested.sum():,} | "
              f"sig@0.05 {df.sig_raw.sum():,} vs noise {df.expected_fp.sum():,.0f} | "
              f"FDR survivors {df.sig_fdr.sum():,}")
        print(f"windows with an empty book: {(df.book_size == 0).sum()}")


if __name__ == "__main__":
    main()
