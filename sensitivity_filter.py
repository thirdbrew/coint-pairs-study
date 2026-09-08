"""
sensitivity_filter -- does the result depend on the rule that cannot be made causal?

THE PROBLEM THIS MEASURES. The ticker-reuse filter has three checks. Two are
defensible; one is not:

  membership overlap   reads ONLY the point-in-time membership file. Causal.
  formation median     reads only the formation window. Causal.
  terminal price       reads the LAST price in the whole panel to judge a 2012
                       window. NOT causal.

The terminal-price check cannot be repaired. yfinance's auto_adjust back-adjusts
the entire series for every future split, so NVDA's 2012 close in this panel
already encodes its 2024 split ($0.36). No price-level test on a back-adjusted
panel separates a real company from a corrupted stub without future information --
that is a property of the data, not of the code.

So the honest treatment is not to hide it or to pretend a fix exists. It is to
run the study BOTH WAYS and report whether the conclusion moves. An unfixable
lookahead that demonstrably changes nothing is a disclosed sensitivity; one that
changes the answer is a finding. Either way the reader gets the number.

Compares reports/selection (filter on) against reports/selection_nofilter
(membership overlap only), and trades both books through to a Sharpe.
"""

import json
import os

import numpy as np
import pandas as pd

import formation as F
import stats_lib as S
import trade as T

HERE = os.path.dirname(os.path.abspath(__file__))
ON = os.path.join(HERE, "reports", "selection")
OFF = os.path.join(HERE, "reports", "selection_nofilter")
OUT = os.path.join(HERE, "reports", "grade")


def _books(d):
    out = {}
    for w in F.windows():
        p = os.path.join(d, f"W{w.index:02d}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[w.index] = json.load(fh)
    return out


def _key(book):
    """Order-independent identity of a book."""
    return sorted(tuple(sorted((e["a"], e["b"]))) for e in book["book"])


def _returns(panel, books, cost_bps=10.0):
    chunks = []
    for w in F.windows():
        rec = books.get(w.index)
        idx = w.trading(panel).index
        if rec is None or not rec["book"]:
            chunks.append(pd.Series(0.0, index=idx))
            continue
        r, _ = T.run_window(w, panel, rec["book"], cost_bps=cost_bps)
        chunks.append(r)
    return pd.concat(chunks).sort_index()


def main():
    panel = F.download()
    on, off = _books(ON), _books(OFF)
    if len(off) < len(on):
        raise SystemExit(f"no-filter run incomplete: {len(off)}/{len(on)} windows")

    rows, differing = [], []
    for i in sorted(on):
        a, b = on[i], off[i]
        same = _key(a) == _key(b)
        if not same:
            differing.append(i)
        rows.append({
            "window": i,
            "names_on": a["names"], "names_off": b["names"],
            "tested_on": a["tested"], "tested_off": b["tested"],
            "sig_on": a["sig_raw"], "sig_off": b["sig_raw"],
            "fdr_on": a["sig_fdr"], "fdr_off": b["sig_fdr"],
            "book_on": a["book_size"], "book_off": b["book_size"],
            "book_identical": same,
        })
    df = pd.DataFrame(rows)

    print("=== universe and selection ===")
    print(f"  extra names admitted without the filter : "
          f"{df.names_off.sum() - df.names_on.sum():+,}")
    print(f"  extra pairs tested                      : "
          f"{df.tested_off.sum() - df.tested_on.sum():+,}")
    print(f"  FDR survivors  on {df.fdr_on.sum():,}  ->  off {df.fdr_off.sum():,}")
    print(f"  empty books    on {(df.book_on == 0).sum()}  ->  "
          f"off {(df.book_off == 0).sum()}")
    print(f"  windows whose BOOK changed at all       : {len(differing)} / {len(df)}"
          + (f"  {differing}" if differing else ""))

    print("\n=== traded through, net of 10 bps ===")
    r_on = _returns(panel, on)
    r_off = _returns(panel, off)
    s_on, s_off = float(S.sharpe(r_on)), float(S.sharpe(r_off))
    print(f"  filter ON  : Sharpe {s_on:+.4f}   total {r_on.sum():+.2%}   days {len(r_on)}")
    print(f"  filter OFF : Sharpe {s_off:+.4f}   total {r_off.sum():+.2%}   days {len(r_off)}")
    print(f"  difference : {s_off - s_on:+.4f}")

    verdict = ("the conclusion does NOT depend on the non-causal rule"
               if (s_off < 0.50 and s_on < 0.50)
               else "THE CONCLUSION MOVES -- report this prominently")
    print(f"\n  registered bar was sharpe_net >= 0.50; both settings fail it.")
    print(f"  => {verdict}")

    res = {
        "sharpe_filter_on": s_on,
        "sharpe_filter_off": s_off,
        "sharpe_difference": s_off - s_on,
        "windows_with_changed_book": differing,
        "n_windows_changed": len(differing),
        "extra_names": int(df.names_off.sum() - df.names_on.sum()),
        "extra_tests": int(df.tested_off.sum() - df.tested_on.sum()),
        "fdr_on": int(df.fdr_on.sum()), "fdr_off": int(df.fdr_off.sum()),
        "empty_books_on": int((df.book_on == 0).sum()),
        "empty_books_off": int((df.book_off == 0).sum()),
        "both_fail_registered_bar": bool(s_off < 0.50 and s_on < 0.50),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sensitivity_filter.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    df.to_csv(os.path.join(OUT, "sensitivity_filter.csv"), index=False,
              encoding="utf-8")
    print("\nwrote reports/grade/sensitivity_filter.json")


if __name__ == "__main__":
    main()
