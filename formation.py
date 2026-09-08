"""
formation -- the price panel and the window schedule.

Everything downstream reads its prices and its dates from here, so that the
formation/trading split exists in exactly one place and cannot drift between
selection, estimation and allocation.

THE SPLIT IS THE WHOLE POINT. Gatev-Goetzmann-Rouwenhorst: 12-month formation,
6-month trading, non-overlapping trading windows rolled forward six months.
Formation windows overlap each other by six months; TRADING windows never
overlap, which is what keeps the out-of-sample returns from being counted twice.

Every quantity a trade depends on -- pair membership, hedge ratio, spread mean
and standard deviation -- is estimated inside the formation window and frozen
before the trading window opens. `Window.formation` and `Window.trading` are
separate slices on purpose; a function that takes the whole panel and a date
range is a function that can accidentally read forward.

PRICES ARE ADJUSTED (auto_adjust=True). An unadjusted panel puts a step
discontinuity in log(A) - beta*log(B) on every ex-dividend date, which reads as
a spread divergence and would be traded as one.

HARD STOP AT 2026-06-30. That is where data/sp500_ticker_start_end.csv ends.
Running past it silently reverts to a stale universe, which is the exact bug
universe.py was written to kill.
"""

import argparse
import os
import time

import pandas as pd
import yfinance as yf

import universe as u

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

START = "2011-01-01"
END = "2026-06-30"          # membership data ends here; do not move it
FORMATION_MONTHS = 12
TRADING_MONTHS = 6


# ------------------------------------------------------------------ windows
class Window:
    """One formation/trading pair. Slices are half-open [start, end)."""

    __slots__ = ("index", "formation_start", "formation_end", "trading_end")

    def __init__(self, index, formation_start, formation_end, trading_end):
        self.index = index
        self.formation_start = formation_start
        self.formation_end = formation_end      # also the trading START
        self.trading_end = trading_end

    @property
    def trading_start(self):
        return self.formation_end

    def formation(self, panel):
        return panel.loc[(panel.index >= self.formation_start)
                         & (panel.index < self.formation_end)]

    def trading(self, panel):
        return panel.loc[(panel.index >= self.trading_start)
                         & (panel.index < self.trading_end)]

    def __repr__(self):
        return (f"<W{self.index:02d} form {self.formation_start:%Y-%m-%d}"
                f"->{self.formation_end:%Y-%m-%d} trade ->{self.trading_end:%Y-%m-%d}>")


def windows(start=START, end=END):
    """The full schedule. 29 windows for the 2011-01-01 .. 2026-06-30 sample."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out, i = [], 0
    while True:
        f_start = start + pd.DateOffset(months=TRADING_MONTHS * i)
        f_end = f_start + pd.DateOffset(months=FORMATION_MONTHS)
        t_end = f_end + pd.DateOffset(months=TRADING_MONTHS)
        if f_end > end:
            break
        out.append(Window(i, f_start, f_end, min(t_end, end + pd.Timedelta(days=1))))
        i += 1
    return out


# ------------------------------------------------------------------ prices
def _cache_path():
    return os.path.join(CACHE, f"closes_{START}_{END}.csv")


def download(chunk=100, pause=1.0, force=False):
    """Sequential, chunked download of the point-in-time union. Cached.

    SEQUENTIAL ON PURPOSE. universe.assert_coverage documents the 2026-08-19
    incident: concurrent yfinance downloads saturated the connection, between
    352 and 803 tickers failed per run, and the labs did not error -- they
    printed confident Sharpe tables on a universe 43% short. A degraded
    download reads exactly like a good one, so this never parallelises and the
    coverage floor is enforced rather than warned about.
    """
    path = _cache_path()
    if os.path.exists(path) and not force:
        print(f"cache hit: {os.path.relpath(path, HERE)}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    union = sorted(set(u.sp500_union(START, END)) | {"SPY"})
    print(f"downloading {len(union)} tickers in chunks of {chunk} (sequential)...")

    frames = []
    for k in range(0, len(union), chunk):
        batch = union[k:k + chunk]
        raw = yf.download(batch, start=START, end=END, auto_adjust=True,
                          progress=False, threads=False)
        if raw is not None and len(raw):
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            frames.append(close)
        print(f"  {k + len(batch):4d}/{len(union)}")
        time.sleep(pause)

    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel.loc[:, ~panel.columns.duplicated()]
    os.makedirs(CACHE, exist_ok=True)
    panel.to_csv(path, encoding="utf-8")
    print(f"cached -> {os.path.relpath(path, HERE)}")
    return panel


def coverage(panel):
    """Print, and enforce, how much of the PIT union actually priced.

    Returns the fraction. Raises below the 70% floor -- see
    universe.assert_coverage for why that is fatal rather than a warning.
    """
    union = set(u.sp500_union(START, END))
    priced = {c for c in panel.columns if panel[c].notna().sum() > 0}
    return u.assert_coverage(priced, min_frac=0.70,
                             label=f"{START}..{END}", union=union)


def members(window, panel, min_obs=200):
    """Tickers eligible for `window`: index members at formation start, priced.

    min_obs is against the formation window, not the whole sample. A name that
    joined the index mid-formation has a short history here and a hedge ratio
    fit on 40 observations is noise -- 200 of ~252 trading days is the floor.
    """
    asof = u.sp500_asof(window.formation_start.strftime("%Y-%m-%d"))
    form = window.formation(panel)
    ok = []
    for t in asof:
        if t in form.columns and form[t].notna().sum() >= min_obs:
            ok.append(t)
    return sorted(ok)


def main():
    ap = argparse.ArgumentParser(description="price panel + window schedule")
    ap.add_argument("--force", action="store_true", help="re-download, ignore cache")
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()

    panel = download(chunk=args.chunk, force=args.force)
    print(f"\npanel: {panel.shape[0]} rows x {panel.shape[1]} cols, "
          f"{panel.index.min():%Y-%m-%d} .. {panel.index.max():%Y-%m-%d}\n")
    coverage(panel)

    ws = windows()
    print(f"\nwindows: {len(ws)}")
    print(f"  first {ws[0]}")
    print(f"  last  {ws[-1]}")

    print("\nper-window eligible names and pair counts:")
    total = 0
    for w in (ws[0], ws[len(ws) // 2], ws[-1]):
        m = members(w, panel)
        n = len(m)
        print(f"  W{w.index:02d} form {w.formation_start:%Y-%m-%d}: "
              f"{n} names, {n * (n - 1) // 2:,} pairs")
    for w in ws:
        n = len(members(w, panel))
        total += n * (n - 1) // 2
    print(f"\ntotal Engle-Granger tests across all {len(ws)} windows: {total:,}")


if __name__ == "__main__":
    main()
