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
import functools
import os
import time

import numpy as np
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


# MAX_PX is 10,000 because NVR genuinely trades near $7,000 -- a ceiling has to
# sit above the real outlier or it deletes a real company to catch a fake one.
MAX_PX = 10_000.0
MIN_LAST_PX = 1.0         # on the TERMINAL price only -- see below
MAX_DAY_MOVE = 1.0        # |log return| on any single day (~2.7x)


def _last_px(series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def panel_rejects(panel):
    """Tickers whose series cannot be the company the symbol claims. {ticker: reason}.

    WHY THIS EXISTS. The 190 unpriceable names are the VISIBLE half of the
    delisting problem. The invisible half is worse: a retired symbol that was
    reused, or that resolves to a foreign listing, returns the WRONG COMPANY'S
    PRICES at plausible-looking magnitudes, and nothing in the pipeline objects.
    Missing data announces itself; wrong data does not. Measured here: PARA
    splices a ~101,500 foreign listing onto Paramount's $3; MHS prices from 2020
    for a company acquired in 2012; SBNY prices from 2024 for a bank that failed
    in 2023.

    Not benign noise. In W23 the mis-resolved PARA produced one pair returning
    +149% and carried the whole window to +72.7%.

    TWO TESTS, AND NEITHER IS AN ARBITRARY PRICE FLOOR -- which matters, because
    the first version of this filter used one and deleted NVIDIA. auto_adjust
    back-adjusts for splits, so a stock with large splits has an arbitrarily
    small price EARLY: NVDA's legitimate 2011 close here is $0.36. An absolute
    floor on the median cannot tell that from a corrupted stub.

      1. MEMBERSHIP OVERLAP. The series must carry prices during a spell when
         the ticker was actually in the index. Zero overlap means the symbol was
         reused by something else. Uses the same PIT file as selection, so it
         costs no new data.
      2. TERMINAL PRICE. Back-adjustment scales history DOWN and leaves the last
         price untouched, so the endpoint is the honest one. NVDA ends at 194.97
         and passes; COL ends at 0.05 and CPWR at 0.01, and they do not.

    NOTE THE DIRECTION, because it is what makes this a data fix rather than a
    result fix: the contaminated pair MADE money. Removing it makes this study's
    numbers WORSE. Specified on data provenance alone, and written before any
    graded metric was read.
    """
    out = {}
    for t in panel.columns:
        s = panel[t].dropna()
        if len(s) == 0:
            continue

        spells = _membership_spells(t)
        if spells and not any(not (s.index[-1] < a or s.index[0] > b)
                              for a, b in spells):
            span = f"{s.index[0]:%Y-%m}..{s.index[-1]:%Y-%m}"
            mem = ", ".join(f"{a:%Y-%m}..{b:%Y-%m}" for a, b in spells)
            out[t] = f"prices {span} never overlap membership {mem}"
            continue

        last = _last_px(s)
        if not np.isfinite(last) or last < MIN_LAST_PX:
            out[t] = f"terminal price {last:,.2f} below {MIN_LAST_PX:,.2f}"
    return out


@functools.lru_cache(maxsize=None)
def _membership_spells(ticker):
    """(start, end) Timestamps this ticker was an index member. Open spells end today."""
    df = pd.read_csv(u.PIT_CSV)
    rows = df[df["ticker"] == ticker]
    spells = []
    for _, r in rows.iterrows():
        a = pd.Timestamp(r["start_date"])
        b = (pd.Timestamp(r["end_date"]) if isinstance(r["end_date"], str)
             and r["end_date"].strip() else pd.Timestamp(END))
        spells.append((a, b))
    return tuple(spells)


def implausible(prices):
    """Formation-window checks: price level and single-day discontinuity."""
    s = prices.dropna()
    if len(s) == 0:
        return "no data"
    med = float(s.median())
    if med > MAX_PX:
        return f"median {med:,.2f} above {MAX_PX:,.0f} (foreign listing / wrong currency)"
    if float(s.min()) <= 0:
        return "non-positive price"
    step = np.abs(np.diff(np.log(s.to_numpy(dtype=float))))
    if len(step) and np.nanmax(step) > MAX_DAY_MOVE:
        return f"single-day log move {np.nanmax(step):.2f} exceeds {MAX_DAY_MOVE:.2f}"
    return None


_PANEL_BAD = {}


def panel_bad(panel):
    """Cached panel-level rejections. Computed once, used by every window."""
    key = (id(panel), panel.shape)
    if key not in _PANEL_BAD:
        _PANEL_BAD[key] = panel_rejects(panel)
    return _PANEL_BAD[key]


def members(window, panel, min_obs=200, rejects=None):
    """Tickers eligible for `window`: index members at formation start, priced,
    and carrying a price series that is plausible for a US listing.

    min_obs is against the formation window, not the whole sample. A name that
    joined the index mid-formation has a short history here and a hedge ratio
    fit on 40 observations is noise -- 200 of ~252 trading days is the floor.

    Pass a dict as `rejects` to collect {ticker: reason} for the audit trail.
    A filter that drops names silently is a second invisible bias.
    """
    asof = u.sp500_asof(window.formation_start.strftime("%Y-%m-%d"))
    form = window.formation(panel)
    bad = panel_bad(panel)
    ok = []
    for t in asof:
        if t not in form.columns or form[t].notna().sum() < min_obs:
            continue
        why = bad.get(t) or implausible(form[t])
        if why is not None:
            if rejects is not None:
                rejects[t] = why
            continue
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
    rejects, total = {}, 0
    counts = {}
    for w in ws:
        n = len(members(w, panel, rejects=rejects))
        counts[w.index] = n
        total += n * (n - 1) // 2
    for i in (0, len(ws) // 2, len(ws) - 1):
        n = counts[i]
        print(f"  W{i:02d} form {ws[i].formation_start:%Y-%m-%d}: "
              f"{n} names, {n * (n - 1) // 2:,} pairs")
    print(f"\ntotal Engle-Granger tests across all {len(ws)} windows: {total:,}")

    print(f"\nprice-plausibility rejections: {len(rejects)} distinct tickers")
    for t, why in sorted(rejects.items()):
        print(f"  {t:<8} {why}")
    if rejects:
        print("\n  These are not missing data -- they PRICED, with the wrong company's")
        print("  numbers. Missing data announces itself; wrong data does not.")


if __name__ == "__main__":
    main()
