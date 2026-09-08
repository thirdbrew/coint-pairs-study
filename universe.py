"""
universe -- point-in-time S&P 500 membership, replacing the survivorship-biased scraper.
========================================================================================

THE BUG THIS REPLACES
---------------------
Fifteen scripts in this repo carried their own copy of:

    def sp500():
        html = urlopen("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return pd.read_html(...)[0]["Symbol"].tolist()

That returns the index as it stands TODAY, and every one of those scripts then applied it
to 15 years of history. So `robustness.py`, `factor_lab.py`, `improve.py` and
`validate_system.py` were all ranking momentum inside a universe pre-filtered to companies
that survived to 2026. Enron-shaped names -- the ones momentum would have bought and then
been destroyed by -- were never in the sample to lose money on. That is textbook
survivorship bias, and it biases results in the direction that flatters the strategy.

It matters most for exactly the results this project already found fragile: `robustness.py`
concluded plain momentum FAILED out-of-sample, and `improve.py` was built to rescue it. Any
conclusion drawn from a survivor-only universe is drawn from an easier game than the one
that was actually played.

WHAT THIS GIVES YOU
-------------------
    sp500_asof(date)          members on that date -- use at each REBALANCE
    sp500_union(start, end)   every ticker ever a member in the window -- the DOWNLOAD list
    membership_matrix(dates)  boolean date x ticker mask -- apply to returns before ranking
    sp500_current()           today's members, from the same PIT file (not Wikipedia)
    coverage_report(...)      how much of the union yfinance could actually price

THE CAVEAT THAT TRAVELS WITH THIS (do not drop it from write-ups)
-----------------------------------------------------------------
This fixes SELECTION, not PRICES. yfinance will not return history for a company that has
been delisted, acquired or renamed, so a chunk of the union downloads empty. The bias goes
from "current members over 15y" to "PIT members minus the ones we cannot price" -- strictly
smaller, not zero. `coverage_report()` prints the size of what remains so it is stated
rather than assumed. A genuinely clean fix needs a paid PIT dataset (CRSP, Norgate).

Data: fja05680/sp500, vendored at data/sp500_ticker_start_end.csv (see .SOURCE.txt).
"""

import functools
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIT_CSV = os.path.join(HERE, "data", "sp500_ticker_start_end.csv")


def _yf(sym):
    """Class shares are dotted upstream (BRK.B) and dashed in yfinance (BRK-B)."""
    return str(sym).strip().replace(".", "-")


@functools.lru_cache(maxsize=1)
def _spells():
    """One row per membership spell. A ticker may have several (52 of them do)."""
    d = pd.read_csv(PIT_CSV)
    d["ticker"] = d["ticker"].map(_yf)
    d["start_date"] = pd.to_datetime(d["start_date"]).dt.strftime("%Y-%m-%d")
    d["end_date"] = pd.to_datetime(d["end_date"]).dt.strftime("%Y-%m-%d")
    return d


def _d(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def sp500_asof(date):
    """Tickers in the index ON `date`. Spells are [start, end); a blank end means still in."""
    date = _d(date)
    s = _spells()
    live = (s["start_date"] <= date) & (s["end_date"].isna() | (s["end_date"] > date))
    return sorted(s.loc[live, "ticker"].unique())


def sp500_union(start, end):
    """Every ticker that was a member at ANY point in [start, end].

    This is the download list. It is deliberately larger than any single date's membership --
    that surplus IS the survivorship correction, and shrinking it to "names that still trade"
    would quietly reintroduce the bug.
    """
    start, end = _d(start), _d(end)
    s = _spells()
    overlaps = (s["start_date"] <= end) & (s["end_date"].isna() | (s["end_date"] >= start))
    return sorted(s.loc[overlaps, "ticker"].unique())


# The last union this process requested. Recorded so assert_coverage() can check what
# actually downloaded against what was asked for, without every lab having to thread the
# list through its own locals.
_LAST_UNION = []


def sp500_union_lookback(years):
    """sp500_union over the trailing `years`, matching a yf.download(period="Ny") call.

    Kept as its own entry point because every backtest in this repo declares its window as a
    yfinance period string, and the universe must cover exactly that span -- a shorter union
    silently reintroduces the survivorship the union exists to remove.
    """
    import datetime as _dt
    global _LAST_UNION
    end = _dt.date.today()
    start = end - _dt.timedelta(days=int(round(365.25 * years)))
    _LAST_UNION = sp500_union(start.isoformat(), end.isoformat())
    return list(_LAST_UNION)


def sp500_current():
    """Today's members, read from the PIT file rather than scraped.

    Screens that legitimately want "what is in the index now" (sp500_screen.py,
    uncorr_screen.py) should call this: one data source, offline, and it cannot drift from
    the historical file the backtests use.
    """
    s = _spells()
    return sorted(s.loc[s["end_date"].isna(), "ticker"].unique())


def membership_matrix(dates, tickers=None):
    """Boolean frame (index=dates, columns=tickers): was this ticker in the index that day?

    Apply it before ranking, so a name only competes on dates it was actually a member:

        mask = membership_matrix(rets.index, rets.columns)
        scores = scores.where(mask)
    """
    dates = pd.DatetimeIndex(dates)
    s = _spells()
    if tickers is None:
        tickers = sorted(s["ticker"].unique())
    tickers = list(tickers)
    ds = pd.Series(dates.strftime("%Y-%m-%d"), index=dates)
    out = pd.DataFrame(False, index=dates, columns=tickers)
    for tkr, grp in s[s["ticker"].isin(tickers)].groupby("ticker"):
        live = pd.Series(False, index=dates)
        for _, row in grp.iterrows():
            a = ds >= row["start_date"]
            b = ds < row["end_date"] if pd.notna(row["end_date"]) else pd.Series(True, index=dates)
            live |= (a & b)
        out[tkr] = live.values
    return out


@functools.lru_cache(maxsize=4096)
def _members_on(date):
    return frozenset(sp500_asof(date))


def pit_filter(scores, date):
    """Drop names that were not index members on `date`. The one-line ranking gate.

    Insert immediately after a cross-sectional score is built:

        s = momS.loc[d].dropna()
        s = universe.pit_filter(s, d)      # <-- only members compete

    Needed because sp500_union() deliberately downloads every name that was EVER a member,
    which is what removes survivorship bias -- but without this gate a 2012 portfolio could
    hold a company that did not join the index until 2019. The union and this filter are two
    halves of one fix; using either alone trades one bias for another.
    """
    members = _members_on(_d(date))
    keep = [c for c in scores.index if c in members]
    return scores.loc[keep]


def coverage_report(union, priced, label=""):
    """State how much of the PIT universe actually had price data. Never leave this implicit.

    `union`  = what sp500_union asked for
    `priced` = what came back with usable history
    """
    union, priced = set(union), set(priced)
    missing = sorted(union - priced)
    # Count the INTERSECTION, not len(priced). Callers pass the whole downloaded frame,
    # which also holds SPY, BIL and any ETF sleeve assets -- none of them index members.
    # Dividing by the raw count printed "coverage 100.2%", which is impossible on its face
    # and would have quietly padded a degraded run with benchmark tickers. (The enforced
    # threshold in assert_coverage always used the intersection, so nothing was let through;
    # only the printed figure was wrong.)
    hit = len(priced & union)
    pct = 100.0 * hit / len(union) if union else 0.0
    lines = [
        f"PIT universe coverage{(' -- ' + label) if label else ''}:",
        f"  requested (ever a member): {len(union)}",
        f"  priced by yfinance:        {hit} ({pct:.1f}%)",
        f"  unpriceable (delisted/renamed/acquired): {len(missing)}",
    ]
    if missing:
        lines.append(f"  residual survivorship bias remains via these {len(missing)} names, e.g. "
                     f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    return "\n".join(lines)


def assert_coverage(priced, min_frac=0.70, label="", union=None):
    """Print download coverage and REFUSE to continue when it is materially incomplete.

    WHY THIS IS HARD-FAILING RATHER THAN A WARNING. On 2026-08-19 three labs were run
    concurrently; the parallel yfinance downloads saturated the connection and between 352
    and 803 tickers failed per run. `diversify_test.py` and `uncorr_screen.py` did not
    error -- they ran to completion on 468 and 466 stocks (a universe 43% short) and printed
    a confident Sharpe table with no indication anything was missing. A degraded input read
    exactly like a good one, which is the failure this whole repo keeps re-learning.

    A clean sequential run prices ~78% of the point-in-time union (642 of 820); the rest are
    delisted names yfinance cannot serve. The 70% floor sits below that and well above the
    57% a contended run produces, so it separates "normal delisting attrition" from "the
    network ate half the universe".
    """
    union = set(union if union is not None else _LAST_UNION)
    if not union:
        return float("nan")
    priced = set(priced)
    frac = len(priced & union) / len(union)
    print(coverage_report(union, priced, label))
    if frac < min_frac:
        raise RuntimeError(
            f"download coverage {frac:.1%} is below the {min_frac:.0%} floor "
            f"({len(priced & union)} of {len(union)} names). This is a degraded download, "
            f"not a real universe -- results computed from it would be quietly wrong. "
            f"Re-run sequentially (concurrent yfinance downloads are the usual cause).")
    return frac


def assert_download(raw, required=("SPY",), min_frac=0.70, label=""):
    """Check a yf.download result BEFORE any code depends on it. Raises a diagnostic error.

    Placed before `raw["Close"].dropna(subset=["SPY"])` on purpose. When Yahoo rate-limits a
    run, SPY itself fails to arrive and that line dies with `KeyError: ['SPY']` -- which says
    nothing about the actual cause and sent one session hunting a pandas bug. The real
    message is "you were rate limited; wait and re-run sequentially".

    Rate limiting is the common cause: yfinance raises
    YFRateLimitError('Too Many Requests') per ticker and `yf.download` swallows it into the
    failed-downloads list rather than propagating, so a fully rate-limited run looks like a
    successful call that returned almost nothing.
    """
    try:
        cols = set(raw["Close"].columns)
    except Exception:
        raise RuntimeError(
            f"download returned no usable 'Close' block{(' -- ' + label) if label else ''}. "
            f"Almost always Yahoo rate limiting: wait a few minutes and re-run, one lab at "
            f"a time.")
    missing_req = [t for t in required if t not in cols]
    if missing_req:
        raise RuntimeError(
            f"benchmark ticker(s) {missing_req} missing from the download"
            f"{(' -- ' + label) if label else ''}. These are single, always-available "
            f"symbols, so their absence means the request failed wholesale -- Yahoo rate "
            f"limiting, not a data problem. Wait a few minutes and re-run sequentially.")
    return assert_coverage(cols, min_frac=min_frac, label=label)


def survivorship_delta(start, end):
    """Diagnostic: how much universe the old current-members-only approach threw away."""
    union = set(sp500_union(start, end))
    now = set(sp500_current())
    dropped = union - now
    return dict(union=len(union), current_only=len(now), dropped=len(dropped),
                dropped_pct=100.0 * len(dropped) / len(union) if union else 0.0)


if __name__ == "__main__":
    import datetime as dt
    end = dt.date.today().isoformat()
    start = (dt.date.today() - dt.timedelta(days=365 * 15)).isoformat()
    d = survivorship_delta(start, end)
    print(f"Window {start} .. {end}")
    print(f"  ever a member:        {d['union']}")
    print(f"  members today:        {d['current_only']}")
    print(f"  dropped by the old scraper: {d['dropped']} ({d['dropped_pct']:.1f}% of the universe)")
    print()
    for probe in [start, "2015-06-30", "2020-03-16", end]:
        print(f"  members on {probe}: {len(sp500_asof(probe))}")
