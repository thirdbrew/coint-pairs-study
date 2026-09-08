"""
test_data_quality -- the ticker-reuse filter, pinned in both directions.

WHY BOTH DIRECTIONS. The first version of this filter used an absolute price
floor and rejected NVDA, because auto_adjust back-adjusts splits and NVIDIA's
legitimate 2011 close in this panel is $0.36. A filter is only as good as the
false positives it does NOT produce, so every test here has a matching
must-retain case.

The must-reject names are all symbols that were retired and later reassigned:
FB (Facebook -> META, symbol reused), EMC (acquired by Dell 2016), S (Sprint ->
SentinelOne), BEAM (Beam Inc spirits -> Beam Therapeutics), SE (Spectra Energy
-> Sea Limited). yfinance returns the NEW company's prices under the OLD symbol,
with no error and no gap.

Needs the cached panel. Run formation.py once first.
"""

import numpy as np
import pandas as pd
import pytest

import formation as F

MUST_REJECT = ["FB", "EMC", "S", "BEAM", "SE", "STI", "SHLD", "SPLS",
               "MHS", "SBNY", "MI", "COL", "CPWR"]
MUST_RETAIN = ["NVDA", "AAPL", "KO", "NVR", "AIV", "FSLR", "SEDG", "MSFT", "JNJ"]


@pytest.fixture(scope="module")
def panel():
    return F.download()


@pytest.fixture(scope="module")
def rejects(panel):
    return F.panel_rejects(panel)


def test_reused_symbols_are_rejected(panel, rejects):
    missed = [t for t in MUST_REJECT if t in panel.columns and t not in rejects]
    assert not missed, (
        f"these symbols were reassigned to a different company and the filter let "
        f"them through: {missed}")


def test_legitimate_names_are_retained(panel, rejects):
    """The NVDA guard. A price floor on back-adjusted data deletes real companies."""
    killed = {t: rejects[t] for t in MUST_RETAIN if t in rejects}
    assert not killed, f"filter rejected legitimate companies: {killed}"


def test_nvda_is_the_specific_regression(panel):
    """Pin the exact case that broke: NVDA's back-adjusted early price is sub-$1."""
    if "NVDA" not in panel.columns:
        pytest.skip("NVDA not in panel")
    s = panel["NVDA"].dropna()
    assert s.iloc[0] < 1.0, (
        "NVDA's early back-adjusted price is no longer sub-$1, so this regression "
        "no longer reproduces -- re-derive the guard rather than deleting it")
    assert s.iloc[-1] > 100.0
    assert "NVDA" not in F.panel_rejects(panel)


def test_max_px_sits_above_the_real_outlier(panel):
    """NVR really trades in the thousands. The ceiling must clear it."""
    if "NVR" not in panel.columns:
        pytest.skip("NVR not in panel")
    assert float(panel["NVR"].dropna().max()) < F.MAX_PX


def test_filter_actually_removes_pairs_from_the_universe(panel):
    """Guard the guard: if members() ignored the filter, every test above passes
    while the contaminated names still trade."""
    w = F.windows()[23]                      # the window PARA contaminated
    rej = {}
    names = F.members(w, panel, rejects=rej)
    assert "PARA" not in names, "PARA is still eligible in W23 -- filter not wired in"


if __name__ == "__main__":
    p = F.download()
    r = F.panel_rejects(p)
    print(f"panel-level rejections: {len(r)}\n")
    missed = [t for t in MUST_REJECT if t in p.columns and t not in r]
    killed = {t: r[t] for t in MUST_RETAIN if t in r}
    print(f"must-reject let through : {missed or 'none'}")
    print(f"must-retain wrongly cut : {killed or 'none'}")
    w = F.windows()[23]
    print(f"PARA eligible in W23    : {'PARA' in F.members(w, p)}")
    assert not missed and not killed
    print("\nall data-quality tests passed")
