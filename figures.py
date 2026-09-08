"""
figures -- the four charts for reports/paper.md.

Palette is the validated categorical default: slot 1 blue #2a78d6, slot 2 orange
#eb6834. Checked with the data-viz validator on the light surface #fcfcfb --
adjacent CVD deltaE 24.7 (protan), normal-vision 33.6, both slots >= 3:1 contrast.
Colour is never the only encoding: every multi-series chart also carries a legend
AND direct end-labels, and the MCS exclusion in figure 4 is stated in text rather
than shown by hue.

One measure per axis, always. No dual axes anywhere.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports", "figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
S1 = "#2a78d6"          # blue
S2 = "#eb6834"          # orange
GRID = "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="600", loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")
    return path


def fig_noise(sel):
    """Observed 'significant' pairs vs what chance alone predicts, per window."""
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = pd.to_datetime(sel["formation_start"])
    ax.plot(x, sel["sig_raw"], color=S1, linewidth=2.0, label="Observed, p<0.05")
    ax.plot(x, sel["expected_fp"], color=S2, linewidth=2.0, linestyle="--",
            label="Expected from chance alone")
    ax.fill_between(x, sel["expected_fp"], sel["sig_raw"],
                    where=sel["sig_raw"] >= sel["expected_fp"],
                    color=S1, alpha=0.10, linewidth=0)

    _style(ax, "Almost all of the screen is noise",
           "Formation window start", "Cointegrated pairs found")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")

    ax.annotate("Observed", xy=(x.iloc[-1], sel["sig_raw"].iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                color=INK, fontsize=9, va="center", fontweight="600")
    ax.annotate("Chance", xy=(x.iloc[-1], sel["expected_fp"].iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                color=INK, fontsize=9, va="center", fontweight="600")

    total_o, total_e = sel["sig_raw"].sum(), sel["expected_fp"].sum()
    ax.text(0.012, 0.94,
            f"{total_o:,.0f} found  ·  {total_e:,.0f} expected by chance  "
            f"({total_o / total_e:.2f}x)\n"
            f"Benjamini-Hochberg leaves 1,446 — 0.83% of the naive count",
            transform=ax.transAxes, color=INK_2, fontsize=9,
            va="top", ha="left", linespacing=1.5)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    ax.set_xlim(x.iloc[0], x.iloc[-1] + pd.Timedelta(days=340))
    return _save(fig, "fig1_noise_ratio.png")


def fig_cost(res):
    """Net Sharpe against round-trip cost. It never crosses zero."""
    curve = pd.DataFrame(res["cost_curve"])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    ax.axhline(0.0, color=MUTED, linewidth=1.2, zorder=1)
    ax.plot(curve["cost_bps"], curve["sharpe"], color=S1, linewidth=2.0,
            marker="o", markersize=5, markerfacecolor=S1,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

    _style(ax, "Costs are not what kills it",
           "Round-trip cost (bps)", "Annualised Sharpe")

    z = curve.iloc[0]
    ax.annotate(f"Free trading: {z['sharpe']:+.3f}",
                xy=(z["cost_bps"], z["sharpe"]), xytext=(14, 16),
                textcoords="offset points", color=INK, fontsize=9,
                fontweight="600",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))
    ax.text(0.985, 0.93,
            "The curve never crosses zero.\nBreakeven cost: none at any level.",
            transform=ax.transAxes, color=INK_2, fontsize=9,
            va="top", ha="right", linespacing=1.5)
    # Anchored to the zero line in DATA coords. At axes-fraction 0.06 this sat at
    # the bottom of the plot, nowhere near the line it names, and read as an
    # x-axis label. Caught by looking at the render, not by the palette check.
    ax.text(curve["cost_bps"].min(), -0.006, "profitable above this line",
            color=MUTED, fontsize=8.5, va="top", ha="left")
    return _save(fig, "fig2_cost_curve.png")


def fig_equity(path):
    """Cumulative return of the FDR-surviving book at 10 bps."""
    r = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))

    ax.axhline(0.0, color=MUTED, linewidth=1.2, zorder=1)
    eq = r.cumsum()
    ax.plot(eq.index, eq.values * 100, color=S1, linewidth=1.8, zorder=3)
    ax.fill_between(eq.index, 0, eq.values * 100, color=S1, alpha=0.10, linewidth=0)

    _style(ax, "The surviving book, traded out-of-sample",
           "", "Cumulative return (%)")
    # Upper right: at lower left this ran straight through the equity line after
    # 2022. The flat stretches are worth naming -- they are the empty books.
    ax.text(0.985, 0.94,
            f"1,446 FDR-surviving pairs · 29 windows · net of 10 bps\n"
            f"{len(r):,} trading days, ending {eq.iloc[-1] * 100:+.1f}%\n"
            f"Flat stretches are the 9 windows with an empty book",
            transform=ax.transAxes, color=INK_2, fontsize=9,
            va="top", ha="right", linespacing=1.5)
    return _save(fig, "fig3_equity_curve.png")


def fig_estimators(c):
    """Sharpe with 95% CI for each hedge-ratio estimator."""
    order = ["static_ols", "rolling_ols", "kalman"]
    label = {"static_ols": "Static OLS", "rolling_ols": "Rolling OLS (60d)",
             "kalman": "Kalman random-walk beta"}
    fig, ax = plt.subplots(figsize=(7.6, 3.4))

    ax.axvline(0.0, color=MUTED, linewidth=1.2, zorder=1)
    for i, k in enumerate(order):
        y = len(order) - 1 - i
        lo, hi = c["sharpe_cis"][k]
        pt = c["sharpes"][k]
        ax.plot([lo, hi], [y, y], color=S1, linewidth=2.0, solid_capstyle="round",
                alpha=0.45, zorder=2)
        ax.plot([pt], [y], marker="o", markersize=9, color=S1,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)
        ax.text(hi + 0.06, y, f"{pt:+.3f}", color=INK, fontsize=9,
                va="center", fontweight="600")
        # Placed ABOVE the interval, not left of it: at ha="right" this ran
        # straight through the y-tick label. Caught by looking at the render.
        tag = ("MCS excluded, p=%.4f" % c["mcs_pvalues"][k]
               if k in c["mcs_excluded"]
               else "in the confidence set, p=%.4f" % c["mcs_pvalues"][k])
        ax.text(pt, y + 0.20, tag, color=INK_2, fontsize=8.5,
                va="bottom", ha="center")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label[k] for k in reversed(order)], fontsize=9.5)
    _style(ax, "The more adaptive the hedge ratio, the worse it does",
           "Annualised Sharpe (point estimate and 95% CI)", "")
    ax.tick_params(axis="y", labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-2.05, 0.85)
    ax.set_ylim(-0.55, 2.62)
    return _save(fig, "fig4_estimators.png")


def main():
    sel = pd.read_csv(os.path.join(HERE, "reports", "selection", "summary.csv"))
    with open(os.path.join(HERE, "reports", "grade", "stage_A_results.json"),
              encoding="utf-8") as fh:
        res_a = json.load(fh)
    with open(os.path.join(HERE, "reports", "grade", "stage_C_results.json"),
              encoding="utf-8") as fh:
        res_c = json.load(fh)

    print("figures:")
    fig_noise(sel)
    fig_cost(res_a)
    fig_equity(os.path.join(HERE, "reports", "grade", "returns_10bps.csv"))
    fig_estimators(res_c)


if __name__ == "__main__":
    main()
