"""
hedge_ratio -- stage C. Three estimators of beta, compared by Model Confidence Set.

THIS IS A MODEL COMPARISON, NOT A MECHANISM HUNT. The distinction matters and is
the reason this stage is in the project at all. Hunting means trying estimators
until one beats a bar and reporting that one. Comparing means specifying the
candidate set up front and letting stats_lib.model_confidence_set return the
subset that cannot be distinguished from the best -- which may well be all three,
and "all three, indistinguishable" is a result. The parent repo has four
consecutive pre-registered failures on record from the hunting version of this.

Each estimator has the same signature, the seam trade.pair_returns expects:

    beta_fn(f_la, f_lb, t_la, t_lb) -> (beta_t, alpha_t)

both arrays aligned to the TRADING window.

THE LOOKAHEAD LINE RUNS THROUGH THE MIDDLE OF THIS FILE. A Kalman SMOOTHER
estimates the state at time t using the whole sample including the future; a
Kalman FILTER uses data through t only. The smoother produces a visibly better
hedge ratio and a completely invalid backtest. This module uses filtered states
everywhere, and rolling OLS is likewise computed on a trailing window that ends
at t. trade.pair_returns then lags positions one further day before applying
returns, so nothing trades on information from its own bar.
"""

import numpy as np
import statsmodels.api as sm


# ------------------------------------------------------------ 1. static OLS
def static_ols(f_la, f_lb, t_la, t_lb):
    """Formation-window OLS, frozen for the whole trading window. Stage A's estimator."""
    ok = np.isfinite(f_la) & np.isfinite(f_lb)
    X = np.column_stack([np.ones(ok.sum()), f_lb[ok]])
    coef, *_ = np.linalg.lstsq(X, f_la[ok], rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    n = len(t_la)
    return np.full(n, beta), np.full(n, alpha)


# ----------------------------------------------------------- 2. rolling OLS
def rolling_ols(f_la, f_lb, t_la, t_lb, lookback=60):
    """Trailing-window OLS, re-fit each trading day on the previous `lookback` bars.

    The window ENDS at t and the concatenated history starts in the formation
    window, so the first trading day has a full lookback rather than a stub.
    """
    la = np.concatenate([f_la, t_la])
    lb = np.concatenate([f_lb, t_lb])
    off = len(f_la)
    n = len(t_la)
    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)

    for k in range(n):
        end = off + k + 1                      # inclusive of today, never beyond
        seg_a = la[max(0, end - lookback):end]
        seg_b = lb[max(0, end - lookback):end]
        ok = np.isfinite(seg_a) & np.isfinite(seg_b)
        if ok.sum() < 20:
            continue
        X = np.column_stack([np.ones(ok.sum()), seg_b[ok]])
        coef, *_ = np.linalg.lstsq(X, seg_a[ok], rcond=None)
        alpha[k], beta[k] = float(coef[0]), float(coef[1])

    beta = _ffill(beta)
    alpha = _ffill(alpha)
    return beta, alpha


def _ffill(a):
    out = a.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    if not np.isfinite(out[0]):
        first = next((v for v in out if np.isfinite(v)), 0.0)
        out[~np.isfinite(out)] = first
    return out


# --------------------------------------------------- 3. Kalman random-walk beta
class TVPRegression(sm.tsa.statespace.MLEModel):
    """Time-varying-parameter regression: la_t = alpha_t + beta_t*lb_t + eps_t,
    with [alpha_t, beta_t] following a random walk.

    Written against statsmodels' state-space machinery rather than pykalman,
    which has been unmaintained since 2013.
    """

    param_names = ["sigma2.obs", "sigma2.alpha", "sigma2.beta"]

    def __init__(self, endog, exog):
        super().__init__(endog, k_states=2, k_posdef=2,
                         initialization="approximate_diffuse")
        exog = np.asarray(exog, dtype=float)
        self["design"] = np.zeros((1, 2, self.nobs))
        self["design"][0, 0, :] = 1.0
        self["design"][0, 1, :] = exog
        self["transition"] = np.eye(2)
        self["selection"] = np.eye(2)

    @property
    def start_params(self):
        return np.array([np.nanvar(self.endog) * 0.5, 1e-6, 1e-6])

    def transform_params(self, unconstrained):
        return unconstrained ** 2

    def untransform_params(self, constrained):
        return np.sqrt(np.abs(constrained))

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self["obs_cov", 0, 0] = params[0]
        self["state_cov"] = np.diag(params[1:])


def kalman_beta(f_la, f_lb, t_la, t_lb, maxiter=50):
    """Fit variances on the formation window ONLY, then filter the full series.

    TWO SEPARATE LOOKAHEAD RISKS LIVE HERE AND BOTH HAVE TO BE CLOSED.

    1. State lookahead -- using the SMOOTHER, whose state at t conditions on the
       whole sample. Closed by reading `filtered_state`, never `smoothed_state`.

    2. PARAMETER lookahead -- fitting the variance parameters by maximum
       likelihood over formation+trading. The states are then causal given the
       parameters, but the parameters saw the future, so the whole path is
       contaminated. This is subtler than (1) and it is what the first version
       of this function actually did: perturbing only the LAST trading day moved
       betas on EARLIER days by 3.8e-02. Closed by fitting on the formation
       window and passing those parameters to `.filter()`, which runs the
       recursion without re-estimating anything.

    Falls back to static OLS when the likelihood fails to converge -- that
    happens on near-degenerate pairs and must not silently become a flat position.
    """
    la = np.concatenate([f_la, t_la])
    lb = np.concatenate([f_lb, t_lb])
    off = len(f_la)

    if (np.isfinite(la) & np.isfinite(lb)).sum() < 100:
        return static_ols(f_la, f_lb, t_la, t_lb)

    f_ok = np.isfinite(f_la) & np.isfinite(f_lb)
    if f_ok.sum() < 60:
        return static_ols(f_la, f_lb, t_la, t_lb)

    try:
        # (a) parameters from the formation window alone
        fit = TVPRegression(np.where(f_ok, f_la, np.nan),
                            np.where(f_ok, f_lb, 0.0)).fit(disp=False, maxiter=maxiter)

        # (b) filter the full series with those parameters held FIXED
        ok = np.isfinite(la) & np.isfinite(lb)
        res = TVPRegression(np.where(ok, la, np.nan),
                            np.where(ok, lb, 0.0)).filter(fit.params)

        state = np.asarray(res.filtered_state)          # (2, nobs), causal
        alpha = _ffill(state[0, off:].astype(float))
        beta = _ffill(state[1, off:].astype(float))
        if not np.all(np.isfinite(beta)):
            raise ValueError("non-finite filtered beta")
        return beta, alpha
    except Exception:
        return static_ols(f_la, f_lb, t_la, t_lb)


ESTIMATORS = {
    "static_ols": static_ols,
    "rolling_ols": rolling_ols,
    "kalman": kalman_beta,
}
