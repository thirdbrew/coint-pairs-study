"""
test_lookahead -- the probe that catches future information leaking backwards.

WHY THIS FILE EXISTS. The first version of hedge_ratio.kalman_beta fit its
variance parameters by maximum likelihood over formation+trading concatenated.
Its filtered states were causal GIVEN those parameters, so it looked right, read
right, and passed review. Perturbing only the LAST trading day moved betas on
EARLIER days by 3.8e-02. Code review did not catch it. This probe did, on the
first run, before any result had been produced from it.

THE INVARIANT. For any causal estimator, changing the data at time k must not
change any output at time < k. So: run the estimator, perturb a suffix of the
input, run it again, and assert the prefix is bit-identical. That test does not
care how the estimator works and cannot be fooled by a plausible-looking
implementation, which is exactly why it is worth more here than reading the code.

Run:  python test_lookahead.py     (or pytest test_lookahead.py)
"""

import numpy as np

import hedge_ratio as H

TOL = 1e-12


def _series(seed=7, n_f=252, n_t=126, beta=1.5):
    """Cointegrated pair: lb is a random walk, la = alpha + beta*lb + stationary noise."""
    rng = np.random.default_rng(seed)
    lb = np.cumsum(rng.normal(0, 0.015, n_f + n_t)) + 4.0
    la = 2.0 + beta * lb + rng.normal(0, 0.02, n_f + n_t)
    return la[:n_f], lb[:n_f], la[n_f:], lb[n_f:]


def _assert_prefix_unchanged(name, fn, f_la, f_lb, t_la, t_lb, cut, bump=5.0):
    """Perturb trading days >= cut; every output before cut must be identical."""
    perturbed = t_la.copy()
    perturbed[cut:] += bump

    b1, a1 = fn(f_la, f_lb, t_la, t_lb)
    b2, a2 = fn(f_la, f_lb, perturbed, t_lb)

    db = np.nanmax(np.abs(np.asarray(b1)[:cut] - np.asarray(b2)[:cut])) if cut else 0.0
    da = np.nanmax(np.abs(np.asarray(a1)[:cut] - np.asarray(a2)[:cut])) if cut else 0.0
    assert db <= TOL, (
        f"{name}: LOOKAHEAD -- perturbing trading day {cut} onward moved beta on "
        f"days 0..{cut - 1} by {db:.3e}. A causal estimator cannot do this.")
    assert da <= TOL, (
        f"{name}: LOOKAHEAD -- perturbing trading day {cut} onward moved alpha on "
        f"days 0..{cut - 1} by {da:.3e}.")
    return db, da


def test_estimators_are_causal():
    f_la, f_lb, t_la, t_lb = _series()
    n_t = len(t_la)
    for name, fn in H.ESTIMATORS.items():
        for cut in (n_t - 1, 100, 60, 30, 1):
            _assert_prefix_unchanged(name, fn, f_la, f_lb, t_la, t_lb, cut)


def test_estimators_recover_known_beta():
    """A causal estimator that recovers nothing is causal and useless. Both matter."""
    f_la, f_lb, t_la, t_lb = _series(beta=1.5)
    for name, fn in H.ESTIMATORS.items():
        b, _ = fn(f_la, f_lb, t_la, t_lb)
        got = float(np.nanmean(b))
        assert abs(got - 1.5) < 0.15, f"{name}: beta {got:.4f}, expected ~1.50"
        assert len(b) == len(t_la), f"{name}: returned {len(b)} betas for {len(t_la)} days"


def test_smoother_would_fail_this_probe():
    """Guard the guard: a deliberately non-causal estimator must be CAUGHT.

    A probe that passes everything is not a probe. This builds a full-sample
    estimator -- the mistake the real one nearly shipped -- and asserts the
    invariant fires on it.
    """
    def full_sample_ols(f_la, f_lb, t_la, t_lb):
        la = np.concatenate([f_la, t_la])
        lb = np.concatenate([f_lb, t_lb])
        ok = np.isfinite(la) & np.isfinite(lb)
        X = np.column_stack([np.ones(ok.sum()), lb[ok]])
        coef, *_ = np.linalg.lstsq(X, la[ok], rcond=None)
        n = len(t_la)
        return np.full(n, float(coef[1])), np.full(n, float(coef[0]))

    f_la, f_lb, t_la, t_lb = _series()
    try:
        _assert_prefix_unchanged("full_sample_ols", full_sample_ols,
                                 f_la, f_lb, t_la, t_lb, cut=60)
    except AssertionError:
        return
    raise AssertionError(
        "the probe did NOT catch a full-sample estimator, so it is not testing "
        "what it claims to test")


if __name__ == "__main__":
    f_la, f_lb, t_la, t_lb = _series()
    n_t = len(t_la)
    print("causality probe -- max |delta| on the untouched prefix\n")
    print(f"{'estimator':14s} " + " ".join(f"cut={c:<4d}" for c in (n_t - 1, 100, 60, 30, 1)))
    for name, fn in H.ESTIMATORS.items():
        cells = []
        for cut in (n_t - 1, 100, 60, 30, 1):
            db, _ = _assert_prefix_unchanged(name, fn, f_la, f_lb, t_la, t_lb, cut)
            cells.append(f"{db:8.1e}")
        print(f"{name:14s} " + " ".join(cells))

    test_estimators_recover_known_beta()
    print("\nbeta recovery: OK")
    test_smoother_would_fail_this_probe()
    print("probe catches a deliberately non-causal estimator: OK")
    print("\nall lookahead tests passed")
