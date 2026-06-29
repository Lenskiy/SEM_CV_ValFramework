import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def remove_baseline(signal, method="als", **kwargs):
    """
    Remove baseline from a sensor signal where the baseline follows
    the UPPER envelope (signal dips below baseline on exposure).

    Parameters
    ----------
    signal : array-like
        The input sensor signal.
    method : str
        'als'    – Asymmetric Least Squares smoothing
        'poly'   – Polynomial fit to upper envelope
        'median' – Rolling-median subtraction

    Keyword args (method-dependent)
    --------------------------------
    als:    lam (1e6)  – smoothness; p (0.99) – asymmetry weight; n_iter (10)
    poly:   degree (3) – polynomial order; quantile (0.95)
    median: window (51) – filter window size

    Returns
    -------
    corrected : np.ndarray
        Signal with baseline removed (dips become negative valleys).
    baseline  : np.ndarray
        The estimated upper baseline.
    """
    y = np.asarray(signal, dtype=float)

    if method == "als":
        baseline = _als_baseline_upper(y, **kwargs)
    elif method == "poly":
        baseline = _poly_baseline_upper(y, **kwargs)
    elif method == "median":
        baseline = _median_baseline_upper(y, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return y - baseline, baseline


# ── ALS fitted to the UPPER envelope ───────────────────────────────
def _als_baseline_upper(y, lam=1e6, p=0.99, n_iter=10):
    """
    Standard ALS but with p close to 1 so the fit is pulled toward
    the tops of the signal rather than the bottoms.

    p = 0.99 means: points ABOVE the current estimate are penalised
    99x more than points below → the fit hugs the upper envelope.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + H
        z = spsolve(Z, w * y)
        # High weight where y > z  → baseline is pushed UP
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


# ── Polynomial fit to upper envelope via quantile weighting ────────
def _poly_baseline_upper(y, degree=3, quantile=0.95, n_iter=5):
    """
    Iteratively reweighted polynomial fit: after each iteration,
    points well below the fit are down-weighted so the polynomial
    converges on the upper quantile of the signal.
    """
    x = np.arange(len(y))
    w = np.ones(len(y))
    for _ in range(n_iter):
        coeffs = np.polyfit(x, y, degree, w=w)
        baseline = np.polyval(coeffs, x)
        residuals = y - baseline
        # Keep high weight for points near or above the fit
        w = np.where(residuals > 0, quantile, 1 - quantile)
    return baseline


# ── Rolling maximum (mirrors the rolling-median idea) ──────────────
def _median_baseline_upper(y, window=51):
    """
    Uses a rolling maximum to track the upper envelope, then smooths
    it with a uniform filter to avoid a jagged baseline.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    upper = maximum_filter1d(y, size=window, mode="reflect")
    return uniform_filter1d(upper, size=window, mode="reflect")