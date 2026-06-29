import numpy as np
from scipy.signal import detrend


def estimate_noise(signal, baseline=None, method="mad", **kwargs):
    """
    Estimate the noise level of a sensor signal.

    Parameters
    ----------
    signal : array-like
        Raw sensor signal.
    baseline : array-like, optional
        Pre-computed baseline. If None, noise is estimated directly
        from the signal (assuming it's already baseline-corrected).
    method : str
        'mad'      – Median Absolute Deviation (robust to peaks/dips)
        'std'      – Standard deviation of quiet regions
        'wavelet'  – Wavelet-based (estimates noise from finest detail coeffs)
        'allan'    – Allan deviation (good for drift-dominated sensors)

    Keyword args
    ------------
    mad:     scale (1.4826) – factor to convert MAD to σ for Gaussian noise
    std:     quantile_range (0.25, 0.75) – fraction of sorted residuals
                 treated as "quiet" (excludes response dips and outliers)
    wavelet: wavelet ('db4') – pywt wavelet name
    allan:   tau (1) – averaging interval in sample units

    Returns
    -------
    noise_level : float
        Estimated RMS / σ of the noise.
    noise_trace : np.ndarray
        Per-sample residual used for the estimate (useful for plotting).
    """
    y = np.asarray(signal, dtype=float)

    if baseline is not None:
        residual = y - np.asarray(baseline, dtype=float)
    else:
        residual = y.copy()

    if method == "mad":
        noise = _noise_mad(residual, **kwargs)
    elif method == "std":
        noise = _noise_std(residual, **kwargs)
    elif method == "wavelet":
        noise = _noise_wavelet(y, **kwargs)
    elif method == "allan":
        noise = _noise_allan(residual, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return noise, residual


# ── MAD (robust σ estimate) ─────────────────────────────────────────
def _noise_mad(residual, scale=1.4826):
    """
    Median Absolute Deviation scaled to equivalent Gaussian σ.
    Very robust: unaffected by sparse large dips (gas responses).
    """
    med = np.median(residual)
    return scale * np.median(np.abs(residual - med))


# ── Std of quiet region ─────────────────────────────────────────────
def _noise_std(residual, quantile_range=(0.25, 0.75)):
    """
    Standard deviation computed only on the 'quiet' middle portion
    of the residual distribution, excluding response dips and spikes.
    """
    lo, hi = np.quantile(residual, quantile_range)
    quiet = residual[(residual >= lo) & (residual <= hi)]
    return np.std(quiet)


# ── Wavelet-based (Donoho & Johnstone) ──────────────────────────────
def _noise_wavelet(signal, wavelet="db4"):
    """
    Uses the finest-scale wavelet detail coefficients, which are
    dominated by noise rather than signal structure.
    """
    import pywt
    coeffs = pywt.wavedec(signal, wavelet, level=1)
    detail = coeffs[-1]  # finest level
    return 1.4826 * np.median(np.abs(detail - np.median(detail)))


# ── Allan deviation ─────────────────────────────────────────────────
def _noise_allan(residual, tau=1):
    """
    Allan deviation at a given averaging interval τ.
    Separates white noise from drift; useful for sensor stability
    characterisation.
    """
    n = len(residual) // tau
    trimmed = residual[: n * tau].reshape(n, tau).mean(axis=1)
    diffs = np.diff(trimmed)
    return np.sqrt(0.5 * np.mean(diffs ** 2))