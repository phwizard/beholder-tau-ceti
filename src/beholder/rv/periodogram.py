"""Lomb-Scargle utilities for RV time series."""

from __future__ import annotations

import numpy as np
from astropy.timeseries import LombScargle


def lomb_scargle(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray | None = None,
    min_period: float = 1.0,
    max_period: float = 4000.0,
    samples_per_peak: int = 10,
) -> tuple[LombScargle, np.ndarray, np.ndarray]:
    """Compute a Lomb-Scargle periodogram.

    Returns ``(ls_object, periods_d, power)``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if dy is not None:
        ls = LombScargle(t, y, dy=np.asarray(dy, dtype=float))
    else:
        ls = LombScargle(t, y)
    freq, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=samples_per_peak,
    )
    return ls, 1.0 / freq, power


def top_n_peaks(
    periods: np.ndarray,
    power: np.ndarray,
    n: int = 10,
    fractional_separation: float = 0.05,
) -> list[tuple[float, float]]:
    """Top-n peaks ``(period, power)`` separated by ≥ `fractional_separation`."""
    order = np.argsort(power)[::-1]
    out: list[tuple[float, float]] = []
    for i in order:
        p = float(periods[i])
        pw = float(power[i])
        if all(abs(p - pp) / max(pp, p) > fractional_separation for pp, _ in out):
            out.append((p, pw))
        if len(out) >= n:
            break
    return out


def fap_baluev(
    ls: LombScargle,
    power: float,
    max_frequency: float = 1.0,
) -> float:
    """Baluev approximation to the false alarm probability of a peak."""
    return float(ls.false_alarm_probability(power, method="baluev", maximum_frequency=max_frequency))
