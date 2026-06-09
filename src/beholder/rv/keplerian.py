"""Analytic Keplerian radial-velocity model.

The RV signal of a single planet on a Keplerian orbit is

    rv(t) = K * (cos(f(t) + ω) + e cos(ω))

where f is the true anomaly, computed from the eccentric anomaly E that
solves Kepler's equation E − e sin E = M = 2π (t − T₀)/P.
"""

from __future__ import annotations

import numpy as np


def solve_kepler(M: np.ndarray, e: float, tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
    """Newton-Raphson solver for Kepler's equation E − e sin(E) = M."""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    e = float(e)
    if not 0.0 <= e < 1.0:
        raise ValueError(f"e must be in [0, 1), got {e}")
    E = M.copy() if e < 0.8 else np.full_like(M, np.pi)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def keplerian_rv(
    t: np.ndarray,
    P: float,
    K: float,
    e: float,
    omega: float,
    T0: float,
) -> np.ndarray:
    """Radial-velocity contribution of a single planet on a Keplerian orbit."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if e < 1e-10:
        f = 2.0 * np.pi * (t - T0) / P
    else:
        M = 2.0 * np.pi * (t - T0) / P
        E = solve_kepler(M, e)
        f = 2.0 * np.arctan2(
            np.sqrt(1.0 + e) * np.sin(E / 2.0),
            np.sqrt(1.0 - e) * np.cos(E / 2.0),
        )
    return K * (np.cos(f + omega) + e * np.cos(omega))


def multi_keplerian_rv(t: np.ndarray, planets: list[dict]) -> np.ndarray:
    """Sum of Keplerian RVs from a list of planet dicts (P, K, e, omega, T0)."""
    rv = np.zeros_like(np.atleast_1d(t), dtype=float)
    for p in planets:
        rv = rv + keplerian_rv(t, **p)
    return rv
