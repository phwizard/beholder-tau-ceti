"""Quasi-periodic GP activity decorrelation for RV time series.

Two-step decorrelation:
  1. Fit a (RotationTerm + Matern32) GP to the log R'HK time series — this
     is the cleanest activity tracer.
  2. Apply the same kernel structure to the RV channel, refit amplitudes,
     predict the activity-induced RV at each observation epoch, and subtract.

This is a heuristic decorrelation (not a fully joint multi-output GP per
Rajpaul+ 2015) but captures the dominant stellar-activity contribution to RV
with O(n) celerite2 scaling. It is intended as the cleanest "next step"
before milestone 05's full Bayesian model.

The kernel is a celerite2 ``RotationTerm`` (a sum of two SHO terms that
approximate a quasi-periodic kernel) plus a ``Matern32Term`` to absorb the
~11-yr magnetic cycle that single rotation periodicity cannot fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import celerite2
import numpy as np
import pandas as pd
from celerite2 import terms
from scipy.optimize import minimize


@dataclass
class QPParams:
    log_sigma: float
    log_period: float
    log_Q0: float
    log_dQ: float
    log_f: float           # converted via sigmoid → f ∈ (0, 1)
    log_rho_long: float    # Matern32 length scale
    log_sigma_long: float  # Matern32 amplitude


def _f_from_log(log_f: float) -> float:
    return float(1.0 / (1.0 + np.exp(-log_f)))


def _params_to_array(p: QPParams) -> np.ndarray:
    return np.array(
        [p.log_sigma, p.log_period, p.log_Q0, p.log_dQ, p.log_f,
         p.log_rho_long, p.log_sigma_long]
    )


def _array_to_params(a: np.ndarray) -> QPParams:
    return QPParams(*[float(x) for x in a])


def build_kernel(p: QPParams):
    rotation = terms.RotationTerm(
        sigma=float(np.exp(p.log_sigma)),
        period=float(np.exp(p.log_period)),
        Q0=float(np.exp(p.log_Q0)),
        dQ=float(np.exp(p.log_dQ)),
        f=_f_from_log(p.log_f),
    )
    long_term = terms.Matern32Term(
        sigma=float(np.exp(p.log_sigma_long)),
        rho=float(np.exp(p.log_rho_long)),
    )
    return rotation + long_term


def _make_gp(p: QPParams, t: np.ndarray, yerr: np.ndarray):
    kernel = build_kernel(p)
    gp = celerite2.GaussianProcess(kernel, mean=0.0)
    gp.compute(t, yerr=yerr)
    return gp


# Default bounds in log-space. Set narrow enough to prevent the optimizer
# from collapsing the rotation kernel onto the 1-yr alias or pushing the
# Matern32 length scale to a single sample step.
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "log_sigma":      (float(np.log(1e-4)),  float(np.log(1e3))),
    "log_period":     (float(np.log(25.0)),  float(np.log(80.0))),
    "log_Q0":         (float(np.log(0.5)),   float(np.log(50.0))),
    "log_dQ":         (float(np.log(1e-4)),  float(np.log(10.0))),
    "log_f":          (-4.0,                 4.0),  # sigmoid → ~0.02..0.98
    "log_rho_long":   (float(np.log(500.0)), float(np.log(8000.0))),
    "log_sigma_long": (float(np.log(1e-4)),  float(np.log(1e3))),
}


def fit_qp_mle(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    log_period_init: float = float(np.log(46.0)),
    log_rho_long_init: float = float(np.log(2000.0)),
    bounds: dict[str, tuple[float, float]] | None = None,
    maxiter: int = 2000,
) -> tuple[QPParams, float, dict]:
    """Fit RotationTerm + Matern32 to (t, y, yerr) by bounded MLE (L-BFGS-B).

    Returns ``(params, mean, info)``. The fit is performed on (y - mean), so
    the prediction must add the mean back.

    Bounds are necessary in practice — without them the rotation period
    collapses to the 1-yr sampling alias for ground-based RV. Defaults are
    deliberately narrow around physical priors (P_rot ∈ [25, 80] d for late-G
    dwarfs, ρ_long ∈ [500, 8000] d to allow magnetic-cycle capture).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    mean = float(np.median(y))
    y_centered = y - mean
    sigma_init = float(np.log(max(np.std(y_centered), 1e-6)))

    p0 = QPParams(
        log_sigma=sigma_init,
        log_period=log_period_init,
        log_Q0=float(np.log(1.0)),
        log_dQ=float(np.log(0.01)),
        log_f=0.0,
        log_rho_long=log_rho_long_init,
        log_sigma_long=sigma_init,
    )

    bnd = dict(DEFAULT_BOUNDS)
    if bounds is not None:
        bnd.update(bounds)
    bounds_array = [
        bnd["log_sigma"],
        bnd["log_period"],
        bnd["log_Q0"],
        bnd["log_dQ"],
        bnd["log_f"],
        bnd["log_rho_long"],
        bnd["log_sigma_long"],
    ]

    def nll(arr: np.ndarray) -> float:
        p = _array_to_params(arr)
        try:
            gp = _make_gp(p, t, yerr)
            ll = gp.log_likelihood(y_centered)
            if not np.isfinite(ll):
                return 1e30
            return float(-ll)
        except Exception:
            return 1e30

    result = minimize(
        nll,
        _params_to_array(p0),
        method="L-BFGS-B",
        bounds=bounds_array,
        options={"maxiter": maxiter, "ftol": 1e-7, "gtol": 1e-5},
    )
    return _array_to_params(result.x), mean, {
        "success": bool(result.success),
        "fun": float(result.fun),
        "nit": int(result.nit),
        "message": str(getattr(result, "message", "")),
    }


def predict(
    p: QPParams,
    t_train: np.ndarray,
    y_train: np.ndarray,
    yerr_train: np.ndarray,
    mean: float,
    t_pred: np.ndarray | None = None,
    return_var: bool = False,
):
    """Predict GP at ``t_pred`` (or training times if ``None``)."""
    t_train = np.asarray(t_train, float)
    yerr_train = np.asarray(yerr_train, float)
    y_centered = np.asarray(y_train, float) - mean
    if t_pred is None:
        t_pred = t_train
    t_pred = np.asarray(t_pred, float)
    gp = _make_gp(p, t_train, yerr_train)
    if return_var:
        mu, var = gp.predict(y_centered, t=t_pred, return_var=True)
        return mu + mean, var
    return gp.predict(y_centered, t=t_pred) + mean


def bin_nightly(
    df: pd.DataFrame,
    time_col: str = "time_rjd",
    pairs: tuple[tuple[str, str], ...] = (
        ("rv", "rv_err"),
        ("rv_centered", "rv_err"),
        ("spectro_rhk", "spectro_rhk_err"),
        ("spectro_smw", "spectro_smw_err"),
        ("ccf_fwhm", "ccf_fwhm_err"),
        ("ccf_bispan", "ccf_bispan_err"),
    ),
    night_offset: float = 0.5,
) -> pd.DataFrame:
    """Inverse-variance weighted nightly binning.

    `night_offset` shifts the day-boundary so observations across UT midnight
    on the same observing night land in the same bin. 0.5 is appropriate for
    sites where local midnight is roughly UT noon-ish — Chile, La Palma. Not
    perfect everywhere, but good enough for activity-decorrelation purposes.
    """
    df = df.copy()
    df = df.dropna(subset=[time_col])
    df["_night"] = np.floor(df[time_col].astype(float) - night_offset).astype(int)

    rows: list[dict] = []
    for night, sub in df.groupby("_night"):
        row: dict = {
            "_night": int(night),
            time_col: float(sub[time_col].mean()),
            "n_in_bin": int(len(sub)),
        }
        for v, e in pairs:
            if v not in sub.columns:
                continue
            mask = sub[v].notna()
            if e in sub.columns:
                mask &= sub[e].notna() & (sub[e] > 0)
            if not mask.any():
                row[v] = np.nan
                if e in sub.columns:
                    row[e] = np.nan
                continue
            vals = sub.loc[mask, v].to_numpy(dtype=float)
            if e in sub.columns:
                errs = sub.loc[mask, e].to_numpy(dtype=float)
            else:
                errs = np.ones_like(vals)
            w = 1.0 / errs**2
            row[v] = float(np.sum(w * vals) / np.sum(w))
            if e in sub.columns:
                row[e] = float(1.0 / np.sqrt(np.sum(w)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(time_col).reset_index(drop=True)
