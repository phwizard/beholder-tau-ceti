"""Joint Bayesian RV model for emcee sampling.

Model
-----

    y_{i, inst} = Σ_p  rv_kep(t_i; P_p, K_p, e_p, ω_p, T0_p)
                  + offset_inst
                  + GP_activity(t_i)
                  + ε_{i, inst}

where ε ~ N(0, σ_eff) with σ_eff² = yerr_i² + σ_jit_inst². The activity
GP is shared across instruments (same star); per-instrument offsets and
jitters are free parameters.

Parameter packing (theta, len = 5·n_planets + 2·n_inst):
  Per planet:
    log_P, log_K, sec = √e cos ω, ses = √e sin ω, T0
  Per instrument:
    offset, log_jitter

The GP hyperparameters are held fixed (taken from milestone 04 fits) so the
sampler explores only the planet + instrument-systematics space. Releasing
the GP is milestone 06+.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import celerite2
import numpy as np

from .activity import QPParams, build_kernel
from .keplerian import multi_keplerian_rv


@dataclass
class JointModelData:
    t: np.ndarray
    y: np.ndarray
    yerr: np.ndarray
    inst_idx: np.ndarray         # int per epoch
    inst_names: list[str] = field(default_factory=list)


@dataclass
class PlanetPrior:
    """Hyperparameters for a per-planet Gaussian / Beta prior."""
    log_P_mean: float
    log_P_sigma: float
    log_K_mean: float
    log_K_sigma: float
    T0_mean: float
    T0_sigma: float
    e_alpha: float = 2.0    # Beta α
    e_beta: float = 5.0     # Beta β


@dataclass
class InstPrior:
    offset_mean: float = 0.0
    offset_sigma: float = 50.0
    log_jitter_mean: float = 0.0      # log(1.0 m/s)
    log_jitter_sigma: float = 1.0


@dataclass
class GPPrior:
    """Gaussian priors on the seven QP-GP hyperparameters (log space).

    Defaults are loosely informative around the values found in milestone 04
    and the literature for late-G dwarfs. Hard bounds are enforced inside
    log_prior to keep the rotation period and Matern32 length-scale
    physically motivated.
    """
    log_sigma:        tuple[float, float] = (0.0, 2.0)
    log_period:       tuple[float, float] = (float(np.log(46.0)), 0.3)
    log_Q0:           tuple[float, float] = (0.0, 1.0)
    log_dQ:           tuple[float, float] = (float(np.log(0.01)), 1.0)
    log_f:            tuple[float, float] = (0.0, 2.0)
    log_rho_long:     tuple[float, float] = (float(np.log(2000.0)), 1.0)
    log_sigma_long:   tuple[float, float] = (0.0, 2.0)
    # Hard bounds (rejection)
    log_period_bounds:     tuple[float, float] = (float(np.log(25.0)), float(np.log(80.0)))
    log_rho_long_bounds:   tuple[float, float] = (float(np.log(500.0)), float(np.log(8000.0)))


class JointRVModel:
    """Joint multi-planet RV model with a shared activity GP."""

    def __init__(
        self,
        data: JointModelData,
        n_planets: int,
        gp_params: QPParams,
        gp_mean: float = 0.0,
        planet_priors: list[PlanetPrior] | None = None,
        inst_prior: InstPrior | None = None,
        fix_gp: bool = True,
        gp_prior: "GPPrior | None" = None,
    ):
        self.data = data
        self.n_planets = n_planets
        self.n_inst = len(data.inst_names)
        if self.n_inst == 0:
            self.n_inst = int(np.max(data.inst_idx)) + 1
        self.gp_params = gp_params
        self.gp_mean = gp_mean
        self.fix_gp = fix_gp
        self.gp_prior = gp_prior or GPPrior()
        self.planet_priors = planet_priors or [
            PlanetPrior(np.log(20.0), 0.05, np.log(0.5), 1.0, 53000.0, 5000.0)
            for _ in range(n_planets)
        ]
        self.inst_prior = inst_prior or InstPrior()

        order = np.argsort(data.t)
        self.t_sorted = data.t[order].astype(float).copy()
        self.y_sorted = data.y[order].astype(float)
        self.yerr_sorted = data.yerr[order].astype(float)
        self.inst_sorted = data.inst_idx[order].astype(int)

        # celerite2 needs strictly increasing t — break ties with epsilon offsets.
        for i in range(1, len(self.t_sorted)):
            if self.t_sorted[i] <= self.t_sorted[i - 1]:
                self.t_sorted[i] = self.t_sorted[i - 1] + 1e-6

        self._kernel = build_kernel(gp_params)
        self._gp = celerite2.GaussianProcess(self._kernel, mean=0.0)

    @property
    def n_dim(self) -> int:
        return 5 * self.n_planets + 2 * self.n_inst + (0 if self.fix_gp else 7)

    def unpack(self, theta: np.ndarray):
        idx = 0
        planets: list[dict] = []
        for _ in range(self.n_planets):
            log_P = float(theta[idx]); idx += 1
            log_K = float(theta[idx]); idx += 1
            sec = float(theta[idx]); idx += 1
            ses = float(theta[idx]); idx += 1
            T0 = float(theta[idx]); idx += 1
            e = sec * sec + ses * ses
            omega = float(np.arctan2(ses, sec))
            planets.append(
                {"P": np.exp(log_P), "K": np.exp(log_K), "e": e, "omega": omega, "T0": T0}
            )
        offsets = np.array(theta[idx : idx + self.n_inst], dtype=float)
        idx += self.n_inst
        log_jitters = np.array(theta[idx : idx + self.n_inst], dtype=float)
        idx += self.n_inst
        if self.fix_gp:
            gp = self.gp_params
        else:
            gp = QPParams(
                log_sigma=float(theta[idx + 0]),
                log_period=float(theta[idx + 1]),
                log_Q0=float(theta[idx + 2]),
                log_dQ=float(theta[idx + 3]),
                log_f=float(theta[idx + 4]),
                log_rho_long=float(theta[idx + 5]),
                log_sigma_long=float(theta[idx + 6]),
            )
        return planets, offsets, log_jitters, gp

    def log_prior(self, theta: np.ndarray) -> float:
        try:
            planets, offsets, log_jitters, gp = self.unpack(theta)
        except Exception:
            return -np.inf
        lp = 0.0
        for p, prior in zip(planets, self.planet_priors):
            log_P = float(np.log(p["P"]))
            log_K = float(np.log(p["K"]))
            lp += -0.5 * ((log_P - prior.log_P_mean) / prior.log_P_sigma) ** 2
            lp += -0.5 * ((log_K - prior.log_K_mean) / prior.log_K_sigma) ** 2
            e = p["e"]
            if not 0.0 <= e < 0.95:
                return -np.inf
            if e > 1e-12:
                lp += (prior.e_alpha - 1.0) * np.log(e) + (prior.e_beta - 1.0) * np.log(1.0 - e)
            lp += -0.5 * ((p["T0"] - prior.T0_mean) / prior.T0_sigma) ** 2
        ip = self.inst_prior
        for i in range(self.n_inst):
            lp += -0.5 * ((offsets[i] - ip.offset_mean) / ip.offset_sigma) ** 2
            lp += -0.5 * ((log_jitters[i] - ip.log_jitter_mean) / ip.log_jitter_sigma) ** 2
        if not self.fix_gp:
            g = self.gp_prior
            lp_lo, lp_hi = g.log_period_bounds
            if not (lp_lo <= gp.log_period <= lp_hi):
                return -np.inf
            rl_lo, rl_hi = g.log_rho_long_bounds
            if not (rl_lo <= gp.log_rho_long <= rl_hi):
                return -np.inf
            for name, val in [
                ("log_sigma", gp.log_sigma),
                ("log_period", gp.log_period),
                ("log_Q0", gp.log_Q0),
                ("log_dQ", gp.log_dQ),
                ("log_f", gp.log_f),
                ("log_rho_long", gp.log_rho_long),
                ("log_sigma_long", gp.log_sigma_long),
            ]:
                m, s = getattr(g, name)
                lp += -0.5 * ((val - m) / s) ** 2
        if not np.isfinite(lp):
            return -np.inf
        return float(lp)

    def log_likelihood(self, theta: np.ndarray) -> float:
        try:
            planets, offsets, log_jitters, gp = self.unpack(theta)
        except Exception:
            return -np.inf
        try:
            rv_planet = multi_keplerian_rv(self.t_sorted, planets)
        except Exception:
            return -np.inf
        offset_at_pt = offsets[self.inst_sorted]
        jit2 = np.exp(2.0 * log_jitters[self.inst_sorted])
        yerr_total = np.sqrt(self.yerr_sorted ** 2 + jit2)
        residuals = self.y_sorted - self.gp_mean - rv_planet - offset_at_pt
        try:
            if self.fix_gp:
                gp_obj = self._gp
            else:
                kernel = build_kernel(gp)
                gp_obj = celerite2.GaussianProcess(kernel, mean=0.0)
            gp_obj.compute(self.t_sorted, yerr=yerr_total)
            ll = gp_obj.log_likelihood(residuals)
            if not np.isfinite(ll):
                return -np.inf
            return float(ll)
        except Exception:
            return -np.inf

    def log_posterior(self, theta: np.ndarray) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def predict_planets(self, theta: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
        """Return the deterministic planet contribution at requested times."""
        planets, _, _, _ = self.unpack(theta)
        if t is None:
            t = self.t_sorted
        return multi_keplerian_rv(np.asarray(t), planets)


def emcee_initial_walkers(theta0: np.ndarray, nwalkers: int, scatter: float = 1e-3) -> np.ndarray:
    """Walker init: theta0 + small Gaussian scatter."""
    rng = np.random.default_rng(0)
    return theta0[None, :] + scatter * rng.standard_normal((nwalkers, len(theta0)))
