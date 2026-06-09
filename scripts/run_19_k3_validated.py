"""Milestone 19 — rerun k=3 (g+h+e) at nlive=2000 for converged BF vs k=4.

m13 ran the full evidence ladder at nlive=500. m17 showed the resulting
posteriors were 25-100× too narrow, and log Z values were biased ~6 nats
high. Specifically m13's BF(k=3/k=4) = 3 (weak preference for k=3) might
not survive proper convergence — it could be sampler bias on each rung
canceling out.

The cleanest test: rerun k=3 at nlive=2000 (matching m17's k=4 setup).
The pair (m17 k=4 + m19 k=3) at the same nlive gives us the only
Bayes factor in the project that's fully converged.

If BF(k=3/k=4) > 5: k=3 preferred decisively. f is not needed.
If BF(k=3/k=4) ≈ 1: data is genuinely indifferent. k=3 and k=4 equally good.
If BF(k=3/k=4) < 0.2: k=4 preferred. f IS needed.

m13's verdict was BF=3 (k=3 preferred at 60% probability). m17 results
suggest this borderline number might shift.
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import dynesty
import dynesty.utils as dyutils
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, norm, truncnorm

from beholder.rv.activity import QPParams, bin_nightly
from beholder.rv.bayes import (
    GPPrior,
    InstPrior,
    JointModelData,
    JointRVModel,
    PlanetPrior,
)

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
FIGS = REPO / "notebooks" / "figures"

FENG_2017: list[tuple[str, float, float, float]] = [
    ("g", 19.98, 0.40, 53000.0),
    ("h", 49.00, 0.34, 53000.0),
    ("e", 162.9, 0.51, 53000.0),
    # f intentionally omitted
]
INSTRUMENTS = ["HARPS03", "HARPS15", "HIRES-PRE04", "HIRES-POST04", "ESPRESSO19", "APF"]

LOOSE_LOG_P_SIGMA = 0.3
LOG_P_ROT_BOUNDS = (float(np.log(25.0)), float(np.log(80.0)))
GP_PRIOR_LOG_P_ROT = (float(np.log(46.0)), 0.3)
LOG_RHO_LONG_BOUNDS = (float(np.log(50.0)), float(np.log(8000.0)))
GP_PRIOR_LOG_RHO_LONG = (float(np.log(500.0)), 2.0)


def _load_data() -> JointModelData:
    df = pd.read_csv(PROCESSED / "rv_dace_tau_ceti.csv")
    df = df[df.instrument.isin(INSTRUMENTS)].copy()
    df = df.dropna(subset=["rv_centered", "rv_err", "time_rjd"])
    binned_per_inst = []
    for inst in INSTRUMENTS:
        sub = df[df.instrument == inst]
        if len(sub) == 0:
            continue
        b = bin_nightly(sub).dropna(subset=["rv_centered", "rv_err"]).reset_index(drop=True)
        b["instrument"] = inst
        binned_per_inst.append(b)
    nightly = pd.concat(binned_per_inst, ignore_index=True).sort_values("time_rjd").reset_index(drop=True)
    name_to_idx = {n: i for i, n in enumerate(INSTRUMENTS)}
    inst_idx = nightly["instrument"].map(name_to_idx).to_numpy().astype(int)
    return JointModelData(
        t=nightly["time_rjd"].to_numpy(),
        y=nightly["rv_centered"].to_numpy(),
        yerr=nightly["rv_err"].to_numpy(),
        inst_idx=inst_idx,
        inst_names=INSTRUMENTS,
    )


def _load_gp_init() -> tuple[QPParams, float]:
    gp_df = pd.read_csv(PROCESSED / "04_gp_hyperparameters.csv")
    gp_row = gp_df[gp_df.channel == "RV"].iloc[0]
    return (
        QPParams(
            log_sigma=float(gp_row.log_sigma),
            log_period=float(gp_row.log_period),
            log_Q0=float(gp_row.log_Q0),
            log_dQ=float(gp_row.log_dQ),
            log_f=float(gp_row.log_f),
            log_rho_long=float(gp_row.log_rho_long),
            log_sigma_long=float(gp_row.log_sigma_long),
        ),
        float(gp_row["mean"]),
    )


def _build_model_k3(data, gp_init, gp_mean) -> JointRVModel:
    planet_priors = [
        PlanetPrior(
            log_P_mean=float(np.log(P)),
            log_P_sigma=LOOSE_LOG_P_SIGMA,
            log_K_mean=float(np.log(0.5)),
            log_K_sigma=1.0,
            T0_mean=53000.0,
            T0_sigma=5000.0,
            e_alpha=2.0,
            e_beta=5.0,
        )
        for _, P, _, _ in FENG_2017
    ]
    inst_prior = InstPrior(offset_mean=0.0, offset_sigma=5.0,
                           log_jitter_mean=float(np.log(1.0)), log_jitter_sigma=1.0)
    return JointRVModel(
        data=data, n_planets=3, gp_params=gp_init, gp_mean=gp_mean,
        planet_priors=planet_priors, inst_prior=inst_prior, fix_gp=False,
        gp_prior=GPPrior(
            log_period=GP_PRIOR_LOG_P_ROT,
            log_rho_long=GP_PRIOR_LOG_RHO_LONG,
            log_period_bounds=LOG_P_ROT_BOUNDS,
            log_rho_long_bounds=LOG_RHO_LONG_BOUNDS,
        ),
    )


_logp_lo, _logp_hi = LOG_P_ROT_BOUNDS
_logp_mu, _logp_sigma = GP_PRIOR_LOG_P_ROT
_logp_a = (_logp_lo - _logp_mu) / _logp_sigma
_logp_b = (_logp_hi - _logp_mu) / _logp_sigma
_rho_lo, _rho_hi = LOG_RHO_LONG_BOUNDS
_rho_mu, _rho_sigma = GP_PRIOR_LOG_RHO_LONG
_rho_a = (_rho_lo - _rho_mu) / _rho_sigma
_rho_b = (_rho_hi - _rho_mu) / _rho_sigma


def make_prior_transform(model: JointRVModel):
    log_P_means = [pp.log_P_mean for pp in model.planet_priors]

    def prior_transform(u: np.ndarray) -> np.ndarray:
        theta = np.empty_like(u)
        idx = 0
        for i in range(model.n_planets):
            theta[idx] = norm.ppf(u[idx], loc=log_P_means[i], scale=LOOSE_LOG_P_SIGMA); idx += 1
            theta[idx] = norm.ppf(u[idx], loc=float(np.log(0.5)), scale=1.0); idx += 1
            e = beta.ppf(u[idx], 2.0, 5.0); e = min(e, 0.949)
            omega = 2.0 * np.pi * u[idx + 1]
            sqrt_e = np.sqrt(max(e, 0.0))
            theta[idx]     = sqrt_e * np.cos(omega)
            theta[idx + 1] = sqrt_e * np.sin(omega)
            idx += 2
            theta[idx] = norm.ppf(u[idx], loc=53000.0, scale=5000.0); idx += 1
        for _ in range(model.n_inst):
            theta[idx] = norm.ppf(u[idx], loc=0.0, scale=5.0); idx += 1
            theta[idx] = norm.ppf(u[idx], loc=0.0, scale=1.0); idx += 1
        theta[idx] = norm.ppf(u[idx], loc=0.0, scale=2.0); idx += 1
        theta[idx] = truncnorm.ppf(u[idx], _logp_a, _logp_b, loc=_logp_mu, scale=_logp_sigma); idx += 1
        theta[idx] = norm.ppf(u[idx], loc=0.0, scale=1.0); idx += 1
        theta[idx] = norm.ppf(u[idx], loc=float(np.log(0.01)), scale=1.0); idx += 1
        theta[idx] = norm.ppf(u[idx], loc=0.0, scale=2.0); idx += 1
        theta[idx] = truncnorm.ppf(u[idx], _rho_a, _rho_b, loc=_rho_mu, scale=_rho_sigma); idx += 1
        theta[idx] = norm.ppf(u[idx], loc=0.0, scale=2.0); idx += 1
        return theta

    return prior_transform


def main():
    data = _load_data()
    gp_init, gp_mean = _load_gp_init()
    model = _build_model_k3(data, gp_init, gp_mean)
    pt = make_prior_transform(model)
    print(f"n_dim = {model.n_dim} (k=3: g+h+e)")

    print("\n=== dynesty: k=3, nlive=2000, multi-mode ===")
    sampler = dynesty.DynamicNestedSampler(
        loglikelihood=model.log_likelihood,
        prior_transform=pt,
        ndim=model.n_dim,
        bound="multi",
        sample="rwalk",
        rstate=np.random.default_rng(2026),
    )
    t0 = time.time()
    sampler.run_nested(
        nlive_init=2000, nlive_batch=500, maxbatch=10,
        wt_kwargs={"pfrac": 1.0},
        print_progress=False,
    )
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.0f}s = {elapsed/60:.1f} min")

    results = sampler.results
    log_z_k3 = float(results.logz[-1])
    log_z_err_k3 = float(results.logzerr[-1])
    print(f"\nlog Z (k=3, nlive=2000) = {log_z_k3:.3f} ± {log_z_err_k3:.3f}")
    print(f"n_iters = {len(results.logl)}")

    samples = results.samples
    weights = results.importance_weights()
    posterior = dyutils.resample_equal(samples, weights)
    print(f"posterior samples (equal-weight): {posterior.shape}")

    np.savez(
        PROCESSED / "19_dynesty_k3_nlive2000.npz",
        samples=samples, weights=weights, posterior=posterior,
        logz=log_z_k3, logz_err=log_z_err_k3, logl=results.logl, elapsed_s=elapsed,
    )

    # Compare against m17 k=4 and m13 k=3
    m17 = np.load(PROCESSED / "17_dynesty_nlive2000.npz")
    log_z_k4_n2000 = float(m17["logz"])
    log_z_k4_err_n2000 = float(m17["logz_err"])

    m13 = pd.read_csv(PROCESSED / "13_evidence_ladder.csv")
    log_z_k3_m13 = float(m13.iloc[m13[m13.k == 3].index[0]].logz)
    log_z_k4_m13 = float(m13.iloc[m13[m13.k == 4].index[0]].logz)

    print("\n=== Validated k=3 vs k=4 (both at nlive=2000) ===")
    print(f"  log Z (m19 k=3, nlive=2000) = {log_z_k3:.3f} ± {log_z_err_k3:.3f}")
    print(f"  log Z (m17 k=4, nlive=2000) = {log_z_k4_n2000:.3f} ± {log_z_k4_err_n2000:.3f}")
    delta = log_z_k3 - log_z_k4_n2000
    bf = np.exp(delta)
    print(f"  Δ log Z (k=3 - k=4) = {delta:+.3f}")
    if delta > 0:
        print(f"  → k=3 preferred by BF {bf:.2g}")
    else:
        print(f"  → k=4 preferred by BF {1/bf:.2g}")

    print("\n=== For comparison: m13 (nlive=500, biased) ===")
    delta_m13 = log_z_k3_m13 - log_z_k4_m13
    print(f"  log Z (m13 k=3, nlive=500) = {log_z_k3_m13:.3f}")
    print(f"  log Z (m13 k=4, nlive=500) = {log_z_k4_m13:.3f}")
    print(f"  Δ log Z (k=3 - k=4) = {delta_m13:+.3f}  (m13 reported BF = 3.1)")

    print("\n=== Δlog Z shift between m13 and m19/m17 (nlive bias) ===")
    print(f"  k=3: m19 - m13 = {log_z_k3 - log_z_k3_m13:+.3f}")
    print(f"  k=4: m17 - m13 = {log_z_k4_n2000 - log_z_k4_m13:+.3f}")
    print(f"  → If the bias was equal across rungs, BF(k=3/k=4) at nlive=2000")
    print(f"    would match m13's. Difference here is {delta - delta_m13:+.3f}.")

    # Save comparison
    out = pd.DataFrame([
        {"source": "m13 k=3 (nlive=500)", "logz": log_z_k3_m13, "logz_err": float(m13.iloc[m13[m13.k == 3].index[0]].logz_err)},
        {"source": "m13 k=4 (nlive=500)", "logz": log_z_k4_m13, "logz_err": float(m13.iloc[m13[m13.k == 4].index[0]].logz_err)},
        {"source": "m19 k=3 (nlive=2000)", "logz": log_z_k3, "logz_err": log_z_err_k3},
        {"source": "m17 k=4 (nlive=2000)", "logz": log_z_k4_n2000, "logz_err": log_z_k4_err_n2000},
    ])
    out["delta_vs_first"] = out.logz - out.logz.iloc[0]
    out.to_csv(PROCESSED / "19_k3_k4_validated.csv", index=False)

    # Quick K posterior comparison for the three planets in k=3 model
    print("\n=== m19 k=3 planet posteriors ===")
    for i, (name, P_init, K_init, _) in enumerate(FENG_2017):
        log_P = posterior[:, 5*i]; log_K = posterior[:, 5*i + 1]
        sec = posterior[:, 5*i + 2]; ses = posterior[:, 5*i + 3]
        P_q = np.percentile(np.exp(log_P), [16, 50, 84])
        K_q = np.percentile(np.exp(log_K), [16, 50, 84])
        e_q = np.percentile(sec**2 + ses**2, [16, 50, 84])
        print(f"  {name}: P = {P_q[1]:.1f} ({P_q[0]:.1f}–{P_q[2]:.1f}), "
              f"K = {K_q[1]:.3f} ({K_q[0]:.3f}–{K_q[2]:.3f}), e = {e_q[1]:.3f}")

    # Plot evidence comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["m13 k=3\n(nlive=500)", "m13 k=4\n(nlive=500)", "m19 k=3\n(nlive=2000)", "m17 k=4\n(nlive=2000)"]
    vals = out.logz.tolist()
    errs = out.logz_err.tolist()
    colors = ["lightblue", "navajowhite", "steelblue", "darkorange"]
    ax.errorbar(range(4), vals, yerr=errs, fmt="o", markersize=12, capsize=5,
                color="black")
    for i, (l, v, c) in enumerate(zip(labels, vals, colors)):
        ax.scatter([i], [v], s=200, c=c, zorder=3)
        ax.text(i, v + 0.5, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("log Z")
    ax.set_title("k=3 vs k=4 evidence: nlive=500 (m13) vs nlive=2000 (m17, m19)")
    fig.tight_layout()
    fig.savefig(FIGS / "19_evidence_validated.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
