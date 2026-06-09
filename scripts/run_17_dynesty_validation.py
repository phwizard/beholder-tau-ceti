"""Milestone 17 — multi-modal high-nlive validation of m12 posteriors.

m10/m12/m13/m14 all reported posteriors with absurdly narrow uncertainties:
  σ(P_g) ≈ 0.01–0.03 d, σ(K_g) ≈ 0.001–0.01 m/s

Two interpretations:
  (a) The data + priors really do constrain these parameters that tightly
      (1,275 nightly bins × multi-instrument zero-points, after all).
  (b) dynesty's dynamic-NS refinement at nlive=500 over-concentrates on
      the single deepest mode and under-samples the broader posterior.

This milestone tests (b) by rerunning the m12 setup with:
  - nlive=2000 (4× more live points → more thorough mode finding)
  - bound="multi" with broader ellipsoid bounding
  - longer chain and explicit posterior-focused weighting

If posteriors broaden by 5-10× under nlive=2000, m12-m14 reported error
bars are too narrow and need to be inflated for any external citation.
If they stay narrow, m12-m14 are trustworthy as written.

Wall-clock estimate: ~2 hours (4× m12's 38 min).
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
    ("f", 636.0, 0.42, 53000.0),
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


def _build_model(data, gp_init, gp_mean) -> JointRVModel:
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
        data=data, n_planets=4, gp_params=gp_init, gp_mean=gp_mean,
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


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    data = _load_data()
    gp_init, gp_mean = _load_gp_init()
    model = _build_model(data, gp_init, gp_mean)
    pt = make_prior_transform(model)
    print(f"n_dim = {model.n_dim}, n_data = {len(data.t)}")

    print("\n=== dynesty: nlive=2000, bound=multi, posterior-focused ===")
    print("(4× more live points than m12; explicit multi-modal bounding)")
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
    log_z = results.logz[-1]
    log_z_err = results.logzerr[-1]
    print(f"\nlog Z = {log_z:.3f} ± {log_z_err:.3f}")
    print(f"n_iters = {len(results.logl)}")

    samples = results.samples
    weights = results.importance_weights()
    posterior = dyutils.resample_equal(samples, weights)
    print(f"posterior samples (equal-weight): {posterior.shape}")

    np.savez(
        PROCESSED / "17_dynesty_nlive2000.npz",
        samples=samples, weights=weights, posterior=posterior,
        logz=log_z, logz_err=log_z_err, logl=results.logl, elapsed_s=elapsed,
    )

    # ---- Compare against m12 ----
    m12_post = np.load(PROCESSED / "12_dynesty_loose_rho.npz")["posterior"]
    m12_logz = float(np.load(PROCESSED / "12_dynesty_loose_rho.npz")["logz"])
    print(f"\n=== Comparison vs m12 (nlive=500) ===")
    print(f"log Z   m12 nlive=500:  {m12_logz:.3f}")
    print(f"log Z   m17 nlive=2000: {log_z:.3f}")
    print(f"Δ log Z (m17 - m12):    {log_z - m12_logz:+.3f}")

    # Per-candidate posterior comparison
    print("\n=== Per-candidate posterior widths: m12 vs m17 ===")
    print("                  m12 (nlive=500)              m17 (nlive=2000)            ")
    print("  cand | param | med   q16   q84   width  | med   q16   q84   width  | width ratio")
    print("  -----|-------|------------------------|------------------------|-----------")
    rows = []
    for i, (name, _, _, _) in enumerate(FENG_2017):
        for label, idx in [("log_P", 5*i), ("log_K", 5*i+1)]:
            m12_q = np.percentile(m12_post[:, idx], [16, 50, 84])
            m17_q = np.percentile(posterior[:, idx], [16, 50, 84])
            m12_w = m12_q[2] - m12_q[0]
            m17_w = m17_q[2] - m17_q[0]
            ratio = m17_w / m12_w if m12_w > 0 else np.inf
            print(f"  {name}    | {label:6s}| {m12_q[1]:6.3f} {m12_q[0]:6.3f} {m12_q[2]:6.3f} {m12_w:6.4f} | "
                  f"{m17_q[1]:6.3f} {m17_q[0]:6.3f} {m17_q[2]:6.3f} {m17_w:6.4f} | {ratio:5.2f}×")
            rows.append({
                "candidate": name, "param": label,
                "m12_med": m12_q[1], "m12_q16": m12_q[0], "m12_q84": m12_q[2], "m12_width": m12_w,
                "m17_med": m17_q[1], "m17_q16": m17_q[0], "m17_q84": m17_q[2], "m17_width": m17_w,
                "width_ratio": ratio,
            })

    pd.DataFrame(rows).to_csv(PROCESSED / "17_posterior_width_comparison.csv", index=False)

    # GP comparison
    gp_idx0 = 5 * 4 + 2 * model.n_inst
    print("\n=== GP hyperparameters: m12 vs m17 ===")
    for label, off in [("P_rot (d)", 1), ("rho_long (d)", 5)]:
        m12_v = np.exp(m12_post[:, gp_idx0 + off])
        m17_v = np.exp(posterior[:, gp_idx0 + off])
        print(f"  {label}:")
        print(f"    m12: {np.median(m12_v):.1f} ({np.percentile(m12_v, 16):.1f} – {np.percentile(m12_v, 84):.1f})")
        print(f"    m17: {np.median(m17_v):.1f} ({np.percentile(m17_v, 16):.1f} – {np.percentile(m17_v, 84):.1f})")

    # Plots
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, (name, P_init, K_init, _) in enumerate(FENG_2017):
        # Period
        ax = axes[0, i]
        P12 = np.exp(m12_post[:, 5*i])
        P17 = np.exp(posterior[:, 5*i])
        lo = min(np.percentile(P12, 1), np.percentile(P17, 1))
        hi = max(np.percentile(P12, 99), np.percentile(P17, 99))
        bins = np.linspace(lo, hi, 60)
        ax.hist(P12, bins=bins, alpha=0.5, color="orange", label="m12 nlive=500", density=True)
        ax.hist(P17, bins=bins, alpha=0.5, color="steelblue", label="m17 nlive=2000", density=True)
        ax.axvline(P_init, color="crimson", linestyle="--", label=f"Feng+ {P_init}")
        ax.set_title(f"τ Cet {name} — P [d]")
        ax.legend(fontsize=7)
        # K
        ax = axes[1, i]
        K12 = np.exp(m12_post[:, 5*i + 1])
        K17 = np.exp(posterior[:, 5*i + 1])
        lo = 0
        hi = max(np.percentile(K12, 99), np.percentile(K17, 99))
        bins = np.linspace(lo, hi, 60)
        ax.hist(K12, bins=bins, alpha=0.5, color="orange", label="m12 nlive=500", density=True)
        ax.hist(K17, bins=bins, alpha=0.5, color="steelblue", label="m17 nlive=2000", density=True)
        ax.axvline(K_init, color="crimson", linestyle="--", label=f"Feng+ {K_init}")
        ax.set_xlabel("K [m/s]")
        ax.legend(fontsize=7)
    fig.suptitle("Tau Ceti — does increasing nlive 500→2000 broaden the posteriors?", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGS / "17_posterior_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("\nArtifacts:")
    for p in [
        PROCESSED / "17_dynesty_nlive2000.npz",
        PROCESSED / "17_posterior_width_comparison.csv",
        FIGS / "17_posterior_comparison.png",
    ]:
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
