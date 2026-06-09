# Development log — 23 milestones

A compact summary of the methodological journey from data fetch to publishable results. Each entry is a one-line description of what was done and what was learned. Detailed per-milestone writeups, conversation logs, and intermediate-state code are kept in the parent project (private; available on request).

| #  | What we did | What we learned |
|---|---|---|
| 01 | TESS photometric rotation search | No coherent rotation signal — Tau Ceti is too quiet for direct photometric P_rot determination |
| 02 | Multi-instrument RV Lomb–Scargle periodograms | Peaks present at all four Feng+ 2017 candidate periods, but mixed with activity-period harmonics |
| 03 | Iterative pre-whitening + injection–recovery | Conservative noise floor at K ≈ 0.5 m/s under simple frequentist analysis |
| 04 | Activity Gaussian-process decorrelation (Ca II HK) | GP timescale ρ_long pinned at 500 d prior bound — first sign that the literature bound was too restrictive |
| 05 | Joint Bayesian k=4 fit, HARPS-only (emcee) | Single-instrument pipeline produces sensible posteriors |
| 06 | Extend to multi-instrument k=4 (fixed GP, emcee) | K uncertainties tighten 30–44%; HARPS pre/post-2015 fibre upgrade offset recovered (+1.27 m/s) |
| 07 | BIC ladder k=0..4 (free-GP, emcee) | Initial BIC suggested k=0 preferred — later retracted as kernel + sampler + BIC-formula confluence |
| 08 | Loose-period sensitivity test (emcee) | K_e and K_f appear to collapse with loose priors — later shown sampler-dependent |
| 09 | Zeus ensemble-slice sampler validation | Emcee and zeus disagree substantially at k=4 — posterior is multi-modal or strongly degenerate |
| 10 | dynesty nested-sampling exploration | Finds a likelihood mode 24 nats above what emcee/zeus reached — but GP ρ_long pinned again |
| 11 | Kernel comparison: QP+Matern vs rotation-only | Data wants the long-term term decisively (BF 10¹⁵) — ρ_long-on-bound is the bound's fault, not the kernel's |
| 12 | Lower ρ_long bound from 500 d to 50 d | ρ_long settles at 96 d (interior); K_f recovers from 0.086 → 0.40 m/s; Bayes factor 2.2×10⁶ in favour |
| 13 | Full evidence ladder k=0..4 (dynesty, nlive=500) | k=3 (g+h+e) preferred at 60% model probability, k=4 second at 20% |
| 14 | "Skip-h" test: is h really needed? | BF 75 against removing h when e is present — h is confirmed real |
| 15 | TESS BLS transit search at recovered periods | No transit detections; useful depth limits (R_p ≤ 0.86 R_⊕ for g) |
| 16 | TESS Gaussian-process detrending alternative | Naïve GP detrending suppresses transit signal — stay with Savitzky-Golay for known periods |
| 17 | Repeat k=4 at nlive=2000 (sampler validation) | **Major correction**: posteriors at nlive=500 were 25–100× too narrow; K_f is bimodal |
| 18 | Project-level science synthesis (`SCIENCE.md`) | Internal documentation milestone |
| 19 | Repeat k=3 at nlive=2000 (BF reversal check) | **m13 verdict flipped**: at convergence, k=4 weakly preferred (BF 2) — data is indifferent between k=3 and k=4 |
| 20 | Pivot to ε Eridani as second target | Pipeline immediately recovers known rotation period and direct-imaged ε Eri b — framework transfers |
| 21 | First ε Eri activity GP attempt | Over-fits and absorbs planet signal (active star, many free amplitudes) |
| 22 | ε Eri linear HK regression as alternative | Under-fits — only 1 free parameter cannot capture both magnetic-cycle drift and rotation oscillation |
| 23 | RadVel cross-validation for τ Ceti | Independent peer-reviewed pipeline agrees with ours to within 1.5%; K-deficit vs Feng+ 2017 is robust to pipeline choice |

## What's in this repository

This public reproducibility kit contains the **headline-result subset**: code and data sufficient to regenerate the three figures and the comparison table in the RNAAS note. Specifically:

- `paper.tex` — RNAAS submission source
- `figures/k_comparison.png` — Figure 1 (cross-validation)
- `figures/evidence_ladder.png`, `figures/posterior_widths.png` — supporting figures referenced in the discussion
- `data/processed/` — cleaned multi-instrument RV input + per-milestone result CSVs
- `scripts/run_17_dynesty_validation.py` — reproduce the nlive=2000 posterior validation
- `scripts/run_19_k3_validated.py` — reproduce the k=3 vs k=4 BF reversal
- `scripts/run_23_radvel_validation.py` — reproduce the RadVel cross-validation
- `src/beholder/` — analysis library (data loaders, activity GP, Keplerian RV, Bayesian model)

The heavy posterior chains (~50 MB each NPZ file) are not included; the scripts regenerate them locally on a 2024 MacBook Pro in ~1.5–2 hours for milestones 17 and 19, ~30 seconds for milestone 23.

Earlier milestones (01–16, 18, 20–22) used intermediate code versions that have been superseded by the corrected pipeline; their outputs are not used for the headline numbers and are not redistributed.
