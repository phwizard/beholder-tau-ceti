# beholder-tau-ceti

Reproducibility repository for the RNAAS submission *"τ Ceti's radial-velocity planets re-examined: cross-validated K-deficit, bimodal candidate f, and a Bayes-factor reversal at higher nested-sampling resolution"* (Filatova, Filatov, in prep.).

## Contents

| Path | What |
|---|---|
| `paper.tex` | RNAAS LaTeX source (≤ 1,000 words, 1 figure, ≤ 6 references). |
| `figures/k_comparison.png` | Headline figure: Feng+ vs RadVel vs ours (no-GP, with-GP). |
| `figures/evidence_ladder.png` | Supplementary: m13 vs m19 BF reversal. |
| `figures/posterior_widths.png` | Supplementary: m17 nlive=2000 posterior widths. |
| `data/23_radvel_validation.csv` | Per-planet K from RadVel (no-GP), ours (no-GP), and ours m17 (with-GP). |
| `data/19_k3_k4_validated.csv` | log Z for k=3 and k=4 at nlive=500 (m13) and nlive=2000 (m17/m19). |
| `data/17_posterior_width_comparison.csv` | Per-parameter posterior widths at nlive=500 vs 2000. |
| `scripts/run_23_radvel_validation.py` | RadVel cross-validation script. |
| `scripts/run_17_dynesty_validation.py` | Posterior-width validation at nlive=2000. |
| `scripts/run_19_k3_validated.py` | k=3 dynesty run at nlive=2000. |

## How to reproduce

Requires Python ≥ 3.11 and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/phwizard/beholder-tau-ceti.git
cd beholder-tau-ceti
uv sync --extra modeling
uv run python scripts/run_23_radvel_validation.py    # ~30 sec
uv run python scripts/run_17_dynesty_validation.py   # ~2 hours
uv run python scripts/run_19_k3_validated.py         # ~1.5 hours
```

The DACE multi-instrument RV input data (`rv_dace_tau_ceti.csv`, 5 MB) is included in `data/`. The heavy posterior chains (NPZ files, ~50 MB each) are not included; the scripts regenerate them locally.

## Authors

- **Vira Filatova** — vira AT deepxhub.com, DeepX https://deepxhub.com/
- **Taras Filatov** — taras AT deepxhub.com, DeepX https://deepxhub.com/

## AI tool use disclosure

Analysis benefited substantially from work performed by the Anthropic Claude language model (Opus 4.7) acting as a research assistant: code authoring, milestone-by-milestone analysis and detection of methodological errors. The full development history (23 milestones, 5,500 lines of analysis code) lives in the parent project at https://github.com/phwizard/beholder (private; available on request).

## License

MIT. See `LICENSE`.

## Citation

If this work is useful to you, please cite the RNAAS note once published, and the DACE platform (Buchschacher et al. 2015) and supporting tools (RadVel, dynesty, celerite2) per their requirements.
