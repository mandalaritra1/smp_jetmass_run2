# Handoff: high-stats rerun of the mglo+pythia modelling-uncertainty suite

**Goal:** produce *physical* bin-by-bin fractional modelling-uncertainty numbers
(per source: shower / colour reconnection / hadronization) for the Z+jet jet-mass
ARC response. The validation pass (4000 events) gave **20–230%/bin — that is MC
noise, not physics**. We need high statistics, then regenerate the per-source plots
+ CSV. This is intentionally split into its own session.

## Context (what already exists, on `main`)
- `rivet/run_mg_local.sh` — local two-image driver: MG5 LO Z+jet → one LHE
  (scailfin image) → showers the **same** LHE 6 ways (hepstore image) so each shape
  difference isolates one effect:
  `pythia` (nominal), `pythia_vincia` (shower), `pythia_cr1`/`pythia_cr2`
  (CR: QCD-inspired / gluon-move), `pythia_fragsoft`/`pythia_fraghard`
  (Lund hadronization soft/hard). → `out/mglo_<name>.yoda` + `out/plots_syst/`.
- `rivet/plot_model_uncertainty.py` — parses those yodas, area-normalises each
  histogram per pT bin, and computes per source
  `env[bin] = max_member |norm_var − norm_nom| / norm_nom`, total = quadrature.
  → `out/model_unc/model_unc_{rho_g,rho_u,mass_g,mass_u}.{png,pdf}` +
  `out/model_unc/model_uncertainty.csv`.
- Docker images already pulled: `scailfin/madgraph5-amc-nlo:mg5_amc3.5.1`,
  `hepstore/rivet-pythia:latest`.

## Run it (from repo root)
```bash
cd rivet
# N drives BOTH the MG5 LHE size and the shower count. --regen forces a fresh,
# larger LHE (the cached 30k LHE caps shower stats at 30k). Pick N high:
./run_mg_local.sh 300000 --regen        # ~? min under amd64 emulation; serial MG5 build
# then the per-source uncertainty plots + numbers (host venv has numpy+matplotlib):
/Users/aritra/Projects/GluonJetMass/.venv/bin/python plot_model_uncertainty.py
open out/model_unc/model_unc_rho_g.png          # + rho_u, mass_g, mass_u
open out/model_unc/model_uncertainty.csv
```

## What to check / decide
- **Convergence:** the bulk bins should drop to a stable ~few–20%; if the tails are
  still noisy at 300k, go higher or restrict the quoted range to populated bins
  (the analysis already drops the empty low-ρ / high-ρ edge bins).
- **High-pT tail:** the inclusive MG LHE thins out at high pT. If `pt400_Inf` stays
  noisy, generate the LHE in **pThat slices** (see `gen/madgraph/mg5_zjet_sliced.dat`
  / `run_mg_sliced.sh`, the LPC-native sliced setup) and stitch, rather than just
  raising N.
- **Sources:** confirm all three appear in `plot_model_uncertainty.py`'s
  `active sources:` print (Shower needs `out/mglo_pythia_vincia.yoda`).
- **Caveat to carry:** these are mode-flips / exploratory brackets on the **default
  (Monash) tune**, not the retuned CP5-CR1/CR2 + eigentune values — fine for sizing
  the effect; swap in the CP5 numbers (arXiv:1903.12179) for the final systematic.

## When done
- Numbers feed the ARC modelling-uncertainty response (the "shower vs hadronisation"
  question; cf. AN Figs 52–58). Compare the size/shape against the current
  Pythia→Herwig reweighting band.
- Log the outcome in `research-notes/projects/zjet.md` (Completed Work Log + the
  `ARC SMP-25-010 round-1` decisive note), per the project convention.
