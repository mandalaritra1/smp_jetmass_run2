# CLAUDE.md

Agent instructions for this repository live in **[AGENTS.md](AGENTS.md)** — read it
first. It covers the architecture, the analysis-context knowledge base, and the hard
invariants (selections are byte-for-byte identical to GluonJetMass; HEM-as-weight;
canonical systematic names; rho = `2*log10(m/(pt*0.8))`; XS in `postprocess`;
binning only in `util_binning`). `README.md` has install + run instructions.

Quick reminders:
- Run from the repo root; the package is the flat `smp_jetmass_run2/` dir (no `src/`).
- Local runs use `~/Projects/GluonJetMass/.venv` (coffea 2026.5.0); test files are in
  `~/Projects/GluonJetMass/test_files/`.
- After touching a processor, re-run the cutflow regression vs GluonJetMass — every
  selection step must still match.
