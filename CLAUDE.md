# CLAUDE.md

Agent instructions for this repository live in **[AGENTS.md](AGENTS.md)** — read it
first. It covers the architecture, the analysis-context knowledge base, and the hard
invariants (selections are byte-for-byte identical to GluonJetMass; HEM-as-weight;
canonical systematic names; rho = `2*log10(m/(pt*0.8))`; XS in `postprocess`;
binning only in `util_binning`). `README.md` has install + run instructions.

Quick reminders:
- **Talks, ARC trackers/plans, response decks, and the documentation ledger
  live in `~/Projects/smp25010-docs/`** (moved out of this repo 2026-07-21).
  Local diagnostic scripts write their figures there
  (`smp25010-docs/review/figs`).
- **Documentation ledger**: any change to analysis behavior, figures, or
  quotable numbers gets an entry in
  `~/Projects/smp25010-docs/review/AN_PAPER_TODO.md` — ask Aritra at the
  moment of the change whether it goes to the AN, the PAS, or both. Check the
  ledger's Pending list at the start of paper/AN-editing sessions.
- Run from the repo root; the package is the flat `smp_jetmass_run2/` dir (no `src/`).
- Local runs use `~/Projects/GluonJetMass/.venv` (coffea 2026.5.0); test files are in
  `~/Projects/GluonJetMass/test_files/`.
- After touching a processor, re-run the cutflow regression vs GluonJetMass — every
  selection step must still match.
- This is the analysis behind **SMP-25-010** (paper, `~/Projects/SMP-25-010/`) and
  **AN-24-162** (analysis note, `~/Projects/AN-24-162/`) — see AGENTS.md for the
  local paths. Read-only unless asked to edit.
