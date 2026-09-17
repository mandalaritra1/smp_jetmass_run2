# Plot map — where every hadronic figure set lives

Maintained by hand. When a session produces a new figure set, add a row (and
keep locations as directories plus naming patterns, not 138 individual rows;
each directory has its own manifest or index for the drill-down).

Paths: `smp` = `~/Projects/smp_jetmass_run2`, `docs` = `~/Projects/smp25010-docs`.

## Pair split validation campaign (2026-08-08, detector level)

| Description | Location |
|---|---|
| Per-era validation sets, genWeight basis (95 figs each: kinematics, rho panels, yields, sculpting; era-labelled) | `docs/internal/2026-08-08-pair-split-validation/campaign_<era>/` for era in 2016APV, 2016, 2017, 2018 (2018 has 102) |
| Frozen LHE-basis 2018 set used by the meeting deck (do not touch) | `docs/internal/2026-08-08-pair-split-validation/campaign/` |
| Groomed-rho normalization window study (window cuts -3.5, -3.0; both channels, all pT bins) | `docs/internal/2026-08-08-pair-split-validation/campaign_2018/rho_window_study/` |
| The 51-page validation deck + its staged figures | same dir: `pair-split-validation.pdf`, `figs/` |
| Batch plotter that regenerates all of the above | `smp/scripts/validate_pairsplit.py` (`--era` wires paths and labels) |

## Run 2 unfolding on pair split (2026-08-11)

| Description | Location |
|---|---|
| Main figure set, 37 figs: unfolded result, uncertainty budget, reco data/MC (per pT bin), purity/stability, amplification, channel comparison, Bayes vs TUnfold, response matrices, before/after | `smp/outputs/figs/hadronic_rho_pairsplit_run2/` (see `index.md` there for one-line captions) |
| Presentation variants, 11 figs: core-window normalization `_win18` (all 8 slices), trijet coarser 2to1 binning `_2to1` (3 slices) | `smp/outputs/figs/hadronic_rho_pairsplit_run2/variants/` |
| Appendix set, 90 figs: self-closure, per-variation closure, signed syst shifts, toy pulls + coverage per bin, config comparison per bin, Bayes iteration scan, profiled refold + pulls, per-era reco ratios + chi2, per-era purity/stability, full binning ladder | `smp/outputs/figs/hadronic_rho_pairsplit_run2/appendix/` (see `appendix_manifest.md` there, grouped by test with provenance) |
| Curated review deck, 29 slides | `docs/internal/2026-08-11-hadronic-rho-unfolding/hadronic-rho-unfolding.pdf` (staged copies in `figs/`) |
| Exhaustive appendix deck, 44 slides, all 138 figures | same dir: `hadronic-rho-unfolding-appendix.pdf` (staged copies in `figs_appendix/`) |
| Plotting drivers | `smp/scripts/plots/plot_hadronic_rho_pairsplit_run2.py` (main + `--variant`), `appendix_rerun_pairsplit_run2.py` + `plot_hadronic_rho_pairsplit_appendix.py` (appendix) |
| Uncertainty budgets with the JES curve added (era-correlated propagation) | `smp/outputs/figs/hadronic_rho_pairsplit_run2/jes_update/` |
| Numbers behind the plots (per-bin tables, check register, config comparison, per-era chi2, JES per-source shifts) | `smp/outputs/pairsplit_unfold/phaseA_summary.md`, `phaseB_summary.md`, `*.json` |

## Herwig7 groomed-mass pT drift: merging vs shower (2026-09-17)

| Description | Location |
|---|---|
| Dijet gen-level flat-vs-MLM figure set, groomed and ungroomed: median mass ratio Herwig7/Pythia8 for the flat (unmerged) and the MLM pair plus flat/MLM per shower (`dijet_<mode>_mass_ratio_flat_vs_mlm`), median slope per sample relative to 200-290 GeV (`dijet_<mode>_slope_flat_vs_mlm`), per-pT-bin shapes with ratio to MLM+Pythia8 (`dijet_<mode>_shape_pt<i>`, nine panels) | `smp/outputs/plots/dijet_flat_vs_mlm/` (PNG + PDF), copies in `~/Projects/research-notes/attachments/herwig_groomed_mass_2026-09-17/` |
| Producer | `smp/scripts/diagnostics/plot_dijet_groomed_flat_vs_mlm.py --mode groomed|ungroomed [--mlm-eras all]`; prints the per-bin median/n_eff/ratio table |
| Inputs: the flat-pT standalone-shower pickles (2018, no MLM), with manifest | `~/cernbox/unfold_inputs/dijet/rho/original/2018/dijet_flat_{herwig,pythia}/`, `.../dijet/rho/original/flat_manifest.json` |
| Z+jet counterpart (Rivet unmerged runs) | `smp/scripts/diagnostics/plot_zjet_groomed_slope_generators.py`, `plot_zjet_gen_rho_merged_vs_unmerged.py` -> `smp/outputs/plots/gen_rho_generators/` |

## Superseded but kept

| Description | Location |
|---|---|
| July full-stat unfolding figures, OLD trijet-priority selection, 2018 only (33 figs) | `smp/outputs/figs/hadronic_rho_fullstat/` + self-contained `smp/outputs/hadronic_rho_unfolding_fullstat.html` |
| Old-axis unfold inputs and Herwig files (old selection) | `~/Projects/unfold/inputs/{dijet,trijet}/rho/` (July files; superseded by the pair_split pickles in `~/cernbox (2)/hadronic_minimal_rho_pairsplit/`) |
