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
| Producer (since 2026-09-18 the one consolidated script) | `smp/scripts/diagnostics/plot_rho_generators.py --channel dijet --level gen --mode groomed|ungroomed --samples mlm_pythia,mlm_herwig,flat_pythia,flat_herwig --pairs "flat_herwig/flat_pythia,mlm_herwig/mlm_pythia:flat_herwig/mlm_herwig,flat_pythia/mlm_pythia" --tag dijet_<mode>` (`--mlm-eras all` for the Run-2 MLM sums); prints the per-bin median/n_eff/ratio table. Output names are now `<tag>_mass_ratio`, `<tag>_slope`, `<tag>_shape_pt<i>` |
| Inputs: the flat-pT standalone-shower pickles (2018, no MLM), with manifest | `~/cernbox/unfold_inputs/dijet/rho/original/2018/dijet_flat_{herwig,pythia}/`, `.../dijet/rho/original/flat_manifest.json` |
| Z+jet counterpart (Rivet unmerged runs; the old `gen_groomed_slope_generators*.png` and `gen_rho_merged_vs_unmerged*.png` in `smp/outputs/plots/gen_rho_generators/`) | same script: `--channel zjet --level gen --mlm-eras all --grid native --samples mlm_pythia,mlm_herwig,lo_herwig_lhe,lo_pythia_lhe,sa_herwig,cp5_pythia,pythia_internal,vincia,sherpa3[,data_unfolded] --figures slope` (slope figure) and `--grid common --samples mlm_pythia,mlm_herwig,sa_herwig[,lo_pythia_lhe] --figures shapes` (one panel per pT bin instead of the old 2x3 grid) |
| Z+jet gen-level Matchbox-vs-MLM figure set (2026-09-18), groomed and ungroomed: median mass ratio against MLM+Pythia8 for the NLO Matchbox Herwig7 (no MLM) and the MLM Herwig7 sample plus Matchbox/MLM Herwig7 (`zjet_<mode>_mass_ratio_matchbox_vs_mlm`), median slope per sample (`zjet_<mode>_slope_matchbox_vs_mlm`), per-pT-bin shapes with ratio (`zjet_<mode>_shape_pt<i>`, four panels). Result: Matchbox drifts like MLM+Herwig7 (1.09 -> 0.63 vs 1.41 -> 0.77) | `smp/outputs/plots/zjet_matchbox_vs_mlm/` (PNG + PDF), copies in `~/Projects/research-notes/attachments/herwig_groomed_mass_2026-09-17/` |
| Producer (since 2026-09-18 the one consolidated script) | `smp/scripts/diagnostics/plot_rho_generators.py --channel zjet --level gen|reco --mode groomed|ungroomed --samples mlm_pythia,mlm_herwig,matchbox[,data] --pairs "matchbox/mlm_pythia,mlm_herwig/mlm_pythia[,data/mlm_pythia]:matchbox/mlm_herwig" --tag zjet_<mode>[_reco] [--rho-floor -3]`; the 2x-refined Matchbox rho axis is rebinned onto the production edges (`--grid reference`); prints the per-bin table. The per-channel gen/reco data-vs-generator sets in `gen_rho_generators/` and `reco_rho_data/` come from the same script with `--samples mlm_pythia,mlm_herwig[,data] --pairs mlm_herwig/mlm_pythia[,data/mlm_pythia] --mlm-eras all` (`--mass-axis` for the GeV panels) |
| Inputs: the Matchbox pickle (2018, `herwig_matchbox` dataset, LPC condor) with manifest | `~/cernbox/unfold_inputs/zjet/rho/original/2018/herwig_matchbox/minimal_rho_zjet_herwig_matchbox_2018.pkl`, `.../zjet/rho/original/herwig_matchbox_manifest.json` |
| Fine-rho version (2026-09-18): the same three samples produced together in `minimal_rho_fine` (4x rho bins), gen groomed on the fine lattice (full window and above -3) and reco with 2018 data (medians on the data lattice, shapes on the fine lattice) | `smp/outputs/plots/zjet_matchbox_vs_mlm_fine/`, `..._fine_floor-3/`; inputs `~/cernbox/unfold_inputs/zjet/rho/fine/2018/minimal_rho_fine_{pythia,herwig,herwig_matchbox}_2018.pkl` + `.../zjet/rho/fine/fine_manifest.json`; `plot_rho_generators.py --samples pythia_fine,herwig_fine,matchbox_fine[,data] --reference pythia_fine|data ...` |

## Superseded but kept

| Description | Location |
|---|---|
| July full-stat unfolding figures, OLD trijet-priority selection, 2018 only (33 figs) | `smp/outputs/figs/hadronic_rho_fullstat/` + self-contained `smp/outputs/hadronic_rho_unfolding_fullstat.html` |
| Old-axis unfold inputs and Herwig files (old selection) | `~/Projects/unfold/inputs/{dijet,trijet}/rho/` (July files; superseded by the pair_split pickles in `~/cernbox (2)/hadronic_minimal_rho_pairsplit/`) |
