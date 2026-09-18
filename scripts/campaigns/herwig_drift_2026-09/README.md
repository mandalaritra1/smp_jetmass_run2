# Herwig7 groomed-mass pT drift: campaign helpers (2026-09-17/18)

Session helpers rescued from the agent scratchpad so the runs are reproducible.
Physics record: research-notes `topics/herwig_groomed_jet_mass_pt_drift.md`
(sections 4b and 6); workflow record: ai-wiki
`topics/hadronic_flat_qcd_merging_test.md`, `topics/zjet_herwig_matchbox_dy_sample.md`.

| file | what | where it runs |
|---|---|---|
| `launch_flat_lpc.sh`, `launch_flat_lxplus.sh` | dijet `herwig_flat` / `pythia_flat` minimal_rho 2018 (smoke, single, chained); `python -u`, setsid/nohup | copy to the LPC deployment dir (`~/nobackup/smp_jetmass_run2_hadronic_herwig`, needs `lpc_container.sh`) or `lxplus:~/smp_jetmass_run2` (LCG_110) |
| `launch_matchbox_lpc.sh`, `launch_matchbox_lxplus.sh` | zjet `herwig_matchbox` minimal_rho 2018 (smoke, run) | same |
| `scan_flat_weights.py` | per-file scan of `genWeight`, `Generator_binvar`, leading `GenJetAK8_pt` over a flat-pT sample: effective statistics per gen pT bin vs a jet/pT-hat cut, the heaviest events per bin; the study behind `HadronicProcessorBase.FLAT_PTHAT_RATIO_MAX` | lxplus (multiprocessing over files through cms-xrd-global), ~3 min for 700 files |
| `stage_flat_pickles.py` | copy the two flat dijet pickles into `~/cernbox/unfold_inputs/dijet/rho/original/2018/dijet_flat_*/` and write `flat_manifest.json` (sha256, events, sumw, veto counts, n_eff per bin, axis check vs the MLM production) | Mac |
| `stage_matchbox_pickle.py` | same for the zjet Matchbox pickle -> `zjet/rho/original/2018/herwig_matchbox/` + `herwig_matchbox_manifest.json` | Mac |

The comparison figures come from `scripts/diagnostics/plot_rho_generators.py`
(one script with a sample registry, since 2026-09-18; `--list` prints the
registry, `--channel`, `--level gen|reco`, `--mode`, `--samples`, `--pairs`,
`--grid`, `--rho-floor`); the exact commands per figure set are in
`PLOT_MAP.md`.

Regression record (2026-09-18): the consolidated script was run against every
one of the six scripts it replaces (zjet gen rho generators, reco rho with
data, merged vs unmerged, slope generators, dijet flat vs MLM, zjet Matchbox
vs MLM) on their own inputs; medians, errors, effective statistics, mass
ratios and slopes agreed to the last printed digit in every table. The one
behavioural unification: negative bin contents are clipped to zero in the
median for every sample (the data scripts already did this; the MC tables are
unchanged by it). Layout change: the merged-vs-unmerged 2x3 grid became one
panel per pT bin.
