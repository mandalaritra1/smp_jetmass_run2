# scripts/

Standalone entry points. Run every one of them **from the repository root** —
they bootstrap `sys.path` to the repo root from their own depth, and import each
other package-style (`from scripts.plot_stamp import stamp`), so `cd scripts &&
python foo.py` will not work.

```
scripts/                 runners and shared modules — the things you still use
scripts/studies/split/   how the dijet/trijet split was chosen (thread closed)
scripts/studies/unfold/  rho binning / unfolding studies (coupled to ~/Projects/unfold)
scripts/plots/           one-off figure producers
```

**Status column.** `imports` means the module was import-checked after the
2026-08-10 reorganization and loads cleanly; it does **not** mean it was run
against data. Anything stronger is stated explicitly.

---

## Top level — runners and shared modules

| file | what it does | status |
|---|---|---|
| `run_analysis_cli.py` | headless runner for all three channels (zjet / dijet / trijet) | live |
| `run_dijet_skim_split.py` | hadronic `rho_skim` over the QCD HT bins, one submit per bin on a single long-lived Dask client | live — named by all 28 pair_split configs |
| `run_zjet_skim.py` | ZJetOmniFoldSkimmer across local / coffea-casa / LPC executors | live |
| `run_sherpa_gen.py` | GEN-only Sherpa jet-mass ntuple over all Sherpa files on coffea-casa | live |
| `run_overnight_skims.sh` | unattended skim batch | live |
| `merge_skims.py` | merge per-chunk skim Parquet into one Parquet per dataset | live |
| `plot_stamp.py` | provenance stamp for working plots — **mandatory**, see the `cms-plot-style` skill | shared module, 3 consumers |
| `channel_overlap.py` | dijet/trijet event overlap per leading-jet pT bin; exports `DIJET_CUTS` / `TRIJET_CUTS` / `bit` / `dedupe` | shared module, 8 consumers — **cut tuples are stale, see below** |
| `extract_pythia_var_rho.py` | gen Z+jet rho from the pythia_var NanoGEN (nom + cr1/cr2/frag\*), HT-stitched → per-variation npz | produces tracked npz |
| `build_pythia_var_weighters.py` | 24-bin CR/hadronization weighters → `corrections/{cr1,cr2,fraghard,fragsoft}_rho_reweight_*.npz` | **broken at import** — `np.load()` runs at module level against a deleted session scratchpad path |
| `build_vincia_reweight_npz.py` | Vincia gen reweight (derived in the unfold repo) → PtVarWeighter format | produces tracked npz |

Untracked, pending review — left at top level because they import
`scripts.channel_overlap` and `scripts.plot_stamp` and need no path edits:

| file | what it does |
|---|---|
| `validate_pairsplit.py` | batch pair_split validation plotter, main+ratio standalone figures, PNG + vector PDF twin (PDFs are the AN copies). 82 KB — outgrown one file |
| `compare_dijet_generators_rho.py` | dijet rho across generators |
| `diagnose_mg10_cut.py` | the mg > 10 cut diagnostic |

---

## `studies/split/` — how the dijet/trijet split was chosen

Thread **closed** on 2026-08-06 by `5f098fb`, which adopted `pair_split`:
one azimuthal boundary on the leading AK8 pair, dijet Δφ₁₂ > π − (200 GeV)/pT_avg
and trijet the exact complement. `split_scan_angular_pair.py` is the rule that
won; the rest are the rejected alternatives, kept because they are the written
justification for the choice.

| file | the question it answered | status |
|---|---|---|
| `split_scan_angular_pair.py` | pair-level, theory-friendly splits with no third-jet variable | **the adopted rule** — cited in `smp25010-docs` |
| `split_scan_angular.py` | angular candidate cuts that split the channels orthogonally | imports |
| `split_scan_momentum.py` | momentum-balance split rules | imports |
| `channel_asym_split.py` | can leading-pair pT asymmetry alone do it? | imports |
| `channel_orthogonality.py` | how to make the channels orthogonal, quantified from skims | imports — cited in `dijet_processor.py` and `hadronic_base.py` comments |
| `qg_tag_feasibility.py` | gen-truth: how much gluon enrichment do kinematic tags buy? | imports |
| `zjet_dphi_split_study.py` | translating the hadronic pT-dependent decorrelation cut to Z+jet | imports |

---

## `studies/unfold/` — rho binning and unfolding

All committed 2026-07-31, all coupled to `~/Projects/unfold`. **Import-checked
only.** Known from earlier sessions: the stored `combine_full` npz predate the
current binning, so the comparison and diagnostics plotters here are expected to
fail against those inputs until the fits are re-run — re-verify before trusting
any number one of these prints.

| file | what it computes | status |
|---|---|---|
| `rho_width_scan.py` | minimum resolvable groomed-rho bin width per pT bin | imports — 3 consumers here |
| `binning_feasibility.py` | purity / stability / fake / miss at any candidate binning | imports |
| `rho_matching_quality.py` | gen↔reco matching quality per (pT, rho) bin | imports |
| `rho_data_mc.py` | data/MC at RECO level in a candidate groomed-rho binning | imports |
| `rho_unfold_inputs.py` | TUnfold inputs for the 12-bin proposal, from the skims | imports |
| `rho_unfold_inputs_from_hists.py` | same, but from the **production histograms** instead of the ntuples | imports |
| `rho_unfold_stability.py` | is the 12-bin proposal stable under a τ=0 TUnfold? | imports |
| `rho_unfold_systematics.py` | propagate JER/JMS/JMR through the adopted unfolds | imports |
| `rho_unfold_crosschecks.py` | comprehensive numerical cross-checks for the locked unfolds | imports |
| `rho_profiled_refold.py` | profile response and feed-in nuisances in the refolding fit | imports — covered by `tests/test_rho_profiled_refold.py` (passing) |
| `rho_alternate_response.py` | nominal vs alternate-generator response on the same rho data | imports |

---

## `plots/` — one-off figure producers

| file | figure | status |
|---|---|---|
| `plot_qg_veto.py` | hadronic q/g content and the trijet-veto effect | imports |
| `qg_content.py` | q/g content of the measured jets, from the minimal_rho productions | imports |
| `plot_njet_hadronic.py` | AK8 jet multiplicity in the hadronic channels | imports |
| `plot_hadronic_alternate_response.py` | Pythia-vs-Herwig hadronic rho response stress test | imports |
| `plot_hadronic_data_mc_before_after.py` | bottom-line data/MC disagreement before and after unfolding | imports |
| `make_met_hem_plots.py` | every MET / HEM / AK4 validation plot, from the validation-mode pickles | imports |
| `plot_zjet_gen_flavor_mass_mpt.py` | GEN jet mass and mpt split by initiating flavor | imports |
| `plot_zjet_gen_flavor_pt_y.py` | Z+jet GEN leading-jet flavor splits in pT and rapidity | imports |
| `print_zjet_quark_fractions.py` | Z+jet GEN leading-jet quark fractions per analysis pT bin | imports — cited in `smp25010-docs` |
| `export_skim_events.py` + `event_viewer_template.html` | dijet event-display tables → standalone 3D event viewer | imports |

---

## Known breakage

**`channel_overlap.py` cut tuples are stale.** `tests/test_channel_overlap_cutbits.py`
has been failing on `dev` since the pair_split selection landed — the processors
gained cuts that were never added to the tuples:

- dijet: `recoPtFloor`, `recodphiPair`, `genPtFloor`, `dphiGenPair`
- trijet: `recoPtFloor3`, `rapLead3`, `recodphiPairInv`, `genPtFloor3`, `rapGenLead3`, `dphiGenPairInv`

Anything reading `DIJET_CUTS` / `TRIJET_CUTS` — including the bit masks in
`validate_pairsplit.py` — is working off the pre-`pair_split` cut list. Fix the
tuples before trusting an overlap or cutbit number.

**`build_pythia_var_weighters.py` fails at import** — it calls `np.load(CACHE)`
at module level and `CACHE` points at a session scratchpad that no longer exists.
Needs the path turned into an argument.
