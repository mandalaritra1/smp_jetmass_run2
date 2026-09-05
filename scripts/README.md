# scripts/

Standalone entry points. Run every one of them **from the repository root** —
they bootstrap `sys.path` to the repo root from their own depth, and import each
other package-style (`from scripts.plot_stamp import stamp`), so `cd scripts &&
python foo.py` will not work.

```
scripts/                 runners and shared modules — the things you still use
scripts/studies/split/   how the dijet/trijet split was chosen (thread closed)
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
| `channel_overlap.py` | dijet/trijet event overlap per leading-jet pT bin; exports `DIJET_CUTS` / `TRIJET_CUTS` / `bit` / `dedupe` | shared module, 8 consumers — cut tuple parity tested |
| `extract_pythia_var_rho.py` | gen Z+jet rho from the pythia_var NanoGEN (nom + cr1/cr2/frag\*), HT-stitched → per-variation npz | produces tracked npz |
| `build_pythia_var_weighters.py` | 24-bin CR/hadronization weighters → `corrections/{cr1,cr2,fraghard,fragsoft}_rho_reweight_*.npz`; takes the gen-rho cache as a positional argument | imports; physics output not regenerated in this cleanup |
| `build_vincia_reweight_npz.py` | Vincia gen reweight (derived in the unfold repo) → PtVarWeighter format | produces tracked npz |

Tracked validation/study tools — left at top level because they import
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

## Unfolding lives in `~/Projects/unfold`, not here

`scripts/studies/unfold/` was removed on 2026-08-13. Unfolding is the unfold
repo's job, and the scripts here had drifted into being a second, partly
uncommitted copy of it.

The 17 files, the smp test that covered them, and the two 2026-08-11 status
docs were salvaged to `~/Projects/unfold/_from_smp_20260813/` as reference
material — see the README there. Six of them were never tracked, and three
carried uncommitted `--binning` work, so that directory is the only copy of
those. The 11 tracked files are additionally recoverable from this repo's
history up to `b4e8e37`.

Do not re-add unfolding scripts to this repo.

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

## Repaired scripts

The `channel_overlap.py` pair_split cut tuples were repaired on 2026-08-13.
`tests/test_channel_overlap_cutbits.py` pins their append-only order to the two
processor `_SKIM_CUTS` tuples.

`build_pythia_var_weighters.py` was made import-safe on 2026-08-13. Its cache
is now a required positional argument; the fine-bin input and correction output
retain their prior defaults.
