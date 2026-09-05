# Hadronic standalone production file map

This directory contains the standalone Pythia + Rivet machinery used to build
the dijet/trijet modelling variations.  For launch commands, normalization
rules, EOS locations, and campaign-specific gates, use [LAUNCH.md](LAUNCH.md).
This file answers the narrower question: which file belongs to which production
wave, and is it still part of the active workflow?

## Production waves

| wave | hard process and shower | build | status |
|---|---|---|---|
| Legacy matched gridpack | official UL QCD-HT MadGraph MLM LHE, then Pythia 8.312 CP5/internal variations with MLM | LCG_106 | Historical. Kept for the original same-ME canary and provenance; do not use its Vincia branch because the MLM hook is invalid with Vincia. |
| HardQCD study | standalone Pythia `HardQCD:all`, CP5/Vincia/internal variations | LCG_107 | Historical high-statistics study. Useful for diagnostics and ME-closure checks, but superseded for final internal-variation shapes by the matched-gridpack campaign. |
| Internal variations | official UL QCD-HT MadGraph MLM LHE, then Pythia 8.313 CP5/CR1/CR2/fragmentation variations | LCG_107 | Current, manifest-gated final internal-variation workflow. The 2026-08-02 campaign is complete. |
| MESS | official UL QCD-HT MadGraph LHE, then Pythia 8.313 MESS merging + Vincia | LCG_107 | Current, manifest-gated supported Vincia workflow. The 1M + 5M 2026-08-02 campaign is complete. |
| MESS pair_split dijet | Same MESS + Vincia chain with the dedicated `CMS_HADRONIC_PAIR_SPLIT` fiducial | LCG_107 | Historical August 2026 campaign; preserve its source and selection. |

`build_rivet.sh` and `build_rivet_313.sh` target ABI-incompatible LCG stacks.
Never copy, rename, or reuse a binary from one build as the other.  The LCG_107
build deliberately writes into `py313/` to keep the products separate.

Status meanings below:

- **current**: belongs to one of the supported manifest-gated workflows or its
  final validation/harvest/plot chain;
- **historical**: retained to reproduce or understand an earlier wave, but not
  the route for a new final production;
- **shared**: used by more than one wave;
- **superseded**: replaced by a named file and should not be used for new work.

## Frozen pair-split campaign, 2026-08-13

`mess_pairsplit_dijet_20260813_v1` is a completed historical campaign. The
saved `results/mess_pairsplit_dijet_20260813_v1/final-all/final_all_audit.json`
records 1,624 accepted pilot2/full manifests and 8.120 million LHE events.
The current `../CMS_HADRONIC_PAIR_SPLIT.cc` source hash matches that audit.

Preserve the Rivet source, `build_pairsplit_313.sh`, `inner_mess_pairsplit.sh`,
`submit_mess_pairsplit.sh` and `submit_mess_pairsplit.jdl` together. The build
also uses the tracked `pythia_rivet_mess.cc` driver and `cp5.cmnd` tune fragment.
The scripts remain in their original locations so their relative paths work.
Their default campaign name, deterministic seeds and cuts describe this
historical production, not a new submission.

The source requires both leading jets above 200 GeV. Current pair-split dijet
processing instead uses a 185 GeV floor, so this sample must not be described
as matching the current processor selection. Changing the historical cuts
would break its provenance; a new production requires a separately reviewed
configuration. No production is launched by preserving these files.

Before reusing the worker, repair failure-log preservation: its submit file
discards stdout/stderr, and early failure removes scratch before useful logs
are staged. The audit and wait helpers also have known acceptance/deadline
issues and must not be treated as verified gates for a new campaign.

## File inventory

| file | what it does | wave | status |
|---|---|---|---|
| `LAUNCH.md` | Operational runbook: launch recipes, campaign gates, EOS layout, normalization, and failure lessons. | all | current documentation |
| `build_rivet.sh` | Builds `pythia_rivet` and the Rivet plugin against LCG_106 (Pythia 8.312 / Rivet 4.0.0). Products live at directory top level. | legacy matched gridpack | historical; never mix with `py313/` |
| `build_rivet_313.sh` | Builds both C++ drivers and the Rivet plugin against LCG_107 (Pythia 8.313 / Rivet 4.0.2), isolated under `py313/`. | HardQCD, internal, MESS | current shared build |
| `cp5.cmnd` | Common CMS CP5 tune fragment prepended before each variation-specific knob. Documents the unavailable NNPDF3.1 shower-PDF caveat. | all | shared physics configuration |
| `pythia_rivet.cc` | Generic Pythia-to-HepMC3-to-Rivet driver; applies MLM hooks when configured and drops LHE multiweights. | legacy, HardQCD, internal | shared driver |
| `pythia_rivet_mess.cc` | Dedicated MESS + Vincia driver; removes zero-weight merging trial events before Rivet and records accumulated cross sections. | MESS | current |
| `submit_all.sh` | Creates `joblist.txt` and submits the original same-LHE/many-showers QCD-HT gridpack wave. Vincia is excluded by default after the MLM-hook defect was found. | legacy matched gridpack | historical |
| `submit_hadronic.jdl` | HTCondor description consumed by `submit_all.sh`; transfers the LCG_106 products and runs `inner_hadronic.sh`. | legacy matched gridpack | historical |
| `inner_hadronic.sh` | Worker that generates one official UL QCD-HT LHE chunk, showers the same events under several configs with Pythia 8.312 MLM, runs Rivet, and stages ntuples/YODA to EOS. | legacy matched gridpack | historical |
| `diag_matching.sh` | Runs a single shared LHE through selected legacy configs and reports MLM acceptance and selected-jet counts. | legacy matched gridpack | historical diagnostic |
| `diag_shape.py` | Text diagnostic comparing normalized groomed-rho shapes from a `diag_matching.sh` work directory. | legacy matched gridpack | historical diagnostic |
| `harvest_cp5_reference.py` | Pools matched-gridpack CP5 chunks once per QCD-HT slice using the combined generated-luminosity denominator. | legacy matched gridpack | current reference utility |
| `submit_hardqcd.sh` | Builds a pThat-sliced, variance-oriented HardQCD job list and submits one config per job. | HardQCD | historical production |
| `submit_hardqcd.jdl` | HTCondor description used by `submit_hardqcd.sh`; transfers the LCG_107 products from `py313/`. | HardQCD | historical production |
| `inner_hardqcd.sh` | Generates direct `HardQCD:all` events in pThat slices, showers paired seeds under each config, runs Rivet, and stages outputs. | HardQCD | historical production |
| `submit_hardqcd_diag.jdl` | Older multi-config diagnostic submit description using top-level binaries and `joblist_diag.txt`. | HardQCD | superseded by `submit_hardqcd.jdl` plus explicit small job lists |
| `diag_slice_weights.py` | Reports each pThat slice's yield and variance contribution, used to tune the HardQCD allocation. | HardQCD | historical diagnostic |
| `harvest_shapes.py` | Generic pre-manifest harvester for QCD-HT or pThat ntuples; stitches slices through their `xsec_pb/sumw` footers and writes shape arrays. | legacy, HardQCD | historical harvester |
| `plot_shapes.py` | Produces CMS-style groomed-rho shape and variation/CP5 ratio figures from `harvest_shapes.py` output. | legacy, HardQCD | historical plotter |
| `plot_me_closure.py` | Compares standalone HardQCD CP5 with matched MadGraph-MLM CP5 to quantify the hard-ME difference. | HardQCD vs legacy | historical diagnostic |
| `submit_internal_variations.sh` | Creates deterministic validation or five-million-event job lists for CP5, CR1, CR2, frag-hard, and frag-soft, then submits them. | internal variations | current |
| `submit_internal_variations.jdl` | HTCondor description for the manifest-gated internal-variation worker. | internal variations | current |
| `inner_internal_variations.sh` | Generates one official UL QCD-HT LHE chunk, showers identical hard events under all five configs with Pythia 8.313 MLM, validates outputs, and stages the completion manifest last. | internal variations | current |
| `validate_internal_campaign.py` | Enforces manifest counts, versions, hashes, config order, finite cross sections/sumw, ntuple row counts, and output integrity before harvest. | internal variations | current gate |
| `harvest_internal_variations.py` | Reads only validated manifests and pools raw histograms plus generated-luminosity denominators per HT slice. | internal variations | current harvester |
| `plot_internal_shapes.py` | Plots the pooled CP5 and four internal-variation groomed-rho shapes with a ratio panel and bootstrap uncertainties. | internal variations | current plotter |
| `plot_internal_variations.py` | Produces paired-bootstrap CR/fragmentation-to-CP5 ratio figures from the manifest-gated harvest. | internal variations | current plotter |
| `submit_mess.sh` | Creates deterministic validation, one-million, or additive five-million MESS job lists and submits them. | MESS | current |
| `submit_mess.jdl` | HTCondor description for the manifest-gated MESS worker. | MESS | current |
| `inner_mess.sh` | Generates an official UL QCD-HT LHE chunk, runs MESS + Vincia with Pythia 8.313, validates local products, and stages the completion manifest last. | MESS | current |
| `validate_mess_campaign.py` | Checks validation/production counts, unique seeds, versions and hashes, nonzero ntuple weights, cross sections, and referenced output files. | MESS | current gate |
| `harvest_mess.py` | Reads complete MESS manifests and pools chunks per HT slice with the correct generated-luminosity normalization. | MESS | current harvester |
| `plot_mess_comparison.py` | Compares matched CP5 and MESS+Vincia groomed-rho shapes and summarizes campaign statistics. | MESS | current plotter |
| `build_pairsplit_313.sh` | Builds the dedicated pair_split Rivet plugin and MESS driver against LCG_107 and freezes both sources under `py313/`. | MESS pair_split dijet | historical build |
| `inner_mess_pairsplit.sh` | Runs one official QCD-HT LHE chunk through MESS+Vincia and the pair_split selection; records normalization, selection counters and per-product hashes, then stages the manifest last. | MESS pair_split dijet | historical worker; repair failure logging before reuse |
| `submit_mess_pairsplit.sh` / `.jdl` | Deterministic six-bin canary, pilot and 8M-event full job lists for the pair_split dijet campaign. | MESS pair_split dijet | historical campaign recipe |
| `audit_mess_pairsplit.py` | Manifest/product/hash/normalization audit and frozen allowlist builder. | MESS pair_split dijet | historical helper; missing-evidence and empty-campaign checks need repair before reuse |
| `mess_campaign_status.py` / `wait_mess_threshold.sh` | Campaign progress counter and threshold watcher. | MESS pair_split dijet | historical helpers; watcher has no deadline and needs failure handling |
| `plot_pairsplit_dijet_mass.py` | Earlier dijet mass comparison from a frozen MESS allowlist. | MESS pair_split dijet | historical prototype; superseded by `plot_pairsplit_model_comparison.py` |
| `plot_pairsplit_model_comparison.py` | Both-channel mass/rho comparisons with saved arrays and provenance checks. | MESS pair_split dijet | historical August 14 snapshot; grid and report assumptions need review before reuse |
| `plot_internal_fullsim_reference.py` | FullSim-anchored CR/fragmentation and MESS-Vincia shapes with the appropriate denominator for each leg. | internal + MESS | combination plotter |

All per-job products, including the command and two logs, carry the HT bin and
seed in their filename. This is required for concurrent jobs in one HT-bin
directory; the original `pilot` phase predated that fix and is excluded from
harvests. The deterministic recovery phase is `pilot2`.

The audit accepts signed FastJet mass roundoff only in the tightly bounded
interval `[-1e-4, 0) GeV`, counts those rows by HT bin, and the plot harvest
maps them to physical zero. More-negative masses remain a hard failure; this is
not a groomed-mass floor or selection cut.

Generated `joblist*.txt`, `py313/` build products, Python caches, Condor logs,
and `results/` outputs are deliberately not source files.  They are ignored and
must not be treated as campaign-completion evidence; completion comes from the
validated manifests and the archived harvest/plot products described in
`LAUNCH.md`.
