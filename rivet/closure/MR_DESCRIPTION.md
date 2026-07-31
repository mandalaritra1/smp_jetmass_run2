# Match the SMP-25-010 plugin to the analysis fiducial selection

Target: `srappocc/Rivet` master. Modifies `SMP/src/CMS_2026_PAS_SMP_25_010.cc`,
adds `SMP/data/*.info` / `*.plot` and a `SMP/test` cmsRun configuration.

The plugin now reproduces the particle-level selection of the coffea processor
the measurement is actually made with (`zjet_processor.py` in
`smp_jetmass_run2`), and is restricted to the Z+jet channel: the dijet and
trijet channels have no unfolded measurement behind them yet, so they do not
belong in a plugin named after the PAS.

## Z+jets — what changed and why it matters

| | before | after |
| --- | --- | --- |
| topology | ΔR(Z, jet) > 1.0 | **Δφ(Z, jet) > 1.57 AND pT asymmetry < 0.3** |
| pT(Z) | no cut | **> 90 GeV** |
| lepton pT | 20 GeV both | **e 40, μ 29 GeV** |
| lepton count | any dilepton candidate, μ preferred | **exactly two SFOS, no third lepton** |
| jet inputs | `FinalState` — neutrinos clustered into jets | **`JetInvisibles::NONE`** |
| jet acceptance | \|η\| < 2.5 | **\|y\| < 2.4** |
| lepton overlap | Z decay products vetoed from the jet input | **jets within ΔR < 0.4 of a dressed lepton removed after clustering** |
| candidate jet | first jet passing the ΔR cut (falls back to jet 2) | **leading jet; event rejected if it fails the topology cut** |

The missing pT-asymmetry requirement is the important one: it is what makes the
sample a balanced Z+jet topology. On 10k identical Pythia Z+jet events the old
selection keeps **26% more jets** (1.28 / 1.21 / 1.21 in the three pT slices) and
the normalized shapes drift by up to ~10% in the outer bins of the higher-pT
slices — comparable to the generator differences the comparison is meant to
resolve.

The lepton-overlap change is subtle but real: clustering the Z decay products
into the jets and then removing overlapping jets is not the same jet sample as
vetoing them from the jet input.

## Histograms

`zjets_{ungroomed,groomed}` keep their names and the HepData binning
(pT `[200, 290, 400, ∞]`, `log10(ρ²)` `[-10, -4.5, -4, …, 0]`), so downstream
tooling that reads the 2D histograms is unaffected. Added `_slice{0,1,2}`
companions normalized to unit area **over the shown bins** — overflows are
excluded on purpose, because single-prong soft-drop jets sit far below the axis
and are outside the measurement, exactly as they are excluded from the unfolded
result.

## Validation

Run against the FullSim sample the measurement uses:
`DYJetsToLL_M-50_HT-400to600` UL18, one NanoAODv9 file (534146 events) and its 12
MiniAODv2 parents. Each parent's only v9 child is that file, so the Rivet side
(MiniAOD → HepMC) and the coffea gen path (NanoAOD) see an identical event set by
construction. CMSSW_15_0_9, Rivet 4.0.2.

- **ungroomed closes**: normalized `log10(ρ²)` agrees bin by bin to better than
  0.4% in every populated bin across all three pT slices; pT-slice fractions
  agree to 1e-4 (`[0.42894, 0.39861, 0.17246]` vs `[0.42891, 0.39863, 0.17246]`).
- **groomed agrees to 0.3–2%**, in the direction expected from the grooming
  definition: the plugin soft-drops the C/A-reclustered constituents, the coffea
  gen path sums the `SubGenJetAK8` pair.

Closure figures: `closure_{un,}groomed_pt{0,1,2}.png`.

## Note for the coffea side (not part of this MR)

`allsel_gen` in `zjet_processor.py` includes `is_matched_gen` (gen jet within
ΔR < 0.4 of the reco jet), so the histograms named `*_gen` are the *matched*
truth, not the fiducial gen spectrum, and cannot be compared to a particle-level
prediction directly. The gen matching efficiency is 0.872 / 0.945 / 0.961 in the
three pT slices and shifts the slice composition by ~5%. The closure above uses a
copy of the processor with that term dropped.
