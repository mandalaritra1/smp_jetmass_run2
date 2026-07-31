# Rivet routines — jet mass

Two routines live here:

| Routine | Scope | Observables |
| --- | --- | --- |
| [`CMS_2026_PAS_SMP_25_010`](CMS_2026_PAS_SMP_25_010.cc) | Z+jet | `log10(rho^2)` in the published pT slices, groomed + ungroomed |
| [`CMS_ZJET_JETMASS`](CMS_ZJET_JETMASS.cc) | Z+jet | mass [GeV] + rho, finer binning, `CH3_NTUPLE` per-event dump |
| [`CMS_HADRONIC_JETMASS`](CMS_HADRONIC_JETMASS.cc) | dijet + trijet | mass + rho per hadronic pT bin, `HAD_NTUPLE` per-event dump |

`CMS_2026_PAS_SMP_25_010` is the one to publish/share: it carries the PAS name,
the HepData reference binning (pT `[200, 290, 400, inf]`, `log10(rho^2)`
`[-10, -4.5, -4, ..., 0]`), and matches the `zjet_processor.py` gen selection
cut for cut — validated against it on the same FullSim events, see
[closure/](closure/). Histogram names `zjets_ungroomed` / `zjets_groomed` are 2D
(pT × log10(rho²)), deliberately matching what `rappoccio/zjets_comparison`'s
`build_comparison.py` expects, so this routine is a drop-in replacement there.
The `_slice0/1/2` companions are the per-slice unit-area distributions.

A three-channel version (adding dijet and trijet, each matching its own
processor) exists on the `smp-25-010-match-analysis-selection` branch of
`gitlab.cern.ch/srappocc/Rivet` at commit `6e91362`, if those channels are
wanted once they have a measurement behind them.

`CMS_HADRONIC_JETMASS` (2026-07-31) is the local working routine for the
hadronic channels, written for the dijet/trijet model-uncertainty production.
It mirrors the CURRENT processor gen selections (`genTot_seq`), including the
trijet-priority orthogonality veto in dijet (>=3 jets, |y(j3)|<2.5, min
pairwise dphi > 1.0, pt3 > 185) that postdates the gitlab branch — earlier
three-channel sketches (dphi>2 + pT-ratio dijet cuts on |eta|, no veto) do
NOT match the processors and must not be used for reweight derivation. Dijet
fills both leading jets, trijet fills jet 3 only; `HAD_NTUPLE=<prefix>` dumps
`<prefix>_dijet.txt` / `<prefix>_trijet.txt` (one row per filled jet:
`jet_pt m_u m_g rho_u rho_g weight`, xsec/sumw footer). Validated: builds
under LCG_106 on lxplus (Rivet 4.0.0); 3k-event HardQCD smoke gives ~77%
dijet selection, orthogonal trijet fills, median groomed rho ~ -2.1.

Build it the same way as below, swapping the file name:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work hepstore/rivet-pythia:latest rivet-build RivetCMS_2026_PAS_SMP_25_010.so CMS_2026_PAS_SMP_25_010.cc
```

`CMS_ZJET_JETMASS` stays as the working/diagnostic routine (it keeps the
per-event ntuple hook used for the Herwig CH3 reweight). The rest of this file
describes it.

---

`CMS_ZJET_JETMASS` is a particle-level Rivet routine that reproduces the
**generator-level (fiducial) selection** of the `zjet` channel
(`QJetMassProcessor`, gen path in
[`smp_jetmass_run2/zjet_processor.py`](../smp_jetmass_run2/zjet_processor.py)).
Use it to compare generators / tunes against the unfolded measurement and for
analysis preservation.

## Fiducial definition (mirrors the coffea gen selection)

| Object | Cut |
| --- | --- |
| Dressed leptons (dR=0.1, no tau ancestors) | e: pT>40, μ: pT>29, \|η\|<2.4 |
| Z boson | exactly 2 same-flavour OS dressed leptons; pT_Z>90; 71<m_ll<111 |
| AK8 jets (anti-kT, R=0.8) | \|y\|<2.4; ΔR(jet,lepton)>0.4; ≥1 jet |
| Candidate jet | highest-pT cleaned jet |
| Topology | Δφ(Z,jet)>1.57; \|pTZ−pTjet\|/(pTZ+pTjet)<0.3 |
| Observables | ungroomed mass, soft-drop mass (β=0, z_cut=0.1), ρ=2·log₁₀(m/(pT·R)), R=0.8 |
| Jet-pT bins | [200, 290, 400, ∞] GeV |

Jets are clustered from all visible final-state particles (neutrinos excluded),
matching CMS `slimmedGenJetsAK8`; jets overlapping a dressed lepton are then
removed by the ΔR<0.4 cleaning — the same two-step logic as the processor.

## Tested environment

Compiled and run-tested against **Rivet 4.1.3 + Pythia 8** in the
`hepstore/rivet-pythia` Docker image (the routine uses the Rivet 4 API:
`LeptonFinder`, YODA2). On an Apple-Silicon Mac run the amd64 image under
emulation with `--platform linux/amd64`.

## Build

```bash
cd rivet
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
  hepstore/rivet-pythia:latest \
  rivet-build RivetCMS_ZJET_JETMASS.so CMS_ZJET_JETMASS.cc
```

`rivet-build` links FastJet + fjcontrib (SoftDrop) automatically. On a native
Rivet install (LCG/cvmfs, lxplus) just run `rivet-build ...` directly and
`export RIVET_ANALYSIS_PATH=$PWD:$RIVET_ANALYSIS_PATH`.

## Generator-comparison suite

Run cards for six generator/shower setups live in [`gen/`](gen/), each a LO
Z(→ℓℓ)+jet sample with a hard ME-level jet cut (150 GeV, below the 200 GeV
fiducial threshold so the lowest measured bin is unbiased):

| Setup | Card | Image / env |
| --- | --- | --- |
| Pythia8 (simple shower) | [gen/pythia.cmnd](gen/pythia.cmnd) | `hepstore/rivet-pythia` |
| Pythia8 (Vincia shower) | [gen/vincia.cmnd](gen/vincia.cmnd) | `hepstore/rivet-pythia` |
| Herwig7 | [gen/herwig.in](gen/herwig.in) | `hepstore/rivet-herwig` |
| Sherpa 3 | [gen/sherpa.yaml](gen/sherpa.yaml) | `hepstore/rivet-sherpa` |
| MG5 LO + Pythia8 | [gen/madgraph/](gen/madgraph/) | cvmfs/LPC (MG5) |
| MG5 LO + Herwig7 | [gen/madgraph/](gen/madgraph/) | cvmfs/LPC (MG5) |

### Drivers

- **[`run_all.sh`](run_all.sh)** `<pythia|vincia|herwig|sherpa> [nevents] [seed]`
  — portable native driver (assumes the tools are on `PATH`, e.g. cvmfs/LPC):
  generate → (HepMC →) `rivet` → `out/<gen>.yoda`.
- **[`run_local.sh`](run_local.sh)** `<gen> [nevents]` — runs `run_all.sh` inside
  the matching amd64-emulated Docker image (for Macs without a native stack).
- **[`plot.sh`](plot.sh)** `[outdir]` — overlays every `out/*.yoda` into
  shape-comparison plots (area-normalised via [gen/compare.plot](gen/compare.plot)).

```bash
cd rivet
for g in pythia vincia herwig sherpa; do ./run_local.sh $g 20000; done
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
  hepstore/rivet-pythia:latest ./plot.sh          # -> out/plots/index.html
```

The four native generators are validated locally (z_mass peaks at 91 GeV for all;
the ungroomed-mass mean spans ~46 GeV (Herwig) to ~57 GeV (Sherpa) — a realistic
generator spread).

### MadGraph LO (run on cvmfs/LPC)

There is no hepstore MadGraph image and MG5 is slow under emulation, so the two
`mglo` setups are prepared but not run locally. On LPC:

```bash
source /cvmfs/...   # environment with mg5_aMC, pythia8, Herwig, rivet
./gen/madgraph/run_mg.sh       # -> out/mglo_pythia.yoda, out/mglo_herwig.yoda
./plot.sh                      # re-overlay including the MG curves
```

`run_mg.sh` makes one LHE file with [gen/madgraph/mg5_zjet.dat](gen/madgraph/mg5_zjet.dat)
and showers it two ways ([shower_pythia.cmnd](gen/madgraph/shower_pythia.cmnd),
[shower_herwig.in](gen/madgraph/shower_herwig.in)). Comments in those files show
how to extend the single-multiplicity LO sample to a full MLM-merged
madgraphMLM one.

### Modelling-systematic variations (Pythia CR + hadronization)

For the ARC's "parton shower vs hadronisation" modelling question, the suite adds
Pythia-family variation cards that change **one** ingredient at a time on top of the
nominal [gen/pythia.cmnd](gen/pythia.cmnd) (same ME + simple shower + default/Monash
tune), so each jet-mass shape difference isolates that effect:

| Card | Varies | Setting |
| --- | --- | --- |
| [gen/pythia_cr1.cmnd](gen/pythia_cr1.cmnd) | colour reconnection | `ColourReconnection:mode = 1` (QCD-inspired) |
| [gen/pythia_cr2.cmnd](gen/pythia_cr2.cmnd) | colour reconnection | `ColourReconnection:mode = 2` (gluon-move) |
| [gen/pythia_fragsoft.cmnd](gen/pythia_fragsoft.cmnd) | hadronization | softer Lund FF (`StringZ:aLund/bLund`↑, `StringPT:sigma`↑) |
| [gen/pythia_fraghard.cmnd](gen/pythia_fraghard.cmnd) | hadronization | harder Lund FF (same params ↓) |

The clean **shower-only** handle is the existing nominal vs Vincia pair
([gen/vincia.cmnd](gen/vincia.cmnd), same hadronization). Run and overlay:

```bash
for g in pythia pythia_cr1 pythia_cr2 pythia_fragsoft pythia_fraghard; do
  ./run_slices.sh "$g" 20000     # flat high-pT stats (or ./run_all.sh "$g")
done
./plot_syst.sh                   # -> out/plots_syst/index.html
```

These are mode-flips / exploratory parameter brackets on the default tune — *not*
the retuned CMS CP5-CR1/CR2 tunes or CP5 eigentune uncertainties. For the
publication modelling systematic, substitute those (CP5 tune paper,
[arXiv:1903.12179](https://arxiv.org/abs/1903.12179)); the card comments flag this.

### Single quick run (smoke test)

`test/zjet.cmnd` is a minimal standalone example:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
  hepstore/rivet-pythia:latest bash -lc '
    export RIVET_ANALYSIS_PATH=/work
    pythia8-main144 -c test/zjet.cmnd -n 1500 -o test/zjet -l
    rivet -a CMS_ZJET_JETMASS test/zjet.hepmc -o test/zjet.yoda
    rivet-mkhtml test/zjet.yoda -o test/zjet-plots
  '
```

1500 events: 604 pass the full selection; pT bins (385+170+49) sum to 604;
z_mass peaks at 91.2 GeV; Z/jet pT balance (~285/292); groomed < ungroomed mass.

## Notes / TODO

- **Status: UNVALIDATED.** Rename to the official `CMS_<year>_I<inspire>` ID and
  fill `BibKey`/`BibTeX`/HepData references once the paper exists.
- The reco-level corrections (JES/JEC, JMS/JMR, rocco, lepton SFs) live only in
  the coffea reco path and have no particle-level analogue — this routine is gen
  only, as Rivet requires.
- Soft-drop mass here is computed by reclustering the AK8 constituents with
  Cambridge/Aachen and applying SoftDrop(β=0, z_cut=0.1). The processor instead
  reads CMS `SubGenJetAK8` soft-drop subjets (same algorithm/parameters); small
  differences from subjet definition are expected at the few-percent level.
