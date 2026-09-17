# Hadronic sample lists (dijet / trijet)

The hadronic channels use the original GluonJetMass JSON fileset format. The
runner filters these JSONs by era and then runs the selected DAS dataset keys
directly, preserving the full dataset name on histogram axes for postprocess
normalization.

- **Data:** `fileset_JetHT_wRedirs.json` or `fileset_JetHT.json`. The dijet/trijet
  processors apply the AK8PFJet trigger prescales bundled in
  `smp_jetmass_run2/corrections/prescales/`.
- **MC:** `fileset_QCD*_*.json`, `fileset_MG_pythia8*.json`, and
  `fileset_HERWIG_wRedirs.json`, matching the keys in `corrections.py`
  (`xsdb`, `sumw_qcd_mg`, `sumw_herwig`).
- **Flat-pT MC (standalone shower, no MLM):** `fileset_HERWIG_FLAT_wRedirs.json`
  (`QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7`, 700 files, 100M events) and
  `fileset_PYTHIA_FLAT_wRedirs.json` (`QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8`,
  20 files, 20M events), UL18 NanoAODv9 only, datasets `herwig_flat` /
  `pythia_flat` (2026-09-17). `genWeight` is the pT-hat weight (spread of
  ten orders of magnitude, so watch the effective statistics per pT bin); the
  on-the-fly sumw normalization handles it. Built from `dasgoclient` file lists
  with the `root://cmsxrootd.fnal.gov/` prefix, like the HERWIG fileset.

Dataset names must contain the era token (`UL18`/`UL2018`, `UL17`, `APV`, etc.) so
the processor can infer the IOV, exactly as in GluonJetMass.
