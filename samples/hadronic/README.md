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

Dataset names must contain the era token (`UL18`/`UL2018`, `UL17`, `APV`, etc.) so
the processor can infer the IOV, exactly as in GluonJetMass.
