# Sherpa 3 pair_split canary

This directory is isolated from `rivet/hadronic_prod` and contains no full
production launcher.  It implements the current hadronic `pair_split`
particle-level fiducial only.  The historical fixed-dphi, pT-asymmetry and
trijet-veto selection is deliberately absent.

## Static checks

```bash
python3 scripts/check_static_contract.py CMS_HADRONIC_PAIR_SPLIT.cc
python3 scripts/render_card.py \
  --template configs/Sherpa.yaml.tmpl --output /tmp/Sherpa.yaml \
  --pt-low 480 --pt-high 570 --events 200 --slice-index 2 --shower Dire
```

## Runtime canary

Inside a pinned environment containing Sherpa 3, Rivet 4 and the required
FastJet contrib plugin:

```bash
./scripts/run_canary.sh canary/dire_pt480_570 200 2 480 570 Dire
```

The completion manifest is written only after generation, Rivet analysis,
selection-parity checks and output hashing succeed.

After copying the completed canary directory back to the analysis machine,
render the required CMS-style plots with the project environment:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
  /Users/aritra/Projects/smp_jetmass_run2/.venv/bin/python \
  scripts/plot_canary.py canary/dire_pt480_570 \
  --repository /Users/aritra/.codex/worktrees/3ef2/smp_jetmass_run2
```

The produced PNGs still require visual inspection before the canary can be
accepted.

`configs/campaign_2018_v1.json` deliberately contains
`"full_production_approved": false`.  There is no checked-in command that can
submit the full campaign.
