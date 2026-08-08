# Dijet/trijet model-uncertainty production — launch recipe (lxplus condor)

Same-ME path as the zjet Vincia production: official UL **QCD_HT madgraphMLM
gridpack** LHE → standalone **Pythia 8.312 (LCG_106)** + MLM matching →
Rivet **CMS_HADRONIC_JETMASS** (mirrors the current dijet/trijet processor
gen selections, incl. the trijet-priority veto) → yoda + per-jet ntuples per
HT bin.  Six configs in the SAME framework, all on the **CP5 base**, so each
ratio to `cp5` isolates one effect:

| config | knob | leg |
|---|---|---|
| `cp5`      | (reference)                                        | — |
| `vincia`   | `PartonShowers:model = 2`                          | PS envelope (with in-sample FSR) |
| `cr1`      | `ColourReconnection:mode = 1` + `remnantMode = 1`  | HAD envelope |
| `cr2`      | `ColourReconnection:mode = 2`                      | HAD envelope |
| `fraghard` | `StringZ:aLund/bLund = 0.58/0.78`, `sigma = 0.305` | HAD envelope |
| `fragsoft` | `StringZ:aLund/bLund = 0.78/1.18`, `sigma = 0.365` | HAD envelope |

MLM parameters are copied from the official UL fragment
(`JME-RunIISummer20UL18wmLHEGEN-00015`): **qCut=14, setMad=off**, scheme=1,
jetAlgorithm=2, etaJetMax=5, coneRadius=1, slowJetPower=1, nQmatch=5,
nJetMax=4.  ≠ the DY production (qCut=19, setMad=on) — don't cross-copy.

HT bins: 300to500 … 2000toInf (6). Bins below 300 contribute nothing at
jet pT > 185 (a 185 GeV dijet already has parton HT ≈ 370).

## Push (from Mac)
```
rsync -av ~/Projects/smp_jetmass_run2/rivet/hadronic_prod/ lxw:~/hadronic_model_prod/
rsync -av ~/Projects/smp_jetmass_run2/rivet/CMS_HADRONIC_JETMASS.cc lxw:~/hadronic_model_prod/
```

## On lxplus (interactive login needed for condor_submit)
```
cd ~/hadronic_model_prod
bash build_rivet.sh            # builds pythia_rivet + RivetCMS_HADRONIC_JETMASS.so (LCG_106)
# CANARY first (cp5+vincia, 1 job/bin, 800 ev -> 12 jobs):
bash submit_all.sh canary
condor_watch_q
# check a yoda + ntuples land in /eos/user/a/amandal/hadronic_model_prod/{cp5,vincia}/QCD_HT*/
# and note per-bin selected-jet counts (dijet + trijet rows) to size the full wave, then:
bash submit_all.sh full 8 5000   # 6 configs x 6 bins x 8 jobs = 288 jobs
```

Each job stages out a yoda plus `ntuple_*_dijet.txt` / `ntuple_*_trijet.txt`
(`jet_pt m_u m_g rho_u rho_g weight` rows + `# xsec_pb` / `# sumw` footer).

## Canary checklist (before the full wave)
- `rho_g_pt*` filled in both channels' hists; dijet/trijet row counts sensible.
- Standalone `cp5` groomed-rho shape vs the mg_pythia8 processor gen output
  (`ptjet_rhojet_g_gen`) — same ~4-5% level of agreement as the zjet
  vincia_prod validation. This is the closure that licenses the ratios.
- Matching acceptance per bin from the log (`Fraction of events accepted`,
  or LHE-events vs ntuple-rows) — sizes the full wave.
- Trijet stats: jet-3 fills at pT>185 come mostly from HT >= 700; if the
  canary shows starvation, weight JOBS_PER toward the high-HT bins rather
  than raising NEVT (short jobs survive better).

## Harvest (locally, once outputs sync to `~/cernbox (2)/hadronic_model_prod`)
Per-config ntuples → per-channel gen-rho ratios varied/cp5 on a finer-than-
analysis gen binning (halved bins, floor/clip as the zjet weighters:
w=1 where cp5 frac < 1e-4 or varied raw count < 25, clip [0.2, 5]) →
`{vincia,cr1,cr2,fraghard,fragsoft}_rho_reweight_{groomed,ungroomed}.npz`
per channel. Stitch HT bins by xsec_pb/sumw from the footers.

## Notes / invariants (inherited from the DY wave — each cost a failed wave)
- gridpack extracted to `$TMPDIR` on the worker, never AFS (quota bomb).
- condor .out/.err/.log → /dev/null (an AFS-resident log held every job once).
- `periodic_release` self-heals transient segfaults/evictions; short jobs
  (~5000 ev) + fresh-seed retries beat long jobs.
- runcmsgrid yields fewer LHE events than requested — size from the canary.
- Never trust GenXsecAnalyzer-style xsecs here; normalize per bin via the
  `# xsec_pb`/`# sumw` footers (LO gridpack xsec), ratios cancel most of it.
- Non-interactive agent sessions CANNOT `condor_submit` on lxplus — a human
  fires the submits; everything else here is agent-preparable.
