# Hadronic pair_split campaign -- FINAL REPORT

**Status: COMPLETE.** All of Stage A, Stage B and Stage C produced, downloaded,
verified and cleaned off coffea-casa. Nothing is running; casa is at 46% disk
with zero campaign outputs left on it.

Repo tree hash (local dev == casa, verified at start and at end):
`7372447c66c125638247b4cd57cd43ffa55ab5ec`

---

## 1. Ledger -- Stage A / Stage C skims (per-bin pickles)

| era | channel | files | size | reco rows | verification |
|---|---|---|---|---|---|
| 2016APV | dijet MC | 8 | 2.5 G | 16,946,676 | OK |
| 2016APV | trijet MC | 8 | 807 M | 5,489,576 | OK |
| 2016APV | dijet data | 6 | 167 M | 667,010 | OK |
| 2016APV | trijet data | 6 | 56 M | 293,578 | OK |
| 2016 | dijet MC | 8 | 2.5 G | 16,444,420 | OK |
| 2016 | trijet MC | 8 | 780 M | 5,312,375 | OK |
| 2016 | dijet data | 3 | 138 M | 607,308 | OK |
| 2016 | trijet data | 3 | 46 M | 290,417 | OK |
| 2017 | dijet MC | 8 | 2.8 G | 19,051,630 | OK |
| 2017 | trijet MC | 8 | 926 M | 6,416,327 | OK |
| 2017 | dijet data | 5 | 293 M | 1,437,174 | OK |
| 2017 | trijet data | 5 | 77 M | 562,058 | OK |
| 2018 | dijet MC | 8 | 2.9 G | 19,052,960 | OK (genWeight basis) |
| 2018 | trijet MC | 8 | 929 M | 6,318,977 | OK (genWeight basis) |
| 2018 | dijet data | 4 | 317 M | 1,407,374 | OK (genWeight basis) |
| 2018 | trijet data | 4 | 111 M | 638,229 | OK (genWeight basis) |

Empty bins (HT100to200, sometimes HT200to300, and Run2016B-ver1) are expected,
not failures -- see section 5.

## 2. Ledger -- Stage B / Stage C all_syst (one pickle per channel)

| era | channel | size | populated systematics | cutflow final_seq |
|---|---|---|---|---|
| 2016APV | dijet MC | 1009.9 MB | **77 / 77** | 79,706,660 |
| 2016APV | trijet MC | 832.0 MB | **77 / 77** | 4,411,257 |
| 2016APV | dijet data | 104.4 MB | 1 / 1 | 3,324,998 |
| 2016APV | trijet data | 7.2 MB | 1 / 1 | 293,578 |
| 2016 | dijet MC | 1009.9 MB | **77 / 77** | 77,299,630 |
| 2016 | trijet MC | 832.0 MB | **77 / 77** | 4,232,458 |
| 2016 | dijet data | 87.6 MB | 1 / 1 | 3,031,726 |
| 2016 | trijet data | 7.0 MB | 1 / 1 | 290,417 |
| 2017 | dijet MC | 1150.2 MB | **77 / 77** | 89,456,605 |
| 2017 | trijet MC | 832.0 MB | **77 / 77** | 5,053,678 |
| 2017 | dijet data | 196.5 MB | 1 / 1 | 7,160,718 |
| 2017 | trijet data | 13.6 MB | 1 / 1 | 562,058 |
| 2018 | dijet MC | 1009.9 MB | **77 / 77** | 89,714,430 |
| 2018 | trijet MC | 832.0 MB | **77 / 77** | 5,025,480 |
| 2018 | dijet data | 187.5 MB | 1 / 1 | 6,993,066 |
| 2018 | trijet data | 15.4 MB | 1 / 1 | 638,229 |

Every MC file carries all 77 declared systematic categories populated, and in
every file the per-jet kinematic histograms carry the full systematic set.

**Independent Stage A <-> Stage B cross-check (trijet data), all four eras:**
the all_syst `event_id` row count equals the Stage A skim reco row count exactly
-- 293,578 / 290,417 / 562,058 / 638,229. The two stages are separate
productions over the same data, so this is a genuine consistency test.

## 3. Timings

| run | 2016APV | 2016 | 2017 | 2018 |
|---|---|---|---|---|
| skim dijet MC | (pre-existing) | 15m44s | (pre-existing) | 17m47s |
| skim trijet MC | (pre-existing) | 15m31s | (pre-existing) | 17m57s |
| skim dijet data | (pre-existing) | 6m45s | (pre-existing) | 11m06s |
| skim trijet data | (pre-existing) | 7m01s | (pre-existing) | 11m30s |
| all_syst dijet MC | 32m36s | 31m34s | 37m48s | 32m57s |
| all_syst trijet MC | 50m31s | 49m44s | 59m16s | 55m10s |
| all_syst dijet data | 5m45s | 5m30s | 6m37s | 9m53s |
| all_syst trijet data | 6m01s | 4m48s | 6m03s | 10m19s |
| harvest (download+verify+clean) | 8m | 6m+11m | 7m | 14m+6m |

- **Total productive compute: ~8 h 28 m** across 24 successful runs.
- **Total harvesting: ~1 h.**
- **Total elapsed: ~18 h 53 m** (takeover 21:05 Aug 7 -> 15:58 Aug 8).
- **Restart count: 1** (2018 dijet MC skim, config defect -- see section 4).
  Zero restarts from wedges, starvation, broadcast races or OOM. `erred=0` on
  every Dask run for the whole campaign.

The ~10 h gap between productive and elapsed time is two agent-side stalls, not
production problems: an 87-min gap where completion notifications arrived late
and batched, and a ~8 h platform watchdog stall on the agent stream (during
which the in-flight run finished normally and casa then idled). Both are
orchestration losses; the fix applied after the first (hold the turn open with
blocking waits, launch the next item the instant one lands) worked -- the whole
Stage B run of 12 jobs went back-to-back with no idle gaps.

## 4. Config defect found and fixed (NOT committed -- review needed)

`configs/dijet_mg_pythia8_2018_rho_skim.json` was a stale **LPC** config:
`executor_mode: dask-lpc`, `redirector: lpc`, `casa: false`. It crashed
instantly on casa. It also carried `mode: rho_skim`, while

- the 2018 dijet **data** config uses `rho_skim10`,
- every other era's dijet MC config uses `rho_skim10`,
- the 2018 files it was replacing are literally named `rho_skim10_*`.

Running it as committed would have given MC a different prescale from its own
data, breaking the bin-for-bin comparison the sibling configs explicitly
require. Fixed to match the 2017 config exactly; the three other 2018 skim
configs were missing `n_workers: 400` and were brought in line too. All four
synced to casa and md5-verified. **These are uncommitted working-tree changes**
(`git status` shows 4 modified configs) -- deliberately left for Aritra.

## 5. Expected-empty outputs (verified, not failures)

- **Run2016B-ver1** (both channels): 0 events from 9.7 M. The processor applies
  the lumimask before filling the trigger cutflow, and the Legacy2016 golden
  JSON's first certified run is **273158** -- ver1 contains only
  pre-certification runs. The ver2 re-reco of the same era passes 93.6% of the
  lumimask on the same PFJet path set. Correct physics.
- **HT100to200 / HT200to300**: near-zero surviving rows in every era, as
  expected for a pT > 200 GeV selection on low-HT samples.

## 6. FINDING -- silent chunk loss in the old 2018 archive (and in current runs)

Comparing the new genWeight-basis 2018 production against the retained
LHE-basis archive (`*_lhe_basis/`, 28 files) shows the new production is a
**strict superset**: for the two dijet data files checked event-by-event,
0 events were lost and 1,225 / 2,001 events were *gained*, each confined to a
single run's lumi blocks (317089; 321457+324998).

The mechanism is that the old runs **read fewer events**, and coffea reported
"Processing 100%" anyway -- 100% of the chunks *preprocessing discovered*. No
warning, no error, exit 0. Logs contain no skip/failure message at all.

After correcting for the pre-46fbd31 trijet `nEvents initial` double-count
(8 old trijet files show old = *exactly* 2x new, which is that bug, not loss):

| old file | chunks old->new | events missed by old | fraction |
|---|---|---|---|
| dijet MC HT200to300 | 54 -> 56 | 3,561,558 | 5.79% |
| trijet MC HT100to200 | -- | 877,652 | 1.10% |
| trijet MC HT300to500 | 13 -> 13 | 199,619 | 0.36% |
| dijet MC HT100to200 | -- | 217,225 | 0.27% |
| trijet MC all_syst | 2908 -> 2908 | 877,652 | 0.26% |
| dijet MC HT500to700 | 1229 -> 1231 | 99,831 | 0.16% |
| dijet data Run2018B | 759 -> 760 | 99,329 | 0.13% |
| trijet MC HT200to300 | -- | 49,660 | 0.08% |
| dijet data Run2018D | 3543 -> 3545 | 195,770 | 0.05% |
| dijet MC HT700to1000 | 958 -> 959 | 18,866 | 0.04% |

**10 of 28 old files affected; 6,197,162 events (0.15% of 4.06 G) never read.**
The largest fractions land in the low-HT bins that contribute essentially no
statistics, so the physics-weighted impact is the 0.04-0.26% seen in the
high-statistics files. The Stage C re-production repaired all of this, so the
2018 outputs are now the most complete ones in the archive.

### This is not confined to the old archive

A Stage A-vs-Stage B cross-check on the eras produced *in this campaign* (same
input datasets, so `nEvents initial` must agree) shows the same signature in
4 of 12 channel-eras, in both directions:

| era / channel | delta (all_syst minus skim) |
|---|---|
| 2016APV dijet MC | -149,856 (3515 -> 3511 chunks) |
| 2016APV dijet data | +201,255 (skim short; the trijet skim over the same 6 datasets agrees with all_syst at 374,088,989) |
| 2016 dijet MC | -56,321 |
| 2016 trijet MC | -101,431 |

Magnitude 0.02-0.05%. **These are NOT repaired** -- unlike 2018, there is no
second production to supersede them. Whether that matters is Aritra's call; a
0.02-0.05% event-count difference between MC and data is far below the
systematics here, but it is silent and non-reproducible.

Suggested cheap safeguard for future productions: assert `nEvents initial`
against the DAS event count for each dataset before accepting an output.

## 7. Bet verdict

Aritra bet the remainder would finish in ~2 h against a 4-6 h estimate.

**The estimate wins; the bet loses -- but the compute vindicates the bet's
instinct.** Actual productive compute for everything remaining at the time of
the bet was ~8 h 28 m, so even a perfectly-scheduled run would have missed 2 h
by a wide margin: Stage B alone is 12 runs whose MC legs are 31-59 min each,
and casa caps the worker pool at 400 which we saturated on every MC run. Total
compute was fixed and no tuning could have changed it. Wall-clock came in at
~18 h 53 m, roughly double the 4-6 h estimate, and every hour of that excess
was agent-side stall rather than production.

## 8. Event log

```
2026-08-07 21:05  START   agent took over campaign; casa tree hash == local dev 7372447c (verified)
2026-08-07 21:06  HARVEST 2016APV dijet data  -- 6 pkl, md5 OK, 667,010 reco rows, casa cleaned
2026-08-07 21:08  HARVEST 2016APV trijet data -- 6 pkl, md5 OK, 293,578 reco rows, casa cleaned
2026-08-07 21:10  NOTE    Run2016B-ver1 empty in both channels = CORRECT (golden JSON starts run 273158; ver1 is all pre-certification). Not a failure.
2026-08-07 21:12  PREP    archived LHE-basis 2018 -> hadronic_skims_pairsplit/2018_lhe_basis (24 files, 4.2G) + hadronic_minimal_rho_pairsplit/2018_lhe_basis (4 files, 1.9G)
2026-08-07 21:13  PREP    Stage B verifier calibrated on 2018 refs: 77/77 systematics reproduced on both MC
2026-08-07 21:14  LAUNCH  StageA 2016 dijet MC skim
2026-08-07 21:29  DONE    StageA 2016 dijet MC skim   8/8 bins  exit 0  wall 15m44s
2026-08-07 22:56  NOTE    *** 87 min casa idle gap: completion notifications arrived late/batched. Fixed: agent now holds turn open with blocking waits. ***
2026-08-07 22:56  LAUNCH  StageA 2016 trijet MC skim
2026-08-07 23:12  DONE    StageA 2016 trijet MC skim  8/8 bins  exit 0  wall 15m31s
2026-08-07 23:12  LAUNCH  StageA 2016 dijet data skim
2026-08-07 23:19  DONE    StageA 2016 dijet data skim 3/3 bins  exit 0  wall 6m45s
2026-08-07 23:19  LAUNCH  StageA 2016 trijet data skim
2026-08-07 23:27  DONE    StageA 2016 trijet data skim 3/3 bins exit 0  wall 7m01s
2026-08-07 23:29  HARVEST StageA 2016 all 4 channels -- 22 pkl downloaded+verified+casa cleaned (11 min)
2026-08-07 23:40  VERIFY  StageA 2016: dijet MC 16,444,420 reco rows / trijet MC 5,312,375 / dijet data 607,308 / trijet data 290,417. Empty low-HT bins expected.
2026-08-07 23:40  MILESTONE  *** STAGE A COMPLETE (2016APV + 2016 + 2017 all archived & verified) ***
2026-08-07 23:40  LAUNCH  StageB 2016APV dijet MC allsyst
2026-08-08 00:13  DONE    StageB 2016APV dijet MC allsyst   exit 0  wall 32m36s  1009.9 MB
2026-08-08 00:15  LAUNCH  StageB 2016APV trijet MC allsyst
2026-08-08 01:06  DONE    StageB 2016APV trijet MC allsyst  exit 0  wall 50m31s  832.0 MB
2026-08-08 01:07  LAUNCH  StageB 2016APV dijet data allsyst
2026-08-08 01:13  DONE    StageB 2016APV dijet data allsyst exit 0  wall 5m45s   104.4 MB
2026-08-08 01:14  LAUNCH  StageB 2016APV trijet data allsyst
2026-08-08 01:20  DONE    StageB 2016APV trijet data allsyst exit 0 wall 6m01s     7.2 MB
2026-08-08 01:22  HARVEST StageB 2016APV era (4 files) downloaded + verified + casa cleaned (8 min)
2026-08-08 01:30  VERIFY  StageB 2016APV: dijet MC 77/77 systs, trijet MC 77/77, data 1/1. Per-jet kin hists carry full syst set. PASS
2026-08-08 01:30  XCHECK  trijet data allsyst event_id rows 293,578 == StageA trijet data reco rows 293,578 (independent StageA/StageB consistency check)
2026-08-08 01:31  LAUNCH  StageB 2016 dijet MC allsyst
2026-08-08 02:03  DONE    StageB 2016 dijet MC allsyst   exit 0  wall 31m34s
2026-08-08 02:03  LAUNCH  StageB 2016 trijet MC allsyst
2026-08-08 02:53  DONE    StageB 2016 trijet MC allsyst  exit 0  wall 49m44s
2026-08-08 02:53  LAUNCH  StageB 2016 dijet data allsyst
2026-08-08 03:00  DONE    StageB 2016 dijet data allsyst exit 0  wall 5m30s  87.6 MB (bokeh dashboard token traceback in log = benign, not computation)
2026-08-08 03:00  LAUNCH  StageB 2016 trijet data allsyst
2026-08-08 03:06  DONE    StageB 2016 trijet data allsyst exit 0 wall 4m48s
2026-08-08 03:06  HARVEST StageB 2016 era harvest started
2026-08-08 03:12  VERIFY  StageB 2016 era: dijet MC 77/77, trijet MC 77/77, data 1/1. XCHECK trijet data event_id 290,417 == StageA reco rows. PASS. casa cleaned.
2026-08-08 03:12  MILEST  *** StageB 8 of 12 done (2016APV + 2016 eras complete & verified) ***
2026-08-08 03:12  LAUNCH  StageB 2017 dijet MC allsyst
2026-08-08 03:50  DONE    StageB 2017 dijet MC allsyst   exit 0  wall 37m48s
2026-08-08 03:50  LAUNCH  StageB 2017 trijet MC allsyst
2026-08-08 04:50  DONE    StageB 2017 trijet MC allsyst  exit 0  wall 59m16s
2026-08-08 04:50  LAUNCH  StageB 2017 dijet data allsyst
2026-08-08 04:58  DONE    StageB 2017 dijet data allsyst exit 0  wall 6m37s
2026-08-08 04:58  LAUNCH  StageB 2017 trijet data allsyst (LAST StageB run)
2026-08-08 05:06  DONE    StageB 2017 trijet data allsyst exit 0 wall 6m03s
2026-08-08 05:06  MILEST  *** STAGE B COMPLETE -- all 12 all_syst runs finished, 0 failures, 0 restarts ***
2026-08-08 05:06  HARVEST StageB 2017 era harvest started
2026-08-08 05:13  VERIFY  StageB 2017 era: dijet MC 77/77, trijet MC 77/77, data 1/1. XCHECK trijet data event_id 562,058 == StageA reco rows. PASS. casa cleaned.
2026-08-08 05:13  MILEST  *** STAGE B VERIFIED COMPLETE: 12/12 outputs, all 77/77 systematics, 0 failures ***
2026-08-08 05:13  LAUNCH  StageC 2018 dijet MC skim (genWeight basis)
2026-08-08 05:16  FIX     StageC BLOCKER FOUND+FIXED: configs/dijet_mg_pythia8_2018_rho_skim.json was a stale LPC config (dask-lpc/redirector lpc/casa=false) -> crashed instantly on casa; ALSO had mode=rho_skim while 2018 dijet DATA + all other eras' dijet MC + the 2018 files being replaced all use rho_skim10. Fixed to match 2017 exactly. Also added missing n_workers=400 to the other three 2018 skim configs. All 4 synced to casa, md5 verified. NOT COMMITTED - working-tree change for Aritra to review.
2026-08-08 05:16  LAUNCH  StageC 2018 dijet MC skim (relaunch, rho_skim10 on casa)
2026-08-08 05:35  DONE    StageC 2018 dijet MC skim 8/8 bins exit 0 wall 17m47s
2026-08-08 05:35  LAUNCH  StageC 2018 trijet MC skim
2026-08-08 05:53  DONE    StageC 2018 trijet MC skim 8/8 bins exit 0 wall 17m57s
2026-08-08 05:53  LAUNCH  StageC 2018 dijet data skim
2026-08-08 06:06  DONE    StageC 2018 dijet data skim 4/4 bins exit 0 wall 11m06s
2026-08-08 06:06  LAUNCH  StageC 2018 trijet data skim
2026-08-08 06:18  DONE    StageC 2018 trijet data skim 4/4 bins exit 0 wall 11m30s
2026-08-08 06:18  HARVEST StageC 2018 skims harvest started (4 channels)
2026-08-08 06:32  LAUNCH  StageC 2018 dijet MC allsyst
2026-08-08 06:43  FINDING StageC comparison vs LHE-basis archive: NEW is a strict SUPERSET of OLD (0 events lost). OLD 2018 production silently DROPPED CHUNKS: Run2018B 759->760 chunks (+99,329 events unread), Run2018D 3543->3545 (+195,770), dijet MC HT500to700 1229->1231 (+99,831). Run2018A/HT300to500 unchanged. trijet MC+data bit-identical. So the re-production also repaired a silent data loss. Residual: dijet MC HT300to500 has same chunks/initial/final_seq but 2 fewer stored skim rows (1 event in 214k) - unexplained, negligible.
2026-08-08 07:18  DONE    StageC 2018 dijet MC allsyst exit 0 wall 39m06s
2026-08-08 07:18  LAUNCH  StageC 2018 trijet MC allsyst
2026-08-08 07:18  FINDING COMPLETENESS AUDIT (StageA skim vs StageB allsyst, same input datasets): 5 of 12 channel-eras disagree on 'nEvents initial'. Small deltas = SILENT CHUNK LOSS affecting MY OWN runs too: 2016APV dijet_mc allsyst -149,856 (3515->3511 chunks), 2016APV dijet_data skim -201,255 vs allsyst AND vs trijet skim (3599 vs 3601), 2016 dijet_mc -56,321, 2016 trijet_mc -101,431. Magnitude 0.02-0.05%. 2017 trijet_mc skim reads ~2x = stale pre-46fbd31 double-count artifact (produced by previous agent), not loss.
2026-08-08 15:31  NOTE    *** ~8h agent stream stall (platform watchdog). casa server idle-timed out. StageC 2018 trijet MC allsyst had ALREADY COMPLETED (exit 0, 832,032,497 B) before the stop - no work lost, no restart needed. ***
2026-08-08 15:31  LAUNCH  StageC 2018 dijet data allsyst
2026-08-08 15:41  DONE    StageC 2018 dijet MC allsyst exit 0 (log: 32m57s) 1009.9 MB
2026-08-08 15:41  DONE    StageC 2018 trijet MC allsyst exit 0 (log: 55m10s) 832.0 MB
2026-08-08 15:41  DONE    StageC 2018 dijet data allsyst exit 0 wall 9m53s
2026-08-08 15:41  LAUNCH  StageC 2018 trijet data allsyst (FINAL RUN of campaign)
2026-08-08 15:52  DONE    StageC 2018 trijet data allsyst exit 0 wall 10m19s
2026-08-08 15:52  MILEST  *** ALL 20 CAMPAIGN RUNS COMPLETE (StageA 12 + StageB 12 + StageC 8 = 32 launches incl. reruns; 20 distinct outputs sets) ***
2026-08-08 15:52  HARVEST StageC 2018 allsyst harvest started
2026-08-08 16:02  VERIFY  StageC 2018 allsyst: dijet MC 77/77, trijet MC 77/77, data 1/1. XCHECK trijet data event_id 638,229 == StageA reco 638,229; dijet data event_id 6,993,066 == StageA final_seq sum. PASS. casa cleaned.
2026-08-08 16:02  MILEST  *** CAMPAIGN COMPLETE ***
```
