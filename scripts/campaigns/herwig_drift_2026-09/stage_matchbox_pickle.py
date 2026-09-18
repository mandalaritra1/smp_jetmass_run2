"""Stage the zjet Matchbox pickle from the LPC outputs into unfold_inputs with a manifest.

    python stage_matchbox_pickle.py <local copy of minimal_rho_herwig_matchbox_r2_2018.pkl> <producer_commit_string>

Copies it to zjet/rho/original/2018/herwig_matchbox/minimal_rho_zjet_herwig_matchbox_2018.pkl and
writes zjet/rho/original/herwig_matchbox_manifest.json with bytes, sha256, event count, sumw,
effective statistics per gen pT bin (groomed and ungroomed) and the axis check against
pythia_2018.pkl (pT axis identical; rho axis 2x refined, the production edges a subset).
"""
import datetime, hashlib, json, pickle, shutil, sys
from pathlib import Path

import hist, numpy as np

src, commit = Path(sys.argv[1]), sys.argv[2]
root = Path.home() / "cernbox/unfold_inputs/zjet/rho/original"
ref = pickle.load(open(root / "pythia_2018.pkl", "rb"))
dst_rel = "2018/herwig_matchbox/minimal_rho_zjet_herwig_matchbox_2018.pkl"
dst = root / dst_rel
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
d = pickle.load(open(dst, "rb"))
PT_LABELS = ["185-200", "200-290", "290-400", ">400"]
DAS = ("/DYToLL_NLO_5FS_TuneCH3_13TeV_matchbox_herwig7/"
       "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/NANOAODSIM")


def neff_per_pt(key):
    h = d[key][{"dataset": sum, "systematic": "nominal"}]
    v, var = np.asarray(h.values()), np.asarray(h.variances())
    return {PT_LABELS[i]: round(float(v[i].sum() ** 2 / max(var[i].sum(), 1e-300)), 1) for i in range(v.shape[0])}


axis_report = {}
for k, v in d.items():
    if not isinstance(v, hist.Hist) or k not in ref:
        continue
    for a, b in zip(v.axes, ref[k].axes):
        if a.name in ("dataset", "systematic"):
            continue
        if np.allclose(a.edges, b.edges) if len(a.edges) == len(b.edges) else False:
            axis_report[f"{k}:{a.name}"] = "identical"
        elif all(np.isclose(a.edges, e).any() for e in b.edges):
            axis_report[f"{k}:{a.name}"] = f"refined {len(a.edges) - 1} bins, production {len(b.edges) - 1} edges all present (rebin 2:1)"
        else:
            axis_report[f"{k}:{a.name}"] = "MISMATCH"
cf = d["cutflow"]
ds = next(iter(d["sumw"]))
manifest = {
    "campaign": "zjet NLO Matchbox Herwig7 CH3 DY (MG5 ME inside Herwig's Matchbox, no external LHE, no MLM) minimal_rho, 2018, nominal only",
    "produced": "2026-09-18 (FNAL LPC HTCondor via dask-lpc, coffea 2025.9.0 container, deployment cmslpc303:~/nobackup/smp_jetmass_run2_hadronic_herwig)",
    "producer_repo_commit": commit,
    "purpose": "the sample CMS GEN (Kostas, 2026-09-18) proposes as the MG+Herwig7 replacement; under the merging hypothesis "
               "(research-notes herwig_groomed_jet_mass_pt_drift.md) its groomed mass must not drift in pT against MLM+Pythia8",
    "config": "configs/zjet_herwig_matchbox_2018_minimal_rho_lpc.json",
    "dataset": DAS,
    "mode": "minimal_rho", "systematic_profile": "no_syst", "systematic_axis": ["nominal"], "rho_refine": 2,
    "weights": "genWeight = signed NLO weight (16% negative), summed on the fly into sumw; xs*lumi*1000/sumw in postprocess "
               "with the inclusive NLO DY xs 6404 pb (the sample's mass window not verified: shape-only use)",
    "branches": "no LHE branches (PDF/Q2 helpers return unity), nPSWeight = 1 dummy, GenJetAK8.partonFlavour populated",
    "caveat": "single era (2018), UL18 is the only era of this sample; rho axis 2x refined relative to the staged production pickles, rebin 2:1 to compare",
    "file": dst_rel, "bytes": dst.stat().st_size,
    "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
    "events_processed": int(cf.get(f"{ds}_all", 0)),
    "events_allsel_gen": int(cf.get(f"{ds}_allsel_gen", 0)),
    "events_allsel_reco": int(cf.get(f"{ds}_allsel_reco", 0)),
    "sumw_genWeight": float(d["sumw"][ds]),
    "histograms": sum(isinstance(v, hist.Hist) for v in d.values()),
    "axes_vs_pythia_2018": axis_report,
    "n_eff_gen_groomed_per_pt": neff_per_pt("ptjet_rhojet_g_gen"),
    "n_eff_gen_ungroomed_per_pt": neff_per_pt("ptjet_rhojet_u_gen"),
    "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
json.dump(manifest, open(root / "herwig_matchbox_manifest.json", "w"), indent=2)
print(json.dumps({k: manifest[k] for k in ("events_processed", "events_allsel_gen", "sumw_genWeight",
                                           "histograms", "n_eff_gen_groomed_per_pt")}, indent=1))
print("axes:", set(axis_report.values()))
print("wrote", dst, "and", root / "herwig_matchbox_manifest.json")
