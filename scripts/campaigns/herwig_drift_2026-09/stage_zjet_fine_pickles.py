"""Stage the fine-rho (minimal_rho_fine, 4x rho bins) 2018 Z+jet pickles into unfold_inputs.

    python stage_zjet_fine_pickles.py <dir with minimal_rho_fine_{pythia,herwig,herwig_matchbox}_2018.pkl> <producer_commit_string>

Copies the three pickles to zjet/rho/fine/2018/ and writes zjet/rho/fine/fine_manifest.json
with bytes, sha256, event counts, sumw, effective statistics per gen pT bin (groomed and
ungroomed) and the axis check against the staged production pythia_2018.pkl (pT axes
identical; rho axes 4x refined with the production edges a subset).
"""
import datetime, hashlib, json, pickle, shutil, sys
from pathlib import Path

import hist, numpy as np

src_dir, commit = Path(sys.argv[1]), sys.argv[2]
root = Path.home() / "cernbox/unfold_inputs/zjet/rho"
ref = pickle.load(open(root / "original/pythia_2018.pkl", "rb"))
dst_dir = root / "fine/2018"
dst_dir.mkdir(parents=True, exist_ok=True)
PT_LABELS = ["185-200", "200-290", "290-400", ">400"]
DAS = {
    "pythia": "DYJetsToLL_M-50_HT-*_TuneCP5_13TeV-madgraphMLM-pythia8 (UL18 NanoAODv9, seven HT bins)",
    "herwig": "/DYJetsToLL_M-50_TuneCH3_13TeV-madgraphMLM-herwig7/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM",
    "herwig_matchbox": "/DYToLL_NLO_5FS_TuneCH3_13TeV_matchbox_herwig7/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1_ext1-v2/NANOAODSIM",
}


def neff_per_pt(d, key):
    h = d[key][{"dataset": sum, "systematic": "nominal"}]
    v, var = np.asarray(h.values()), np.asarray(h.variances())
    return {PT_LABELS[i]: round(float(v[i].sum() ** 2 / max(var[i].sum(), 1e-300)), 1) for i in range(v.shape[0])}


manifest = {
    "campaign": "zjet minimal_rho_fine (4x rho bins: 48 gen / 96 reco) 2018, nominal only, Pythia8 MLM, Herwig7 MLM and "
                "NLO Matchbox Herwig7 produced together on LPC condor with the same code (2026-09-18)",
    "purpose": "finer-rho version of the Herwig7 groomed-mass pT-drift comparison (research-notes "
               "herwig_groomed_jet_mass_pt_drift.md section 6) so the three samples are directly comparable on one lattice",
    "producer_repo_commit": commit,
    "configs": "configs/zjet_{pythia,herwig,herwig_matchbox}_2018_minimal_rho_fine_lpc.json",
    "mode": "minimal_rho_fine", "systematic_profile": "no_syst", "systematic_axis": ["nominal"], "rho_refine": 4,
    "weights": "genWeight summed on the fly into sumw; xs*lumi*1000/sumw in postprocess (Matchbox: inclusive NLO DY xs, shape-only)",
    "caveat": "single era (2018); the June 2026 zjet/model/finebins pickles carry the old 0-based pT axis and are NOT comparable",
    "files": {},
    "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
for ds in ("pythia", "herwig", "herwig_matchbox"):
    src = src_dir / f"minimal_rho_fine_{ds}_2018.pkl"
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    d = pickle.load(open(dst, "rb"))
    report = {}
    for k, v in d.items():
        if not isinstance(v, hist.Hist) or k not in ref:
            continue
        for a, b in zip(v.axes, ref[k].axes):
            if a.name in ("dataset", "systematic"):
                continue
            if len(a.edges) == len(b.edges) and np.allclose(a.edges, b.edges):
                report[f"{k}:{a.name}"] = "identical"
            elif all(np.isclose(a.edges, e).any() for e in b.edges):
                report[f"{k}:{a.name}"] = f"refined {len(a.edges) - 1} bins, production edges all present"
            else:
                report[f"{k}:{a.name}"] = "MISMATCH"
    cf = d["cutflow"]
    sumw = {k: float(v) for k, v in d["sumw"].items()}
    manifest["files"][ds] = {
        "file": f"2018/{src.name}", "bytes": dst.stat().st_size,
        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "dataset": DAS[ds],
        "events_processed": int(sum(v for k, v in cf.items() if k.endswith("_all"))),
        "events_allsel_gen": int(sum(v for k, v in cf.items() if k.endswith("_allsel_gen"))),
        "sumw_genWeight": sumw,
        "histograms": sum(isinstance(v, hist.Hist) for v in d.values()),
        "axes_vs_production": sorted(set(report.values())),
        "n_eff_gen_groomed_per_pt": neff_per_pt(d, "ptjet_rhojet_g_gen"),
        "n_eff_gen_ungroomed_per_pt": neff_per_pt(d, "ptjet_rhojet_u_gen"),
    }
    f = manifest["files"][ds]
    print(ds, "->", dst, f["events_processed"], "events,", f["events_allsel_gen"], "gen selected, axes",
          f["axes_vs_production"], "n_eff", f["n_eff_gen_groomed_per_pt"])
json.dump(manifest, open(root / "fine/fine_manifest.json", "w"), indent=2)
print("wrote", root / "fine/fine_manifest.json")
