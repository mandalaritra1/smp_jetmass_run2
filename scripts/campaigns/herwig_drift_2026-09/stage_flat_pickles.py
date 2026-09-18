"""Stage the flat-pT dijet pickles from the lxplus outputs into unfold_inputs with a manifest.

    python stage_flat_pickles.py <lpc_outputs_dir_local_copy> <producer_commit_string>

Copies
  minimal_rho_dijet_herwig_flat_2018.pkl -> dijet/rho/original/2018/dijet_flat_herwig/minimal_rho_dijet_flat_herwig_2018.pkl
  minimal_rho_dijet_pythia_flat_2018.pkl -> dijet/rho/original/2018/dijet_flat_pythia/minimal_rho_dijet_flat_pythia8_2018.pkl
and writes dijet/rho/original/flat_manifest.json with bytes, sha256, event counts, sumw,
effective statistics per gen pT bin (groomed and ungroomed) and the axis check against the
2018 MLM production pickle.
"""
import datetime, hashlib, json, pickle, shutil, sys
from pathlib import Path

import hist, numpy as np

src_dir, commit = Path(sys.argv[1]), sys.argv[2]
root = Path.home() / "cernbox/unfold_inputs/dijet/rho/original"
ref = pickle.load(open(root / "2018/dijet_mc/minimal_rho_dijet_mg_pythia8_2018.pkl", "rb"))
plan = {
    "flat_herwig": ("minimal_rho_dijet_herwig_flat_2018.pkl", "2018/dijet_flat_herwig/minimal_rho_dijet_flat_herwig_2018.pkl",
                    "configs/dijet_herwig_flat_2018_minimal_rho_lxplus.json",
                    "/QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM"),
    "flat_pythia": ("minimal_rho_dijet_pythia_flat_2018.pkl", "2018/dijet_flat_pythia/minimal_rho_dijet_flat_pythia8_2018.pkl",
                    "configs/dijet_pythia_flat_2018_minimal_rho_lxplus.json",
                    "/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM"),
}
PT_LABELS = ["185-200", "200-290", "290-400", "400-480", "480-570", "570-680", "680-760", "760-820", ">820"]


def neff_per_pt(d, key):
    h = d[key][{"dataset": sum, "systematic": "nominal"}]
    v, var = np.asarray(h.values()), np.asarray(h.variances())
    return {PT_LABELS[i]: round(float(v[i].sum() ** 2 / max(var[i].sum(), 1e-300)), 1) for i in range(v.shape[0])}


manifest = {
    "campaign": "dijet flat-pT QCD (standalone Herwig7 CH3 / Pythia8 CP5, no MadGraph, no MLM) minimal_rho, 2018, nominal only",
    "produced": "2026-09-17 (CERN lxplus HTCondor via dask-lxplus, LCG_110 coffea 2026.5.0, deployment lxplus:~/smp_jetmass_run2; the LPC pool was saturated and coffea-casa down)",
    "producer_repo_commit": commit,
    "purpose": "test whether the pT drift of the soft-drop groomed mass in MLM+Herwig7 is made by the merging: "
               "Flat-Herwig7/Flat-Pythia8 per pT bin against MLM-Herwig7/MLM-Pythia8",
    "mode": "minimal_rho", "systematic_profile": "no_syst", "systematic_axis": ["nominal"],
    "selection": "pair_split (processor default): pT floors, |y| < 2.4, dphi12 > pi - 200/ptavg",
    "binning": "2026-08-27 aligned groomed rho lattice, 185 GeV pT sink (util_binning)",
    "weights": "genWeight = pT-hat weight of the flat sample, summed on the fly into sumw; xs*lumi*1000/sumw in postprocess "
               "(flat Pythia8 xs 1.37e9 pb is approximate, shape-only use)",
    "flat_pthat_outlier_veto": "events whose leading gen or reco AK8 jet exceeds 3 x pT-hat (Generator_binvar) are vetoed "
                               "pre-selection (HadronicProcessorBase.FLAT_PTHAT_RATIO_MAX, commit e9ab84b); without it the Herwig "
                               "flat sample has n_eff of 2-27102 in eight of nine gen pT bins (scan of all 700 files, 2026-09-17); "
                               "the veto drops < 0.01% of the events per bin; Pythia8 flat has no jet above 3 x pT-hat",
    "caveat": "single era (2018); flat pT-hat spectrum inside a bin differs from the HT-binned MLM samples; effective statistics below",
    "files": {},
    "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
for key, (src_name, dst_rel, cfg, das) in plan.items():
    src, dst = src_dir / src_name, root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    d = pickle.load(open(dst, "rb"))
    bad = []
    for k, v in d.items():
        if not isinstance(v, hist.Hist) or k not in ref:
            continue
        for a, b in zip(v.axes, ref[k].axes):
            if a.name != b.name or (a.name not in ("dataset", "systematic", "partonFlav") and not np.allclose(a.edges, b.edges)):
                bad.append((k, a.name))
    nhist = sum(isinstance(v, hist.Hist) for v in d.values())
    cf = d["cutflow"]
    datastr = next(iter(cf))
    n_initial = int(cf[datastr].get("nEvents initial (all datasets)", 0))
    manifest["files"][key] = {
        "file": dst_rel, "bytes": dst.stat().st_size,
        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "config": cfg, "dataset": das,
        "events_processed": n_initial, "sumw_genWeight": float(d["sumw"][das]),
        "events_vetoed_flat_pthat_outlier": int(cf[datastr].get("nEvents flat pthat outlier veto", 0)),
        "histograms": nhist,
        "axes_identical_to_mlm_2018": not bad, "axis_mismatches": bad,
        "n_eff_gen_groomed_per_pt": neff_per_pt(d, "ptjet_rhojet_g_gen"),
        "n_eff_gen_ungroomed_per_pt": neff_per_pt(d, "ptjet_rhojet_u_gen"),
    }
    print(key, "->", dst, n_initial, "events, hists", nhist, "axes ok" if not bad else bad)
json.dump(manifest, open(root / "flat_manifest.json", "w"), indent=2)
print("wrote", root / "flat_manifest.json")
