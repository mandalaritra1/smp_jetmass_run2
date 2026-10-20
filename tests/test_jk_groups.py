"""One-pass event-group jackknife (jk_groups) on the staged local NanoAOD files.

Checks, for dijet and trijet, MC and data (pair_split, mode minimal_rho, nominal only):
1. closure: each "<name>_jkgroups" histogram summed over its groups equals the
   nominal "<name>" histogram (in-range bins), values and variances;
2. no side effects: every other histogram and the cutflow are identical to a run
   with jk_groups=0;
3. the group hash is deterministic, roughly uniform, and depends on the event
   identity only;
4. jk_groups always runs nominal only, whatever systematics are requested, and
   writes its own output file.

Run standalone from the repo root (no pytest in the GluonJetMass venv):
    PYTHONPATH=. ~/Projects/GluonJetMass/.venv/bin/python tests/test_jk_groups.py [--maxchunks 2]
"""
import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GJM = Path("~/Projects/GluonJetMass").expanduser()
N_GROUPS = 100


def build_fileset(is_data):
    kind = "data" if is_data else "mc"
    links = sorted(glob.glob(str(GJM / "test_files/store" / kind / "**/*.root"), recursive=True))
    if not links:
        sys.exit(f"No staged files under {GJM}/test_files/store/{kind}/")
    fs = defaultdict(list)
    for f in links:
        parts = Path(f).parts
        i = parts.index("store")
        fs[f"/{parts[i+3]}/{parts[i+2]}-{parts[i+5]}/{parts[i+4]}"].append(f)
    return dict(fs)


def run(proc_name, is_data, jk_groups, chunksize, maxchunks):
    from coffea import processor
    from coffea.nanoevents import NanoAODSchema
    if proc_name == "dijet":
        from smp_jetmass_run2.dijet_processor import DijetProcessor as P
    else:
        from smp_jetmass_run2.trijet_processor import TrijetProcessor as P
    proc = P(do_gen=not is_data, mode="minimal_rho", jet_systematics=["nominal"],
             systematics=["nominal"], jk_groups=jk_groups)
    runner = processor.Runner(executor=processor.IterativeExecutor(workers=1, status=False),
                              schema=NanoAODSchema, chunksize=chunksize,
                              maxchunks=(maxchunks or None), skipbadfiles=False)
    return runner(build_fileset(is_data), proc, treename="Events")


def check_hash():
    from smp_jetmass_run2.hadronic_base import HadronicProcessorBase as B
    rng = np.random.default_rng(1)
    run = np.full(200000, 1, dtype=np.uint32)
    lumi = rng.integers(1, 5000, 200000).astype(np.uint32)
    event = rng.integers(1, 2**40, 200000).astype(np.uint64)
    g = B.jk_group_of(run, lumi, event, N_GROUPS)
    assert g.min() >= 0 and g.max() < N_GROUPS
    assert np.array_equal(g, B.jk_group_of(run, lumi, event, N_GROUPS)), "not deterministic"
    counts = np.bincount(g, minlength=N_GROUPS)
    expected = len(g) / N_GROUPS
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # chi2 with 99 dof: mean 99, sd 14; a broken hash gives thousands
    assert chi2 < 99 + 6 * 14, f"group occupancy not uniform: chi2={chi2:.0f}"
    # consecutive event numbers (the usual MC pattern) must not collapse onto few groups
    gc = B.jk_group_of(np.ones(10000, np.uint32), np.ones(10000, np.uint32),
                       np.arange(1, 10001, dtype=np.uint64), N_GROUPS)
    assert len(np.unique(gc)) == N_GROUPS
    print(f"hash: deterministic, uniform (chi2/ndf = {chi2:.0f}/99), all groups hit by consecutive events")


def check_nominal_only():
    from smp_jetmass_run2.dijet_processor import DijetProcessor
    from smp_jetmass_run2.trijet_processor import TrijetProcessor
    from smp_jetmass_run2.notebook_utils import make_output_filename
    for P in (DijetProcessor, TrijetProcessor):
        proc = P(do_gen=True, mode="minimal_rho", systematics=None, jet_systematics=None,
                 jk_groups=N_GROUPS)
        assert proc.jet_systematics == ["nominal"] and proc.systematics == ["nominal"], P.__name__
    name = make_output_filename(False, "mg_pythia8", "2018", mode="minimal_rho", channel="dijet",
                                output_dir="o", jk_groups=N_GROUPS)
    assert name.endswith(f"minimal_rho_dijet_mg_pythia8_jkg{N_GROUPS}_2018.pkl"), name
    print("nominal only: all_syst request forced to nominal; output file", Path(name).name)


def check_closure(out, label):
    from smp_jetmass_run2.hadronic_base import HadronicProcessorBase as B
    names = [n for n in B.JK_GROUP_HISTS if n + B.JK_GROUP_SUFFIX in out]
    assert names, f"{label}: no _jkgroups histograms registered"
    worst = 0.0
    occupancy = {}
    for name in names:
        base, grp = out[name], out[name + B.JK_GROUP_SUFFIX]
        assert len(grp.axes["jk"]) == N_GROUPS
        for ds in grp.axes["dataset"]:
            b = base[{"dataset": ds, "systematic": "nominal"}]
            gsum = grp[{"dataset": ds}].project(*[a.name for a in b.axes])
            assert [a.name for a in gsum.axes] == [a.name for a in b.axes]
            for what in ("values", "variances"):
                x, y = getattr(b, what)(), getattr(gsum, what)()
                scale = max(np.abs(x).max(), 1e-300)
                diff = float(np.abs(x - y).max() / scale)
                worst = max(worst, diff)
                assert diff < 1e-10, f"{label} {name} {ds} {what}: max rel diff {diff:.2e}"
            occ = int((grp[{"dataset": ds}].project("jk").values() > 0).sum())
            occupancy[name] = occ
            # a non-empty spectrum must spread over more than one group
            assert occ > 1 or b.values().sum() == 0, f"{label} {name}: all entries in one group"
    print(f"{label}: closure OK for {len(names)} histograms (worst rel diff {worst:.1e}); "
          f"groups filled: {occupancy}")


def check_no_side_effects(out_on, out_off, label):
    import hist
    from smp_jetmass_run2.hadronic_base import HadronicProcessorBase as B
    keys_off = set(out_off)
    keys_on = {k for k in out_on if not k.endswith(B.JK_GROUP_SUFFIX)}
    assert keys_on == keys_off, f"{label}: key sets differ: {keys_on ^ keys_off}"
    for k in sorted(keys_off):
        a, b = out_off[k], out_on[k]
        if isinstance(a, hist.Hist):
            assert np.array_equal(a.view(flow=True), b.view(flow=True)), f"{label}: {k} changed"
        elif k == "cutflow":
            assert a == b, f"{label}: cutflow changed"
    print(f"{label}: all {len(keys_off)} non-jk outputs identical to the jk_groups=0 run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxchunks", type=int, default=2)
    ap.add_argument("--chunksize", type=int, default=20000)
    a = ap.parse_args()
    check_hash()
    check_nominal_only()
    for proc in ("dijet", "trijet"):
        for is_data in (False, True):
            label = f"{proc} {'data' if is_data else 'MC'}"
            on = run(proc, is_data, N_GROUPS, a.chunksize, a.maxchunks)
            off = run(proc, is_data, 0, a.chunksize, a.maxchunks)
            check_closure(on, label)
            check_no_side_effects(on, off, label)
    print("PASS")


if __name__ == "__main__":
    main()
