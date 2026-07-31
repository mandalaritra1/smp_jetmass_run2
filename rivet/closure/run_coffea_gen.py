"""Coffea half of the gen-level closure: run the zjet processor's gen path over
the NanoAOD child of the MiniAOD files the Rivet routine was run on, and dump the
gen-truth rho histograms.

Run under an LCG view with the package on PYTHONPATH:

    source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
    PYTHONPATH=~/closure_repo python3 run_coffea_gen.py <nanoaod-file> <out.npz>

The dataset key is the real sample name so the processor takes its normal
HT-binned code path; only shapes and relative yields are used downstream, so the
absolute cross-section normalization is irrelevant to the comparison.
"""

import sys
import numpy as np

from smp_jetmass_run2.notebook_utils import run_once

# postprocess() parses the dataset key as <sample>_<IOV>_<HT-bin> and requires
# "pythia" in the name to pick the madgraphMLM cross-section table.
DATASET = "pythia_UL18NanoAODv9_HT-400to600"


def main(nanofile, outfile, test=False):
    fileset = {DATASET: [nanofile]}
    out = run_once(
        fileset,
        client=None,
        test=test,
        data=False,
        mode="minimal_rho",
        channel="zjet",
        systematic_profile="no_syst",
        chunksize=50_000,
        chunksize_test=50_000,
        executor_mode="futures",
    )

    hists = out["hists"] if isinstance(out, dict) and "hists" in out else out
    saved = {}
    for name in ("ptjet_rhojet_u_gen", "ptjet_rhojet_g_gen"):
        if name not in hists:
            print(f"MISSING histogram {name}; available: {sorted(hists)[:40]}")
            continue
        h = hists[name]
        # collapse dataset and systematic axes, keep (ptgen, rho)
        hh = h[{"dataset": sum}] if "dataset" in [a.name for a in h.axes] else h
        if "syst" in [a.name for a in hh.axes]:
            hh = hh[{"syst": sum}]
        if "systematic" in [a.name for a in hh.axes]:
            hh = hh[{"systematic": sum}]
        saved[name + "__values"] = hh.values()
        saved[name + "__variances"] = (
            hh.variances() if hh.variances() is not None else np.zeros_like(hh.values())
        )
        for ax in hh.axes:
            saved[f"{name}__edges__{ax.name}"] = np.asarray(ax.edges)
        print(f"{name}: shape={hh.values().shape} sum={hh.values().sum():.4f}")

    np.savez(outfile, **saved)
    print("wrote", outfile)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], test=(len(sys.argv) > 3 and sys.argv[3] == "test"))
