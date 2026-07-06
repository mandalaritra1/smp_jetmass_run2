#!/usr/bin/env python3
"""Convert the Vincia gen reweight (derived in the unfold repo) into the
PtVarWeighter format consumed by the zjet_processor `reweight_pythia_rho` path.

Source : unfold/outputs/zjet/rho/vincia_reweight/vincia_reweight.npz
         (w_{g,u} shape (3 pT, nbin), rw_edges_{g,u}; pT bins 200-290/290-400/400-inf)
Output : smp_jetmass_run2/corrections/vincia_rho_reweight_{groomed,ungroomed}.npz
         keys: pt_edges (5,), rho_grids (4,) object, w_grids (4,) object
         + labelling metadata (source, reweight_edges_*).

The reweight is encoded as a TRUE STEP on the reweight axis: np.interp grid with
duplicated bin edges (x=repeat(edges,2)[1:-1], w=repeat(w_bin,2)) so PtVarWeighter
returns each rho bin's Vincia/Pythia ratio flat, no spline/smoothing. pT bin 0
([0,200], below the jet-pT>200 selection) is a flat w=1 placeholder.
"""
import os
from pathlib import Path
import numpy as np

SRC = Path(os.environ.get(
    "VINCIA_REWEIGHT_NPZ",
    Path.home() / "Projects/unfold/outputs/zjet/rho/vincia_reweight/vincia_reweight.npz"))
OUTDIR = Path(__file__).resolve().parents[1] / "smp_jetmass_run2" / "corrections"
PT_EDGES = np.array([0., 200., 290., 400., 13000.])   # 4 pT bins; bin 0 = placeholder


def step_grid(edges, w_bins):
    """(x, w) reproducing a per-bin step under np.interp (duplicated edges)."""
    return np.repeat(edges, 2)[1:-1], np.repeat(w_bins, 2)


def build(mode_key, edges_key, out_name):
    z = np.load(SRC)
    w = z[mode_key]                 # (3 pT, nbin)  pT bins 1,2,3
    edges = z[edges_key]
    rho_grids = np.empty(4, dtype=object)
    w_grids = np.empty(4, dtype=object)
    # pT bin 0 ([0,200]): flat unity placeholder
    rho_grids[0] = np.array([edges[0], edges[-1]], float)
    w_grids[0] = np.array([1.0, 1.0], float)
    for k in range(3):              # pT bins 1,2,3 -> analysis bins 200-290/290-400/400-inf
        x, wv = step_grid(edges, w[k])
        rho_grids[k + 1] = x.astype(float)
        w_grids[k + 1] = wv.astype(float)
    out = OUTDIR / out_name
    np.savez(
        out,
        pt_edges=PT_EDGES,
        rho_grids=rho_grids,
        w_grids=w_grids,
        source="vincia_over_pythia_cp5_madgraphMLM_UL18",
        reweight_edges=edges,
        note="per-bin STEP reweight w=Vincia/Pythia; pT bin0 [0,200] is unity placeholder",
    )
    print(f"wrote {out}  (pt bins {len(PT_EDGES)-1}, {len(edges)-1} rho bins)")


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    build("w_g", "rw_edges_g", "vincia_rho_reweight_groomed.npz")
    build("w_u", "rw_edges_u", "vincia_rho_reweight_ungroomed.npz")
