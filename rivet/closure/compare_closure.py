"""Compare the Rivet routine and the coffea gen path on the same FullSim events.

    python3 compare_closure.py rivet_gen.npz coffea_gen_full.npz

Both sides are rebinned onto the published log10(rho^2) axis. The two rho
binnings differ and neither is a superset of the other, so:

  ungroomed  coffea [-10,-6,-5,-4.5,-4,...,0]  -> merge the first three bins
             rivet  is already the published axis
  groomed    coffea [-10,...,-4.75,-4.5,-4.25,...,-1,0] -> merge to the same
             edges, except it has no -0.5 edge, so the last TWO published bins
             are merged on both sides

Absolute normalizations differ by construction (coffea applies xs*lumi/sumw,
Rivet fills raw generator weights), so the comparison is: per-pT-slice shape,
and the relative population of the three pT slices.
"""

import sys

import numpy as np

PUB_RHO = np.array([-10., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0.])
PUB_PT = np.array([200., 290., 400., 13000.])
SLICE = ["200-290", "290-400", ">400"]


def rebin(values, edges, target):
    """Sum `values` (1D, over `edges`) into `target` bins. Target edges must be a
    subset of source edges."""
    edges = np.round(np.asarray(edges, float), 6)
    target = np.round(np.asarray(target, float), 6)
    for e in target:
        if not np.any(np.isclose(edges, e)):
            raise ValueError(f"target edge {e} is not a source edge: {list(edges)}")
    out = np.zeros(len(target) - 1)
    for i in range(len(target) - 1):
        lo, hi = target[i], target[i + 1]
        mask = (edges[:-1] >= lo - 1e-9) & (edges[1:] <= hi + 1e-9)
        out[i] = values[mask].sum()
    return out


def load_coffea(npz, name):
    v = npz[f"{name}__values"]
    rho = npz[f"{name}__edges__mpt_gen"]
    pt = npz[f"{name}__edges__ptgen"]
    return v, pt, rho


def main(rivet_npz, coffea_npz):
    R = np.load(rivet_npz)
    C = np.load(coffea_npz)

    for groom, cname in (("ungroomed", "ptjet_rhojet_u_gen"),
                         ("groomed", "ptjet_rhojet_g_gen")):
        rv = R[f"zjets_{groom}__values"]
        rrho = R[f"zjets_{groom}__rhoedges"]
        cv, cpt, crho = load_coffea(C, cname)

        # published rho axis; groomed loses the -0.5 edge on the coffea side
        target = PUB_RHO if groom == "ungroomed" else np.delete(PUB_RHO, -2)

        print("=" * 78)
        print(f"{groom}   target rho edges: {list(target)}")
        tot_r, tot_c = [], []
        for i, lab in enumerate(SLICE):
            r = rebin(rv[i], rrho, target)
            # coffea ptgen axis is [185, 200, 290, 400, inf] -> skip the sink bin
            c = rebin(cv[i + 1], crho, target)
            tot_r.append(r.sum()); tot_c.append(c.sum())
            rs = r / r.sum() if r.sum() else r
            cs = c / c.sum() if c.sum() else c
            print(f"\n  pT {lab}:  rivet N={r.sum():.4g}   coffea N={c.sum():.4g}")
            print("    bin        rivet      coffea     ratio")
            for j in range(len(target) - 1):
                ratio = rs[j] / cs[j] if cs[j] else float("nan")
                print(f"    [{target[j]:5.2f},{target[j+1]:5.2f}]  "
                      f"{rs[j]:9.5f}  {cs[j]:9.5f}  {ratio:8.4f}")
        tot_r, tot_c = np.array(tot_r), np.array(tot_c)
        print(f"\n  pT-slice fractions  rivet: {np.round(tot_r/tot_r.sum(), 5)}")
        print(f"  pT-slice fractions coffea: {np.round(tot_c/tot_c.sum(), 5)}")
        print(f"  overall scale rivet/coffea: {tot_r.sum()/tot_c.sum():.6g}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
