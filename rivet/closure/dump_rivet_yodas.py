"""Sum the RAW 2D histograms across the per-MiniAOD-file YODA outputs.

Run in the CMSSW environment (which provides the yoda python module):

    python3 dump_rivet_yodas.py 'yodas/*.yoda' rivet_gen.npz

Uses the /RAW/ copies, which hold the pre-finalize sumW, so the per-file outputs
can simply be added; the finalized copies are already normalized per file and
must not be summed. Only the nominal weight variation is kept -- rivetAnalyzer
writes one set of histograms per generator weight, tagged with a "[...]" suffix.
"""

import glob
import sys

import numpy as np
import yoda

ANA = "CMS_2026_PAS_SMP_25_010"


PTEDGES = np.array([200., 290., 400., 13000.])
RHOEDGES = np.array([-10., -4.5, -4., -3.5, -3., -2.5, -2., -1.5, -1., -0.5, 0.])


def cells(ao):
    """Return sumW[npt, nrho] for the routine's 2D histogram.

    The scatter carries bin *densities*, so each cell is multiplied back by its
    area. Bins are located against the known edges rather than reconstructed
    from the point positions.
    """
    out = np.zeros((len(PTEDGES) - 1, len(RHOEDGES) - 1))
    for p in ao.mkScatter().points():
        # YODA 2 Point3D: bin extents via xMin/xMax/yMin/yMax, z is the density
        dx, dy = p.xMax() - p.xMin(), p.yMax() - p.yMin()
        i = int(np.searchsorted(PTEDGES, 0.5 * (p.xMin() + p.xMax())) - 1)
        j = int(np.searchsorted(RHOEDGES, 0.5 * (p.yMin() + p.yMax())) - 1)
        if 0 <= i < out.shape[0] and 0 <= j < out.shape[1]:
            out[i, j] += p.z() * dx * dy
    return PTEDGES, RHOEDGES, out


def main(pattern, outfile):
    files = sorted(glob.glob(pattern))
    print(f"{len(files)} yoda files")
    acc, edges = {}, {}
    for f in files:
        aos = yoda.read(f)
        for groom in ("ungroomed", "groomed"):
            key = f"/RAW/{ANA}/zjets_{groom}"
            if key not in aos:
                print(f"  {f}: missing {key}")
                continue
            xe, ye, v = cells(aos[key])
            acc[groom] = v if groom not in acc else acc[groom] + v
            edges[groom] = (xe, ye)
        print(f"  {f}: ok")

    saved = {}
    for groom, v in acc.items():
        saved[f"zjets_{groom}__values"] = v
        saved[f"zjets_{groom}__ptedges"] = edges[groom][0]
        saved[f"zjets_{groom}__rhoedges"] = edges[groom][1]
        print(f"zjets_{groom}: shape={v.shape} total={v.sum():.1f}")
    np.savez(outfile, **saved)
    print("wrote", outfile)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
