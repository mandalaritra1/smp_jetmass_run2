"""Stitch the hadronic model-production ntuples and report, per channel and
per pT bin, the normalized groomed-rho shape of each variation relative to
cp5 -- i.e. the raw material for the HAD-leg reweight.

  python3 harvest_shapes.py [EOSBASE] [--channel dijet|trijet]

Per file the absolute normalization is xsec_pb / sumw (LO gridpack xsec after
matching); HT bins are summed with that weight.  Shapes are then normalized
to unit area WITHIN each pT bin, so the reported ratio is a pure shape
effect.  All configs in a job shower the SAME LHE, so these ratios are
statistically correlated and the quoted per-bin errors (independent-Poisson)
are an OVERESTIMATE of the ratio uncertainty.
"""
import glob
import os
import sys

import numpy as np

CHANNEL = "dijet"
DUMP = None
argv = sys.argv[1:]
positional = []
i = 0
while i < len(argv):
    if argv[i] == "--channel":
        CHANNEL = argv[i + 1]; i += 2
    elif argv[i] == "--dump":
        DUMP = argv[i + 1]; i += 2
    else:
        positional.append(argv[i]); i += 1
EOSBASE = positional[0] if positional else "/eos/user/a/amandal/hadronic_model_prod"

#### vincia = the parton-shower leg (matching-free HardQCD path);
#### the rest are the hadronisation leg.
CONFIGS = ["cp5", "vincia", "cr1", "cr2", "fraghard", "fragsoft"]
CONFIGS = [c for c in CONFIGS
           if glob.glob(os.path.join(EOSBASE, c, "*", "*_dijet.txt"))
           or glob.glob(os.path.join(EOSBASE, c, "*", "*_trijet.txt"))]
# Generation bins are whatever subdirectories exist: QCD_HT* for the gridpack
# path, pthat* for the HardQCD path. Both are stitched the same way (each file
# carries its own xsec/sumw), so one harvest serves both productions.
BINS = sorted(os.path.basename(d.rstrip("/"))
              for d in glob.glob(os.path.join(EOSBASE, "cp5", "*"))
              if os.path.isdir(d))
RHO = np.array([-10, -5, -4, -3.4, -2.85, -2.25, -1.8, -1.5,
                -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.])
PT = [(200, 290), (290, 400), (400, 13000)]


def load(cfg):
    """Histogram incrementally, one file at a time.

    Returns {(ptlo): (h, sumw2, nraw)} keyed by pT-bin low edge.  Accumulating
    per file instead of concatenating keeps this linear in memory and avoids
    holding tens of millions of rows: the HardQCD wave produces ~4M jets per
    config, where a concatenate-then-histogram pass is both slow (np.loadtxt)
    and memory-hostile.
    """
    acc = {lo: [np.zeros(len(RHO) - 1) for _ in range(3)] for lo, _ in PT}
    ntot = 0
    for b in BINS:
        pat = os.path.join(EOSBASE, cfg, b, "ntuple_*_%s.txt" % CHANNEL)
        for f in glob.glob(pat):
            xs = sumw = None
            with open(f) as fh:
                for line in fh:
                    if line.startswith("# xsec_pb"):
                        xs = float(line.split()[2])
                    elif line.startswith("# sumw"):
                        sumw = float(line.split()[2])
            if not xs or not sumw:
                continue
            try:
                a = np.loadtxt(f, ndmin=2, comments="#")
            except ValueError:
                continue
            if a.size == 0:
                continue
            pt, rho, w = a[:, 0], a[:, 4], a[:, 5] * xs / sumw
            ntot += len(pt)
            for lo, hi in PT:
                m = (pt >= lo) & (pt < hi) & (rho > -99)
                if not m.any():
                    continue
                h, _ = np.histogram(rho[m], bins=RHO, weights=w[m])
                h2, _ = np.histogram(rho[m], bins=RHO, weights=w[m] ** 2)
                n, _ = np.histogram(rho[m], bins=RHO)
                acc[lo][0] += h; acc[lo][1] += h2; acc[lo][2] += n
    return acc, ntot


loaded = {c: load(c) for c in CONFIGS}
data = {c: loaded[c][0] for c in CONFIGS}
print("channel: %s   (jets loaded: %s)"
      % (CHANNEL, ", ".join("%s=%d" % (c, loaded[c][1]) for c in CONFIGS)))

if DUMP:
    out = {"rho_edges": RHO, "pt_edges": np.array([p[0] for p in PT] + [PT[-1][1]]),
           "configs": np.array(CONFIGS)}
    for lo, hi in PT:
        for c in CONFIGS:
            h, h2, n = data[c][lo]
            key = "%s_pt%d" % (c, lo)
            out[key + "_h"] = h
            out[key + "_e"] = np.sqrt(h2)
            out[key + "_n"] = n
    np.savez(DUMP, **out)
    print("wrote %s" % DUMP)

for lo, hi in PT:
    print("\n=== pT [%d, %s) ===" % (lo, "inf" if hi > 9999 else hi))
    ref = None
    rows = {}
    for c in CONFIGS:
        h, h2, n = data[c][lo]
        tot = h.sum()
        rows[c] = (h / tot, np.sqrt(h2) / tot, n)
        if c == "cp5":
            ref = rows[c]
    print("%-16s" % "rho_g bin"
          + "".join("%16s" % c for c in CONFIGS[1:])
          + "%10s" % "N(cp5)")
    nsig = ntot = 0
    devs_all, errs_all = [], []
    for i in range(len(RHO) - 1):
        if ref[0][i] <= 0 or ref[2][i] < 50:
            continue
        line = "[%6.2f,%6.2f]" % (RHO[i], RHO[i + 1])
        for c in CONFIGS[1:]:
            r = rows[c][0][i] / ref[0][i]
            # independent-Poisson error on the ratio (ignores the shared-LHE
            # correlation, so this is an UPPER bound on the true uncertainty)
            e = r * np.hypot(rows[c][1][i] / rows[c][0][i], ref[1][i] / ref[0][i])
            line += "%16s" % ("%.3f+-%.3f" % (r, e))
            devs_all.append(abs(r - 1)); errs_all.append(e)
            ntot += 1
            if abs(r - 1) > 2 * e:
                nsig += 1
        line += "%10d" % ref[2][i]
        print(line)
    devs_all = np.array(devs_all); errs_all = np.array(errs_all)
    print("  typical |ratio-1| = %.1f%%   typical stat error = %.1f%%"
          % (100 * np.median(devs_all), 100 * np.median(errs_all)))
    print("  bins deviating >2 sigma: %d / %d  (%.0f%% expected by chance if pure noise: ~5%%)"
          % (nsig, ntot, 100.0 * nsig / max(ntot, 1)))
