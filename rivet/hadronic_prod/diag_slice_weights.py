"""Where does the statistical power of each measured bin actually come from?

For each pT bin, break the CP5 sum down BY pThat SLICE, reporting each
slice's share of the yield (sum w) and of the variance (sum w^2), plus the
resulting effective N.  A slice that contributes little yield but most of the
variance is the one starving the measurement -- that is where events must be
added, and no amount of generating elsewhere will help.

    python3 diag_slice_weights.py <EOSBASE> --channel trijet
"""
import glob
import os
import sys

import numpy as np

CHANNEL = "dijet"
argv = sys.argv[1:]
pos = []
i = 0
while i < len(argv):
    if argv[i] == "--channel":
        CHANNEL = argv[i + 1]; i += 2
    else:
        pos.append(argv[i]); i += 1
EOSBASE = pos[0] if pos else "/eos/user/a/amandal/hadronic_model_prod_hardqcd"
PT = [(200, 290), (290, 400), (400, 13000)]

slices = sorted((os.path.basename(d.rstrip("/"))
                 for d in glob.glob(os.path.join(EOSBASE, "cp5", "*"))
                 if os.path.isdir(d)),
                key=lambda s: int(s.split("_")[0].replace("pthat", "")))

acc = {(lo, s): [0.0, 0.0, 0] for lo, _ in PT for s in slices}
for s in slices:
    for f in glob.glob(os.path.join(EOSBASE, "cp5", s, "ntuple_*_%s.txt" % CHANNEL)):
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
        pt, rho, wt = a[:, 0], a[:, 4], a[:, 5] * xs / sumw
        for lo, hi in PT:
            m = (pt >= lo) & (pt < hi) & (rho > -99)
            if not m.any():
                continue
            acc[(lo, s)][0] += wt[m].sum()
            acc[(lo, s)][1] += (wt[m] ** 2).sum()
            acc[(lo, s)][2] += int(m.sum())

print("channel: %s\n" % CHANNEL)
for lo, hi in PT:
    W = sum(acc[(lo, s)][0] for s in slices)
    W2 = sum(acc[(lo, s)][1] for s in slices)
    N = sum(acc[(lo, s)][2] for s in slices)
    print("=== pT [%d,%s)   raw N %d   effective N %.0f" % (lo, hi, N, (W ** 2 / W2) if W2 else 0))
    print("    %-16s %10s %10s %10s   %s" % ("slice", "yield%", "variance%", "rawN", "per-event weight"))
    for s in slices:
        w, w2, n = acc[(lo, s)]
        if n == 0:
            continue
        print("    %-16s %9.1f%% %9.1f%% %10d   %.3g"
              % (s, 100 * w / W, 100 * w2 / W2, n, w / n))
    print()
