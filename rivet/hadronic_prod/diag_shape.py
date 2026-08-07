"""Compare normalized groomed-rho shapes between configs showered from the
SAME LHE (diag_matching.sh workdir).  Usage: python3 diag_shape.py <workdir>"""
import sys
import numpy as np

work = sys.argv[1] if len(sys.argv) > 1 else "."
edges = np.array([-10, -5, -4, -3.4, -2.85, -2.25, -1.8, -1.5,
                  -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.])
configs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cp5", "vincia"]

shapes = {}
for c in configs:
    a = np.loadtxt("%s/nt_%s_dijet.txt" % (work, c))
    pt, rho_g = a[:, 0], a[:, 4]
    m = (pt > 200) & (rho_g > -99)
    h, _ = np.histogram(rho_g[m], bins=edges)
    shapes[c] = h.astype(float)
    print("%-8s jets pt>200: %d" % (c, int(h.sum())))

ref = configs[0]
nref = shapes[ref] / shapes[ref].sum()
print("\n%-18s %10s %10s %8s" % ("rho_g bin", ref, configs[1], "ratio"))
for i in range(len(edges) - 1):
    nvar = shapes[configs[1]] / shapes[configs[1]].sum()
    r = nvar[i] / nref[i] if nref[i] > 0 else float("nan")
    print("[%6.2f,%6.2f] %10.4f %10.4f %8s"
          % (edges[i], edges[i + 1], nref[i], nvar[i],
             "%.2f" % r if np.isfinite(r) else "--"))

# core measured window
core = (edges[:-1] >= -2.85) & (edges[:-1] < -0.55)
nvar = shapes[configs[1]] / shapes[configs[1]].sum()
dev = np.abs(nvar[core] / nref[core] - 1)
print("\ncore window [-2.85,-0.55]: max |ratio-1| = %.1f%%, mean = %.1f%%"
      % (100 * np.nanmax(dev), 100 * np.nanmean(dev)))
