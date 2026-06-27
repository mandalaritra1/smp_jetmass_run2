#!/usr/bin/env python3
"""Build a Rivet reference-data YODA from the unfolded zjet rho HepData export.

Reads refdata/hepdata_export_{groomed,ungroomed}.npz and writes
refdata/CMS_ZJET_JETMASS.yoda with /REF/CMS_ZJET_JETMASS/rho_{g,u}_pt<bin>
Scatter2D objects (per-pT-bin unit-area density 1/sigma dsigma/dx, x=log10(rho^2),
total up/down uncertainties). The first (wide) control bin is excluded.
"""
import numpy as np
import yoda

PTMAP = {0: "pt200_290", 1: "pt290_400", 2: "pt400_Inf"}
SPECS = [("groomed", "rho_g"), ("ungroomed", "rho_u")]

aos = []
for tag, obs in SPECS:
    z = np.load(f"refdata/hepdata_export_{tag}.npz")
    for i in range(3):
        edges = z[f"pt{i}__edges"]
        val = z[f"pt{i}__value"]
        up = z[f"pt{i}__total_up"]
        dn = z[f"pt{i}__total_down"]
        s = yoda.Scatter2D(path=f"/REF/CMS_ZJET_JETMASS/{obs}_{PTMAP[i]}")
        for b in range(len(val)):
            if b == 0:          # wide control bin (excluded)
                continue
            lo, hi = float(edges[b]), float(edges[b + 1])
            xc, xw = 0.5 * (lo + hi), 0.5 * (hi - lo)
            s.addPoint(yoda.Point2D(xc, float(val[b]), xw, xw,
                                    abs(float(dn[b])), abs(float(up[b]))))
        aos.append(s)

yoda.write(aos, "refdata/CMS_ZJET_JETMASS.yoda")
print(f"wrote refdata/CMS_ZJET_JETMASS.yoda ({len(aos)} REF objects)")
