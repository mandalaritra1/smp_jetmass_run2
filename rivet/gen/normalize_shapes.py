#!/usr/bin/env python3
"""Area-normalise each CMS_ZJET_JETMASS distribution to unit integral.

rivet-mkhtml's NormalizeToIntegral directive is a no-op on the finalized
BinnedEstimate objects this routine produces, so the shape comparison has to be
normalised here instead: each observable (per pT bin) is divided by its own
integral, giving 1/sigma dsigma/dx. Normalisation is over the full histogram
range (the plot view may be cropped).

    normalize_shapes.py <in.yoda> <out.yoda>
"""
import sys
import yoda

infile, outfile = sys.argv[1], sys.argv[2]
out = {}
for path, ao in yoda.read(infile).items():
    if not path.startswith("/CMS_ZJET_JETMASS/"):
        continue
    ao = ao.clone()
    integral = sum(b.val() * (b.xMax() - b.xMin()) for b in ao.bins())
    if integral > 0:
        ao.scale(1.0 / integral)
    out[path] = ao
yoda.write(out, outfile)
