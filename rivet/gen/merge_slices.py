#!/usr/bin/env python3
"""Stitch disjoint pThat (or HT) slices into one prediction.

Each slice's Rivet YODA holds the finalized, absolute (pb) differential
cross sections dsigma/dx -- already scaled by crossSection()/sumOfWeights() in
the routine's finalize(). The slices are disjoint in pThat, so the correct
stitch is the bin-by-bin SUM of the finalized estimates (values add, statistical
errors combine in quadrature -- exactly what YODA's '+=' does).

Usage:
    merge_slices.py <out.yoda> <slice1.yoda> <slice2.yoda> ...
"""
import sys
import yoda

if len(sys.argv) < 3:
    sys.exit("usage: merge_slices.py <out.yoda> <slice1.yoda> ...")

outfile, infiles = sys.argv[1], sys.argv[2:]
acc = {}
for f in infiles:
    for path, ao in yoda.read(f).items():
        # only the finalized analysis objects (skip /RAW, /TMP, /_XSEC, ...)
        if not path.startswith("/CMS_ZJET_JETMASS/"):
            continue
        if path in acc:
            acc[path] += ao
        else:
            acc[path] = ao.clone()

yoda.write(acc, outfile)
print(f"stitched {len(infiles)} slices -> {outfile} ({len(acc)} objects)")
