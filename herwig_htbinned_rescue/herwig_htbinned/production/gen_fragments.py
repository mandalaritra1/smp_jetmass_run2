#!/usr/bin/env python3
"""Make 6 HT-binned cff fragments from the finalized Herwig7_DY_HT_CH3_fragment.py
by swapping ONLY the gridpack bin token (200to400 -> <BIN>). Prints per-bin diff."""
import os, re, difflib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "inputs", "Herwig7_DY_HT_CH3_fragment.py")
OUTD = os.path.join(ROOT, "Configuration", "GenProduction", "python")
BINS = ["200to400","400to600","600to800","800to1200","1200to2500","2500toInf"]
base = open(SRC).read()
assert base.count("DYJets_HT-200to400_slc7") == 1, "expected exactly one 200to400 gridpack token"
os.makedirs(OUTD, exist_ok=True)
for b in BINS:
    new = base.replace("DYJets_HT-200to400_slc7", "DYJets_HT-%s_slc7" % b)
    out = os.path.join(OUTD, "Herwig7_DY_HT-%s_CH3_cff.py" % b)
    open(out, "w").write(new)
    ch = [d for d in difflib.unified_diff(base.splitlines(True), new.splitlines(True))
          if d[:1] in "+-" and d[:3] not in ("+++","---")]
    print("=== HT-%s (%d changed lines)" % (b, len(ch)))
    for d in ch: print("   ", d.rstrip())
