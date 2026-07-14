#!/usr/bin/env python3
"""Inverse-variance average N chunk yodas (disjoint event subsets of ONE slice/variation,
each a correctly-normalised dsigma/dx estimate) into the full-stats slice yoda.

This is the merge half of the split-LHE parallelism: split a slice's LHE into N chunks,
shower each as a separate (condor) job, then average the N chunk yodas here -> identical
to the monolithic full-stats result, but produced in ~one-chunk wall time.

Usage: chunk_merge.py <out.yoda> <chunk1.yoda> <chunk2.yoda> ...
"""
import os, re, sys
import numpy as np


def parse(path):
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = re.match(r"BEGIN YODA_ESTIMATE1D_V3 (\S+)", lines[i])
        if not m:
            i += 1; continue
        hp = m.group(1); edges = None; vals = []; errs = []; tab = False; i += 1
        while i < n and not lines[i].startswith("END YODA_ESTIMATE1D_V3"):
            ln = lines[i]
            if ln.startswith("Edges(A1):"):
                edges = [float(x) for x in re.findall(r"[-+0-9.eE]+", ln.split(":", 1)[1])]
            elif ln.startswith("# value"):
                tab = True
            elif tab:
                tok = ln.split()
                if tok:
                    if tok[0] == "nan":
                        vals.append(np.nan); errs.append(np.nan)
                    else:
                        vals.append(float(tok[0]))
                        errs.append(abs(float(tok[2])) if len(tok) > 2 and tok[2] != "---" else 0.0)
            i += 1
        if hp.startswith("/CMS_ZJET_JETMASS/") and edges is not None and len(vals) >= 2:
            v = np.nan_to_num(np.array(vals[1:-1])); e = np.nan_to_num(np.array(errs[1:-1]))
            ed = np.array(edges)
            if len(v) == len(ed) - 1:
                out[hp.split("/")[-1]] = (ed, v, e)
        i += 1
    return out


def write_yoda(path, hists):
    with open(path, "w") as fh:
        for key, (ed, v, e) in hists.items():
            full = f"/CMS_ZJET_JETMASS/{key}"
            fh.write(f"BEGIN YODA_ESTIMATE1D_V3 {full}\nPath: {full}\nTitle: ~\nType: Estimate1D\n---\n")
            fh.write("Edges(A1): [" + ", ".join(f"{x:.6e}" for x in ed) + "]\n")
            fh.write('ErrorLabels: ["stats"]\n# value\terrDn(1)\terrUp(1)\n')
            fh.write("nan\t---\t---\n")
            for val, err in zip(v, e):
                fh.write(f"{val:.6e}\t{-err:.6e}\t{err:.6e}\n")
            fh.write("nan\t---\t---\nEND YODA_ESTIMATE1D_V3\n\n")


def main():
    out, ins = sys.argv[1], sys.argv[2:]
    parsed = [parse(f) for f in ins]
    parsed = [p for p in parsed if p]
    if not parsed:
        sys.exit("no valid chunk yodas")
    keys = set().union(*[set(p.keys()) for p in parsed])
    merged = {}
    for key in keys:
        samp = [p[key] for p in parsed if key in p]
        ed = samp[0][0]
        V = np.stack([s[1] for s in samp]); E = np.stack([s[2] for s in samp])
        w = np.where(E > 0, 1.0 / E**2, 0.0); sw = w.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(sw > 0, (w * V).sum(axis=0) / sw, V.mean(axis=0))
            e = np.where(sw > 0, 1.0 / np.sqrt(sw), 0.0)
        merged[key] = (ed, np.nan_to_num(v), np.nan_to_num(e))
    write_yoda(out, merged)
    print(f"merged {len(parsed)} chunks -> {out} ({len(merged)} histos)")


if __name__ == "__main__":
    main()
