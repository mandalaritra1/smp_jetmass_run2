#!/usr/bin/env python3
"""Combine independent gen samples into one high-stats prediction per variation.

Two distinct operations, in the statistically correct order:
  1. SAME slice, multiple independent samples (e.g. LPC-50k + akasha-50k for s2/s3):
     INVERSE-VARIANCE AVERAGE per bin -- both are estimates of the same dsigma/dx, so
     they must be averaged (NOT summed; summing would double the cross section).
        v = sum(v_k/sig_k^2) / sum(1/sig_k^2),   sig = 1/sqrt(sum(1/sig_k^2))
  2. DISJOINT pT slices (s0..s3): SUM per bin (each event lands in one slice).
        V = sum_slice v_slice,   sig = sqrt(sum_slice sig_slice^2)

Inputs are finalized Rivet YODA ESTIMATE1D_V3 (absolute dsigma/dx, pb, with 'stats'
errors). Pure stdlib+numpy, parses/writes YODA3 text directly (no yoda module), so the
output is read by plot_model_uncertainty.py / plot_model_unc_stat.py.

Usage:
    combine_hires.py <out_prefix> <dir1> [dir2 ...]
e.g.
    combine_hires.py out/hi out/slices_50k out/slices_lpc
writes out/hi_<variation>.yoda for each variation found.
"""
import os, re, sys, glob
import numpy as np

VARIATIONS = ["pythia", "pythia_cr1", "pythia_cr2",
              "pythia_fragsoft", "pythia_fraghard", "pythia_vincia"]
SLICES = [0, 1, 2, 3]


def parse(path):
    """{hpath: (edges, body_vals, body_errs)} for ESTIMATE1D_V3 under /CMS_ZJET_JETMASS/.
    body drops the underflow/overflow rows; errs are the (symmetric) 'stats' errUp."""
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


def invvar_average(samples):
    """samples: list of (edges, v, e) for the SAME slice. Returns (edges, v, e)."""
    ed = samples[0][0]
    V = np.stack([s[1] for s in samples])
    E = np.stack([s[2] for s in samples])
    w = np.where(E > 0, 1.0 / E**2, 0.0)
    sw = w.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(sw > 0, (w * V).sum(axis=0) / sw, V.mean(axis=0))
        e = np.where(sw > 0, 1.0 / np.sqrt(sw), 0.0)
    return ed, np.nan_to_num(v), np.nan_to_num(e)


def write_yoda(path, hists):
    """hists: {hpath_key: (edges, v, e)} -> ESTIMATE1D_V3 text the plotters can read."""
    with open(path, "w") as fh:
        for key, (ed, v, e) in hists.items():
            full = f"/CMS_ZJET_JETMASS/{key}"
            fh.write(f"BEGIN YODA_ESTIMATE1D_V3 {full}\n")
            fh.write(f"Path: {full}\nTitle: ~\nType: Estimate1D\n---\n")
            edstr = ", ".join(f"{x:.6e}" for x in ed)
            fh.write(f"Edges(A1): [{edstr}]\n")
            fh.write('ErrorLabels: ["stats"]\n')
            fh.write("# value\terrDn(1)\terrUp(1)\n")
            fh.write("nan\t---\t---\n")                      # underflow
            for val, err in zip(v, e):
                fh.write(f"{val:.6e}\t{-err:.6e}\t{err:.6e}\n")
            fh.write("nan\t---\t---\n")                      # overflow
            fh.write("END YODA_ESTIMATE1D_V3\n\n")


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: combine_hires.py <out_prefix> <dir1> [dir2 ...]")
    prefix, dirs = sys.argv[1], sys.argv[2:]
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    for name in VARIATIONS:
        # gather, per slice, all sample files across the input dirs
        per_slice = {}
        for idx in SLICES:
            files = [os.path.join(d, f"mglo_{name}_s{idx}.yoda") for d in dirs]
            files = [f for f in files if os.path.exists(f)]
            if files:
                per_slice[idx] = [parse(f) for f in files]
        if not per_slice:
            print(f"skip {name} (no slices found)"); continue

        # 1. average same-slice samples; 2. sum disjoint slices
        combined = {}
        report = []
        for idx, samp_list in per_slice.items():
            keys = set().union(*[set(s.keys()) for s in samp_list])
            for key in keys:
                got = [s[key] for s in samp_list if key in s]
                ed, v, e = invvar_average(got)
                if key not in combined:
                    combined[key] = [ed, v.copy(), e.copy()]
                else:
                    combined[key][1] += v
                    combined[key][2] = np.sqrt(combined[key][2]**2 + e**2)
            report.append(f"s{idx}x{len(samp_list)}")
        out = {k: (c[0], c[1], c[2]) for k, c in combined.items()}
        path = f"{prefix}_{name}.yoda"
        write_yoda(path, out)
        print(f"wrote {path}  [{', '.join(sorted(report))}]  ({len(out)} histos)")


if __name__ == "__main__":
    main()
