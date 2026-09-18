"""Scan a flat-pT sample: genWeight vs pT-hat (Generator_binvar) vs leading gen AK8 jet pT.

Per gen-pT bin (the dijet ptgen edges) it reports the effective statistics
(sum w)^2 / sum w^2 with and without a jet-pT / pT-hat outlier cut, and lists
the heaviest events.  Runs file-parallel over xrootd; only three branches read.

    python scan_flat_weights.py <fileset.json> <out.json> [nproc]
"""
import json, sys, time
from multiprocessing import Pool

import numpy as np
import uproot

PT_EDGES = np.array([185., 200., 290., 400., 480., 570., 680., 760., 820., 13000.])
RATIO_CUTS = (None, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5)
PREPEND = "root://cms-xrd-global.cern.ch/"


def one_file(path):
    for attempt in range(3):
        try:
            with uproot.open(path, timeout=120) as f:
                ev = f["Events"]
                a = ev.arrays(["genWeight", "Generator_binvar", "GenJetAK8_pt"], library="ak")
            break
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            time.sleep(5)
    else:
        return {"file": path, "error": err}
    import awkward as ak
    w = np.asarray(a["genWeight"], dtype=np.float64)
    pthat = np.asarray(a["Generator_binvar"], dtype=np.float64)
    lead = np.asarray(ak.fill_none(ak.firsts(a["GenJetAK8_pt"]), 0.0), dtype=np.float64)
    ratio = np.where(pthat > 0, lead / np.where(pthat > 0, pthat, 1.0), np.inf)
    ibin = np.digitize(lead, PT_EDGES) - 1
    out = {"file": path, "n": int(len(w)), "sumw": float(w.sum()), "sumw2": float((w * w).sum()),
           "wmax": float(w.max()), "bins": {}}
    for b in range(len(PT_EDGES) - 1):
        m = ibin == b
        row = {}
        for cut in RATIO_CUTS:
            mm = m if cut is None else (m & (ratio < cut))
            ww = w[mm]
            row[str(cut)] = [int(mm.sum()), float(ww.sum()), float((ww * ww).sum())]
        # heaviest 5 events in this bin
        idx = np.flatnonzero(m)
        top = idx[np.argsort(w[idx])[::-1][:5]]
        row["top"] = [[int(i), float(w[i]), float(pthat[i]), float(lead[i])] for i in top]
        out["bins"][str(b)] = row
    # global weight-vs-ratio profile: 2D counts of log10(w) (0.5 wide) x ratio (0.5 wide up to 10)
    return out


def main():
    fileset, outpath = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    top = json.load(open(fileset))
    files = []
    for _g, datasets in top.items():
        for _das, lst in datasets.items():
            for p in lst:
                k = p.find("/store/")
                files.append(PREPEND + p[k:])
    print(len(files), "files", flush=True)
    t0 = time.time()
    with Pool(nproc) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(one_file, files), 1):
            results.append(r)
            if i % 50 == 0:
                print(f"{i}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(outpath, "w"))
    bad = [r for r in results if "error" in r]
    print("done", len(results), "files,", len(bad), "errors", flush=True)
    # summary
    for cut in RATIO_CUTS:
        line = []
        for b in range(len(PT_EDGES) - 1):
            n = sw = sw2 = 0.0
            for r in results:
                if "error" in r:
                    continue
                nn, s, s2 = r["bins"][str(b)][str(cut)]
                n += nn; sw += s; sw2 += s2
            line.append(f"{(sw * sw / sw2 if sw2 > 0 else 0):9.0f}/{n:9.0f}")
        print(f"cut {str(cut):>5s}: " + " ".join(line), flush=True)


if __name__ == "__main__":
    main()
