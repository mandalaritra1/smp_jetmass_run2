"""Pool the matched-gridpack CP5 chunks once per QCD-HT slice.

For a chunk, ``xsec_pb / sumw`` is its pb-per-generated-event coefficient.
Chunks in one HT directory are partitions of one slice, so the pooled slice is

    H_HT = sum_i(H_i) / sum_i(sumw_i / xsec_i)

not the sum of individually cross-section-normalized chunks.
"""

import argparse
import glob
import os

import numpy as np


RHO_EDGES = np.array(
    [-10, -5, -4, -3.4, -2.85, -2.25, -1.8, -1.5,
     -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.0]
)
PT_BINS = [(200, 290), (290, 400), (400, 13000)]


def read_header(path):
    values = {}
    with open(path) as source:
        for line in source:
            if line.startswith("# xsec_pb") or line.startswith("# sumw"):
                fields = line.split()
                values[fields[1]] = float(fields[2])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eosbase")
    parser.add_argument("--channel", choices=("dijet", "trijet"), required=True)
    parser.add_argument("--dump", required=True)
    args = parser.parse_args()

    totals = {
        low: [np.zeros(len(RHO_EDGES) - 1),
              np.zeros(len(RHO_EDGES) - 1),
              np.zeros(len(RHO_EDGES) - 1, dtype=int)]
        for low, _ in PT_BINS
    }
    ht_directories = sorted(glob.glob(os.path.join(args.eosbase, "cp5", "QCD_HT*")))
    if not ht_directories:
        raise SystemExit(f"no CP5 HT directories under {args.eosbase}")

    for directory in ht_directories:
        files = sorted(glob.glob(os.path.join(directory, f"ntuple_*_{args.channel}.txt")))
        denominators = []
        rows_by_file = []
        for path in files:
            header = read_header(path)
            if header.get("xsec_pb", 0) <= 0 or header.get("sumw", 0) <= 0:
                continue
            denominators.append(header["sumw"] / header["xsec_pb"])
            try:
                rows = np.loadtxt(path, comments="#", ndmin=2)
            except ValueError:
                rows = np.empty((0, 6))
            if rows.size:
                rows_by_file.append(rows)
        denominator = sum(denominators)
        if denominator <= 0:
            continue

        slice_raw = {low: 0 for low, _ in PT_BINS}
        for rows in rows_by_file:
            pt, rho = rows[:, 0], rows[:, 4]
            weight = rows[:, 5] / denominator
            for low, high in PT_BINS:
                selected = (pt >= low) & (pt < high) & (rho > -99) & (rows[:, 5] != 0)
                if not selected.any():
                    continue
                hist, _ = np.histogram(rho[selected], RHO_EDGES, weights=weight[selected])
                hist2, _ = np.histogram(rho[selected], RHO_EDGES,
                                        weights=weight[selected] ** 2)
                raw, _ = np.histogram(rho[selected], RHO_EDGES)
                totals[low][0] += hist
                totals[low][1] += hist2
                totals[low][2] += raw
                slice_raw[low] += int(raw.sum())
        print(f"{os.path.basename(directory)} files={len(files)} "
              + " ".join(f"pt{low}={slice_raw[low]}" for low, _ in PT_BINS))

    output = {
        "rho_edges": RHO_EDGES,
        "pt_edges": np.array([low for low, _ in PT_BINS] + [PT_BINS[-1][1]]),
        "channel": np.array(args.channel),
        "sample": np.array("MG+MLM+CP5"),
    }
    for low, _ in PT_BINS:
        hist, hist2, raw = totals[low]
        output[f"pt{low}_h"] = hist
        output[f"pt{low}_e"] = np.sqrt(hist2)
        output[f"pt{low}_n"] = raw
        neff = hist.sum() ** 2 / hist2.sum() if hist2.sum() else 0.0
        print(f"pT={low} raw={raw.sum()} Neff={neff:.1f} yield_pb={hist.sum():.8g}")
    np.savez(args.dump, **output)
    print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
