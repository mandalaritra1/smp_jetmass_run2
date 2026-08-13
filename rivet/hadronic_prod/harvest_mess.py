"""Harvest complete MESS chunks with correct per-HT normalization.

Each QCD-HT gridpack chunk is a statistical partition of the same slice, not
an independent copy of its cross section.  For one HT slice this script uses

    H_HT = sum_i(input_xsec_i * H_i) / sum_i(N_LHE_i)

where H_i contains the nominal MESS event weights.  Scaling every chunk by
``xsec_i / sumw_i`` and adding would count the full HT cross section once per
file and is wrong when a slice has more than one job.

Only chunks with a complete manifest are consumed.  Examples:

    python3 harvest_mess.py /eos/user/a/amandal/hadronic_model_prod_mess \
        mess_trijet_20260802_v1 --channel trijet --dump trijet_mess.npz
"""

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict

import numpy as np


RHO_EDGES = np.array(
    [-10, -5, -4, -3.4, -2.85, -2.25, -1.8, -1.5,
     -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.0]
)
PT_BINS = [(200, 290), (290, 400), (400, 13000)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("eosbase")
    parser.add_argument("campaign")
    parser.add_argument("--channel", choices=("dijet", "trijet"), default="trijet")
    parser.add_argument("--dump")
    return parser.parse_args()


def read_header(path):
    header = {}
    with open(path) as source:
        for line in source:
            # The column header is written first, while Rivet appends xsec and
            # sumw during finalize(), after the event rows.
            if not line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) >= 3 and fields[1] in ("xsec_pb", "sumw"):
                header[fields[1]] = float(fields[2])
    return header


def load_rows(path):
    try:
        rows = np.loadtxt(path, comments="#", ndmin=2)
    except ValueError:
        return np.empty((0, 6))
    return rows if rows.size else np.empty((0, 6))


def read_metrics(path):
    metrics = {}
    with open(path) as source:
        for line in source:
            key, separator, value = line.strip().partition("=")
            if separator:
                metrics[key] = float(value)
    return metrics


def physics_config_hash(manifest):
    """Hash physics settings while excluding per-job LHE path and seed."""
    if "physics_config_sha256" in manifest:
        return manifest["physics_config_sha256"]
    command_path = os.path.join(manifest["_directory"], "run_mess.cmnd")
    with open(command_path, "rb") as source:
        canonical = b"".join(
            line for line in source
            if not line.startswith((b"Beams:LHEF", b"Random:seed"))
        )
    return hashlib.sha256(canonical).hexdigest()


def main():
    args = parse_args()
    campaign_dir = os.path.join(args.eosbase, args.campaign)
    manifest_paths = sorted(glob.glob(os.path.join(campaign_dir, "*", "manifest_*.json")))
    if not manifest_paths:
        raise SystemExit(f"no completion manifests found under {campaign_dir}")

    manifests_by_bin = defaultdict(list)
    expected_hashes = None
    for path in manifest_paths:
        with open(path) as source:
            manifest = json.load(source)
        if manifest.get("complete") is not True:
            continue
        manifest["_directory"] = os.path.dirname(path)
        hashes = (physics_config_hash(manifest), manifest["driver_sha256"],
                  manifest["analysis_sha256"])
        if expected_hashes is None:
            expected_hashes = hashes
        elif hashes != expected_hashes:
            raise RuntimeError(f"configuration hash mismatch in {path}")
        manifests_by_bin[manifest["bin"]].append(manifest)

    total = {
        low: [np.zeros(len(RHO_EDGES) - 1),
              np.zeros(len(RHO_EDGES) - 1),
              np.zeros(len(RHO_EDGES) - 1, dtype=int)]
        for low, _ in PT_BINS
    }
    ht_metrics = {}
    print(f"campaign={args.campaign} channel={args.channel} complete_chunks={len(manifest_paths)}")

    for ht_bin, manifests in sorted(manifests_by_bin.items()):
        total_lhe = sum(int(manifest["lhe_events"]) for manifest in manifests)
        if total_lhe <= 0:
            raise RuntimeError(f"nonpositive generated denominator for {ht_bin}")

        metric_totals = {"requested": 0.0, "analyzed": 0.0,
                         "zero_weight_skipped": 0.0, "next_failed": 0.0}
        for manifest in manifests:
            metrics_path = os.path.join(manifest["_directory"], manifest["metrics"])
            metrics = read_metrics(metrics_path)
            for key in metric_totals:
                metric_totals[key] += metrics.get(key, 0.0)
        ht_metrics[ht_bin] = metric_totals

        slice_sumw = {low: 0.0 for low, _ in PT_BINS}
        slice_sumw2 = {low: 0.0 for low, _ in PT_BINS}
        slice_raw = {low: 0 for low, _ in PT_BINS}
        for manifest in manifests:
            ntuple_name = manifest[f"{args.channel}_ntuple"]
            ntuple_path = os.path.join(manifest["_directory"], ntuple_name)
            header = read_header(ntuple_path)
            if "xsec_pb" not in header or "sumw" not in header:
                raise RuntimeError(f"missing Rivet normalization header in {ntuple_path}")
            if not np.isclose(header["xsec_pb"], manifest["sample_xsec_pb"], rtol=2e-6):
                raise RuntimeError(f"sample cross-section mismatch in {ntuple_path}")

            rows = load_rows(ntuple_path)
            if rows.shape[1] != 6:
                raise RuntimeError(f"unexpected ntuple schema in {ntuple_path}")
            if np.any(rows[:, 5] == 0.0):
                raise RuntimeError(f"zero-weight MESS row found in {ntuple_path}")

            pt = rows[:, 0]
            rho = rows[:, 4]
            # The pooled estimator uses the total number generated in this HT
            # slice. input_xsec can fluctuate slightly between gridpack runs.
            physical_weight = rows[:, 5] * float(manifest["input_xsec_pb"]) / total_lhe
            for low, high in PT_BINS:
                selected = (pt >= low) & (pt < high) & (rho > -99)
                if not selected.any():
                    continue
                hist, _ = np.histogram(rho[selected], RHO_EDGES,
                                       weights=physical_weight[selected])
                hist2, _ = np.histogram(rho[selected], RHO_EDGES,
                                        weights=physical_weight[selected] ** 2)
                raw, _ = np.histogram(rho[selected], RHO_EDGES)
                total[low][0] += hist
                total[low][1] += hist2
                total[low][2] += raw
                slice_sumw[low] += hist.sum()
                slice_sumw2[low] += hist2.sum()
                slice_raw[low] += int(raw.sum())

        summary = []
        for low, _ in PT_BINS:
            neff = slice_sumw[low] ** 2 / slice_sumw2[low] if slice_sumw2[low] else 0.0
            summary.append(
                f"pt>={low}:raw={slice_raw[low]},Neff={neff:.1f},"
                f"yield_pb={slice_sumw[low]:.8g}"
            )
        print(f"  {ht_bin}: chunks={len(manifests)} LHE={total_lhe} " + " ".join(summary))

    for low, high in PT_BINS:
        hist, hist2, raw = total[low]
        neff = hist.sum() ** 2 / hist2.sum() if hist2.sum() else 0.0
        print(f"pT=[{low},{high}) raw={raw.sum()} Neff={neff:.1f} yield_pb={hist.sum():.8g}")

    if args.dump:
        output = {
            "rho_edges": RHO_EDGES,
            "pt_edges": np.array([low for low, _ in PT_BINS] + [PT_BINS[-1][1]]),
            "channel": np.array(args.channel),
            "campaign": np.array(args.campaign),
        }
        ht_names = sorted(ht_metrics)
        output["ht_bins"] = np.array(ht_names)
        for key in ("requested", "analyzed", "zero_weight_skipped", "next_failed"):
            output[f"ht_{key}"] = np.array([ht_metrics[name][key] for name in ht_names])
        for low, _ in PT_BINS:
            hist, hist2, raw = total[low]
            output[f"pt{low}_h"] = hist
            output[f"pt{low}_e"] = np.sqrt(hist2)
            output[f"pt{low}_n"] = raw
        np.savez(args.dump, **output)
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
