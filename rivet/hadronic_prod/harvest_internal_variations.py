#!/usr/bin/env python3
"""Manifest-only, per-HT pooled harvest for Pythia internal variations."""

import argparse
import glob
import json
import os

import numpy as np


CONFIGS = ("cp5", "cr1", "cr2", "fraghard", "fragsoft")
PT_BINS = ((200, 290), (290, 400), (400, 13000))
RHO_EDGES = np.array([-10, -5, -4, -3.4, -2.85, -2.25, -1.8, -1.5,
                      -1.3, -1.1, -0.9, -0.75, -0.65, -0.55, 0.0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("campaign")
    parser.add_argument("--channel", choices=("dijet", "trijet"), required=True)
    parser.add_argument("--mode", choices=("all", "validation", "five_million"), default="all")
    parser.add_argument("--manifest-list")
    parser.add_argument("--dump", required=True)
    args = parser.parse_args()

    campaign_dir = os.path.join(args.base, args.campaign)
    if args.manifest_list:
        with open(args.manifest_list, encoding="utf-8") as handle:
            manifest_paths = [line.strip() for line in handle if line.strip()]
    else:
        manifest_paths = sorted(glob.glob(os.path.join(campaign_dir, "QCD_HT*", "manifest_*.json")))
    manifests = []
    seen = set()
    for path in manifest_paths:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not manifest.get("complete") or manifest.get("campaign") != args.campaign:
            continue
        if args.mode != "all" and manifest.get("mode") != args.mode:
            continue
        key = (manifest["bin"], manifest["seed"])
        if key in seen:
            raise RuntimeError(f"duplicate manifest key {key}")
        seen.add(key)
        manifests.append((path, manifest))
    if not manifests:
        raise RuntimeError("no matching completion manifests")

    bins = sorted({manifest["bin"] for _, manifest in manifests})
    chunk_hist = {
        (config, pt_low): np.zeros((len(manifests), len(RHO_EDGES) - 1))
        for config in CONFIGS for pt_low, _ in PT_BINS
    }
    chunk_hist2 = {
        (config, pt_low): np.zeros((len(manifests), len(RHO_EDGES) - 1))
        for config in CONFIGS for pt_low, _ in PT_BINS
    }
    chunk_count = {
        (config, pt_low): np.zeros((len(manifests), len(RHO_EDGES) - 1), dtype=np.int64)
        for config in CONFIGS for pt_low, _ in PT_BINS
    }
    chunk_denominator = {
        config: np.zeros(len(manifests)) for config in CONFIGS
    }
    accum = {
        (config, bin_name, pt_low): {
            "raw_h": np.zeros(len(RHO_EDGES) - 1),
            "raw_h2": np.zeros(len(RHO_EDGES) - 1),
            "raw_n": np.zeros(len(RHO_EDGES) - 1, dtype=np.int64),
            "denominator": 0.0,
        }
        for config in CONFIGS for bin_name in bins for pt_low, _ in PT_BINS
    }

    for chunk_index, (path, manifest) in enumerate(manifests):
        entries = {entry["config"]: entry for entry in manifest["configs"]}
        if tuple(entries) != CONFIGS:
            raise RuntimeError(f"wrong config set in {path}")
        for config in CONFIGS:
            entry = entries[config]
            ntuple = os.path.join(os.path.dirname(path), entry[f"{args.channel}_ntuple"])
            values = np.loadtxt(ntuple, comments="#", ndmin=2)
            if values.size == 0:
                values = np.empty((0, 6))
            denominator = float(entry["sumw"]) / float(entry["xsec_pb"])
            if denominator <= 0:
                raise RuntimeError(f"invalid generated-luminosity denominator in {path}")
            pt = values[:, 0]
            rho = values[:, 4]
            weight = values[:, 5]
            chunk_denominator[config][chunk_index] = denominator
            for pt_low, pt_high in PT_BINS:
                target = accum[(config, manifest["bin"], pt_low)]
                target["denominator"] += denominator
                mask = (pt >= pt_low) & (pt < pt_high) & (rho > -99)
                raw_h = np.histogram(rho[mask], RHO_EDGES, weights=weight[mask])[0]
                raw_h2 = np.histogram(rho[mask], RHO_EDGES, weights=weight[mask] ** 2)[0]
                raw_n = np.histogram(rho[mask], RHO_EDGES)[0]
                target["raw_h"] += raw_h
                target["raw_h2"] += raw_h2
                target["raw_n"] += raw_n
                chunk_hist[(config, pt_low)][chunk_index] = raw_h
                chunk_hist2[(config, pt_low)][chunk_index] = raw_h2
                chunk_count[(config, pt_low)][chunk_index] = raw_n

    output = {
        "configs": np.array(CONFIGS),
        "rho_edges": RHO_EDGES,
        "pt_edges": np.array([200, 290, 400, 13000]),
        "manifest_count": np.array(len(manifests)),
        "requested_events": np.array(sum(manifest["requested"] for _, manifest in manifests)),
        "lhe_events": np.array(sum(manifest["lhe_events"] for _, manifest in manifests)),
        "channel": np.array(args.channel),
        "mode": np.array(args.mode),
        "campaign": np.array(args.campaign),
        "chunk_bins": np.array([manifest["bin"] for _, manifest in manifests]),
        "chunk_seeds": np.array([manifest["seed"] for _, manifest in manifests]),
    }
    for config in CONFIGS:
        output[f"{config}_chunk_denominator"] = chunk_denominator[config]
        for pt_low, _ in PT_BINS:
            hist = np.zeros(len(RHO_EDGES) - 1)
            variance = np.zeros(len(RHO_EDGES) - 1)
            raw_n = np.zeros(len(RHO_EDGES) - 1, dtype=np.int64)
            for bin_name in bins:
                source = accum[(config, bin_name, pt_low)]
                denominator = source["denominator"]
                if denominator == 0:
                    continue
                hist += source["raw_h"] / denominator
                variance += source["raw_h2"] / denominator ** 2
                raw_n += source["raw_n"]
            prefix = f"{config}_pt{pt_low}"
            output[f"{prefix}_h"] = hist
            output[f"{prefix}_sumw2"] = variance
            output[f"{prefix}_n"] = raw_n
            neff = hist.sum() ** 2 / variance.sum() if variance.sum() else 0.0
            output[f"{prefix}_neff"] = np.array(neff)
            output[f"{prefix}_chunk_h"] = chunk_hist[(config, pt_low)]
            output[f"{prefix}_chunk_h2"] = chunk_hist2[(config, pt_low)]
            output[f"{prefix}_chunk_n"] = chunk_count[(config, pt_low)]
            print(f"{config:9s} pt>={pt_low:3d}: raw={raw_n.sum():8d} Neff={neff:10.1f} yield_pb={hist.sum():.8g}")

    np.savez(args.dump, **output)
    print(f"wrote {args.dump} from {len(manifests)} unique manifests")


if __name__ == "__main__":
    main()
