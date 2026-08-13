#!/usr/bin/env python3
"""Validate manifest-gated MG+MLM Pythia internal-variation production."""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict


CONFIGS = ("cp5", "cr1", "cr2", "fraghard", "fragsoft")
LEGACY_CP5_XSEC_PB = {
    "QCD_HT300to500": 321895.68,
    "QCD_HT500to700": 29226.296,
    "QCD_HT700to1000": 6280.6886,
    "QCD_HT1000to1500": 1090.6864,
    "QCD_HT1500to2000": 106.60794,
    "QCD_HT2000toInf": 21.484082,
}


def ntuple_summary(path):
    rows = 0
    zero_rows = 0
    xsec_pb = None
    sumw = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# xsec_pb"):
                xsec_pb = float(line.split()[2])
            elif line.startswith("# sumw"):
                sumw = float(line.split()[2])
            elif line.strip() and not line.startswith("#"):
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError(f"malformed ntuple row in {path}: {line.rstrip()}")
                rows += 1
                zero_rows += float(fields[5]) == 0.0
    return rows, zero_rows, xsec_pb, sumw


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="local EOS/CERNBox root containing the campaign")
    parser.add_argument("campaign")
    parser.add_argument("--expected-validation", type=int, default=6)
    parser.add_argument("--expected-production", type=int, default=0)
    parser.add_argument("--mode", choices=("all", "validation", "five_million"), default="all")
    parser.add_argument("--manifest-list")
    parser.add_argument("--snapshot-out")
    parser.add_argument("--dump")
    args = parser.parse_args()

    campaign_dir = os.path.join(args.base, args.campaign)
    if args.manifest_list:
        with open(args.manifest_list, encoding="utf-8") as handle:
            paths = [line.strip() for line in handle if line.strip()]
    else:
        paths = sorted(glob.glob(os.path.join(campaign_dir, "QCD_HT*", "manifest_*.json")))
    errors = []
    manifests = []
    seen = set()
    mode_counts = Counter()
    version_pairs = set()
    driver_hashes = set()
    analysis_hashes = set()
    physics_hashes = defaultdict(set)
    requested_by_mode = Counter()
    lhe_by_mode = Counter()
    selected_paths = []

    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        if args.mode != "all" and manifest.get("mode") != args.mode:
            continue
        selected_paths.append(path)
        key = (manifest.get("bin"), manifest.get("seed"))
        if key in seen:
            errors.append(f"duplicate completion key {key}")
            continue
        seen.add(key)
        manifests.append(manifest)
        mode = manifest.get("mode")
        mode_counts[mode] += 1
        requested_by_mode[mode] += int(manifest.get("requested", 0))
        lhe_by_mode[mode] += int(manifest.get("lhe_events", 0))
        version_pairs.add((manifest.get("pythia_version"), manifest.get("rivet_version")))
        driver_hashes.add(manifest.get("driver_sha256"))
        analysis_hashes.add(manifest.get("analysis_sha256"))

        if manifest.get("campaign") != args.campaign or not manifest.get("complete"):
            errors.append(f"invalid campaign/complete flag in {path}")
        if manifest.get("schema") != 1:
            errors.append(f"unsupported schema in {path}")
        if (manifest.get("pythia_version"), manifest.get("rivet_version")) != ("8.313", "4.0.2"):
            errors.append(f"wrong Pythia/Rivet version in {path}")
        bin_name = manifest.get("bin")
        legacy_xsec = LEGACY_CP5_XSEC_PB.get(bin_name)
        entries = manifest.get("configs", [])
        if tuple(entry.get("config") for entry in entries) != CONFIGS:
            errors.append(f"wrong config order/set in {path}")
            continue
        cp5_xsec = float(entries[0].get("xsec_pb", 0))
        if not legacy_xsec or not 0.5 < cp5_xsec / legacy_xsec < 1.5:
            errors.append(f"CP5 matching xsec outside legacy sanity band in {path}: {cp5_xsec}")

        for entry in entries:
            config = entry["config"]
            physics_hashes[config].add(entry.get("physics_config_sha256"))
            xsec_pb = float(entry.get("xsec_pb", 0))
            sumw = float(entry.get("sumw", 0))
            if not (math.isfinite(xsec_pb) and math.isfinite(sumw) and xsec_pb > 0 and sumw > 0):
                errors.append(f"invalid xsec/sumw for {config} in {path}")
            if cp5_xsec and not 0.5 < xsec_pb / cp5_xsec < 1.5:
                errors.append(f"{config}/CP5 matching xsec ratio outside sanity band in {path}")
            for channel in ("dijet", "trijet"):
                field = f"{channel}_ntuple"
                ntuple = os.path.join(os.path.dirname(path), entry.get(field, ""))
                if not os.path.isfile(ntuple) or not os.path.getsize(ntuple):
                    errors.append(f"missing {field} for {config} in {path}")
                    continue
                try:
                    rows, zero_rows, file_xsec, file_sumw = ntuple_summary(ntuple)
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))
                    continue
                if rows != int(entry.get(f"{channel}_rows", -1)):
                    errors.append(f"row-count mismatch for {config} {channel} in {path}")
                if zero_rows or int(entry.get(f"{channel}_zero_rows", -1)) != 0:
                    errors.append(f"zero-weight rows for {config} {channel} in {path}")
                if not close(file_xsec, xsec_pb) or not close(file_sumw, sumw):
                    errors.append(f"header/manifest mismatch for {config} {channel} in {path}")
            for field in ("yoda", "cmnd", "log"):
                output = os.path.join(os.path.dirname(path), entry.get(field, ""))
                if not os.path.isfile(output) or not os.path.getsize(output):
                    errors.append(f"missing {field} for {config} in {path}")

    expected_counts = {
        "validation": args.expected_validation,
        "five_million": args.expected_production,
    }
    for mode, expected in expected_counts.items():
        if mode_counts[mode] != expected:
            errors.append(f"expected {expected} {mode} manifests, found {mode_counts[mode]}")
    if version_pairs and version_pairs != {("8.313", "4.0.2")}:
        errors.append(f"inconsistent versions: {sorted(version_pairs)}")
    if len(driver_hashes) > 1 or len(analysis_hashes) > 1:
        errors.append("inconsistent driver or analysis hashes")
    for config, hashes in physics_hashes.items():
        if len(hashes) > 1:
            errors.append(f"inconsistent physics config hash for {config}")

    summary = {
        "campaign": args.campaign,
        "complete": not errors,
        "manifest_count": len(manifests),
        "mode_counts": dict(mode_counts),
        "requested_events": dict(requested_by_mode),
        "lhe_events": dict(lhe_by_mode),
        "versions": sorted([list(pair) for pair in version_pairs]),
        "driver_hashes": sorted(driver_hashes),
        "analysis_hashes": sorted(analysis_hashes),
        "physics_config_hashes": {key: sorted(value) for key, value in physics_hashes.items()},
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.snapshot_out:
        with open(args.snapshot_out, "w", encoding="utf-8") as handle:
            for path in selected_paths:
                handle.write(f"{os.path.abspath(path)}\n")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
