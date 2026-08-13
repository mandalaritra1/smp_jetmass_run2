"""Validate completion manifests before pooling a MESS+Vincia campaign."""

import argparse
import glob
import hashlib
import json
import math
import os
from collections import Counter


VALIDATION_SEEDS = {7_300_000, 7_400_000, 7_500_000,
                    7_600_000, 7_700_000, 7_800_000}


def physics_config_hash(manifest):
    if manifest.get("physics_config_sha256"):
        return manifest["physics_config_sha256"]
    command_path = os.path.join(os.path.dirname(manifest["_path"]), "run_mess.cmnd")
    with open(command_path, "rb") as source:
        canonical = b"".join(
            line for line in source
            if not line.startswith((b"Beams:LHEF", b"Random:seed"))
        )
    return hashlib.sha256(canonical).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eosbase")
    parser.add_argument("campaign")
    parser.add_argument("--expected-production", type=int, required=True)
    parser.add_argument("--dump", required=True)
    args = parser.parse_args()

    campaign_dir = os.path.join(args.eosbase, args.campaign)
    paths = sorted(glob.glob(os.path.join(campaign_dir, "*", "manifest_*.json")))
    manifests = []
    for path in paths:
        with open(path) as source:
            manifest = json.load(source)
        manifest["_path"] = path
        manifests.append(manifest)

    validation = [m for m in manifests if int(m["seed"]) in VALIDATION_SEEDS]
    production = [m for m in manifests if int(m["seed"]) not in VALIDATION_SEEDS]
    errors = []
    if len(validation) != 6:
        errors.append(f"expected 6 validation manifests, found {len(validation)}")
    if len(production) != args.expected_production:
        errors.append(
            f"expected {args.expected_production} production manifests, "
            f"found {len(production)}"
        )

    keys = [(m.get("bin"), int(m.get("seed", -1))) for m in manifests]
    duplicates = [key for key, count in Counter(keys).items() if count != 1]
    if duplicates:
        errors.append(f"duplicate bin/seed manifest keys: {duplicates[:5]}")

    hash_tuples = {
        (physics_config_hash(m), m.get("driver_sha256"), m.get("analysis_sha256"))
        for m in manifests
    }
    if len(hash_tuples) != 1:
        errors.append(f"found {len(hash_tuples)} physics/driver/analysis hash tuples")

    requested = 0
    lhe_events = 0
    per_bin = Counter()
    for manifest in manifests:
        label = manifest["_path"]
        if manifest.get("complete") is not True:
            errors.append(f"incomplete manifest: {label}")
        if manifest.get("campaign") != args.campaign:
            errors.append(f"campaign mismatch: {label}")
        if manifest.get("pythia_version") != "8.313":
            errors.append(f"Pythia version mismatch: {label}")
        if manifest.get("rivet_version") != "4.0.2":
            errors.append(f"Rivet version mismatch: {label}")
        for channel in ("dijet", "trijet"):
            if int(manifest[f"{channel}_rows"]) != int(manifest[f"{channel}_nonzero_rows"]):
                errors.append(f"zero-weight {channel} rows: {label}")
        for field in ("input_xsec_pb", "sample_xsec_pb"):
            value = float(manifest[field])
            if not math.isfinite(value) or value <= 0:
                errors.append(f"invalid {field}: {label}")
        error = float(manifest["sample_xsec_err_pb"])
        if not math.isfinite(error) or error < 0:
            errors.append(f"invalid sample_xsec_err_pb: {label}")
        directory = os.path.dirname(label)
        for field in ("metrics", "dijet_ntuple", "trijet_ntuple", "yoda"):
            if not os.path.isfile(os.path.join(directory, manifest[field])):
                errors.append(f"missing {field}: {label}")
        requested += int(manifest["requested"])
        lhe_events += int(manifest["lhe_events"])
        per_bin[manifest["bin"]] += 1

    summary = {
        "campaign": args.campaign,
        "valid": not errors,
        "manifest_count": len(manifests),
        "validation_manifests": len(validation),
        "production_manifests": len(production),
        "requested_events_including_validation": requested,
        "lhe_events_including_validation": lhe_events,
        "per_bin_manifests": dict(sorted(per_bin.items())),
        "physics_driver_analysis_hashes": list(next(iter(hash_tuples), ())),
        "errors": errors,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
    with open(args.dump, "w") as target:
        json.dump(summary, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
