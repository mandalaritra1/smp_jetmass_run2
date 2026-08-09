#!/usr/bin/env python3
"""Create and validate the canary completion manifest, staged last."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_optional(path: Path) -> str | None:
    """Return a stripped one-line file, or None when the step did not run."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return path.read_text().strip()


def footer(path: Path, name: str, default: float | None = None) -> float:
    pattern = re.compile(rf"^# {re.escape(name)}\s+(.+)$")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    if default is not None:
        return default
    raise ValueError(f"missing {name} footer in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--shower", required=True)
    parser.add_argument("--slice-index", type=int, required=True)
    parser.add_argument("--pt-low", type=int, required=True)
    parser.add_argument("--pt-high", type=int, required=True)
    parser.add_argument("--requested-events", type=int, required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--qcut", type=float, default=20.0)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=310_000_000)
    args = parser.parse_args()

    run = args.run_directory
    outputs = [
        run / "pair_split.yoda",
        run / "pairsplit_dijet.txt",
        run / "pairsplit_trijet.txt",
        run / "events.txt",
        run / "Sherpa.yaml",
        run / "selection_validation.json",
        run / "environment.txt",
        run / "lhapdf_sets.txt",
        run / "Results.zip",
        run / "References.tex",
        run / "sherpa.log",
        run / "rivet.log",
        run / "rivet_build.log",
        run / "shower_generators.log",
        run / "wall_seconds.txt",
        run / "sherpa_version.txt",
    ]
    settings_report = run / "Settings_Report/Settings_Report.md"
    if settings_report.is_file() and settings_report.stat().st_size > 0:
        outputs.append(settings_report)
    missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit(f"missing or empty outputs: {missing}")

    event_file = run / "events.txt"
    manifest = {
        "schema_version": 1,
        "campaign": "pairsplit_2018_sherpa3_canary",
        "generator": "Sherpa",
        "shower": args.shower,
        "hadronization": "Ahadic",
        "me_generator": "Comix",
        "me_multiplicities": ["2to2", "2to3", "2to4"],
        "qcut_GeV": args.qcut,
        "slice_index": args.slice_index,
        "chunk_index": args.chunk_index,
        "seed": args.base_seed + 10_000 * args.slice_index + args.chunk_index,
        # Which integration grid this chunk consumed, and the one it produced.
        # Chunks of a slice must agree on grid_in_sha256 for their cross
        # sections to be comparable.
        "grid_in_sha256": read_optional(run / "grid_in.sha256"),
        "grid_out_sha256": read_optional(run / "grid_out.sha256"),
        "slice_low_GeV": args.pt_low,
        "slice_high_GeV": args.pt_high,
        "requested_events": args.requested_events,
        "generated_events": int(footer(event_file, "generated_events")),
        "cross_section_pb": footer(event_file, "cross_section_pb"),
        "cross_section_error_pb": footer(
            event_file, "cross_section_error_pb", default=0.0
        ),
        "sumw": footer(event_file, "sumw"),
        "raw_sumw": footer(event_file, "raw_sumw"),
        "sumw2": footer(event_file, "sumw2"),
        "wall_seconds": int((run / "wall_seconds.txt").read_text()),
        "sherpa_version": (run / "sherpa_version.txt").read_text().strip(),
        "environment": (run / "environment.txt").read_text().splitlines(),
        "config_sha256": sha256(run / "Sherpa.yaml"),
        "rivet_source_sha256": sha256(args.source),
        "source_bundle_sha256": args.source_bundle_sha256,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        },
        "selection_validation": json.loads((run / "selection_validation.json").read_text()),
        "status": "complete",
    }
    if manifest["generated_events"] != args.requested_events:
        raise SystemExit("generated event count does not match request")
    if manifest["selection_validation"]["status"] != "pass":
        raise SystemExit("selection validation failed")
    if not manifest["cross_section_pb"] > 0 or manifest["sumw"] == 0:
        raise SystemExit("invalid normalization metadata")
    (run / "completion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
