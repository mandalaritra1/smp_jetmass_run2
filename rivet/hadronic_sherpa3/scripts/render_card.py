#!/usr/bin/env python3
"""Render one deterministic Sherpa canary card from the checked-in template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def deterministic_seed(base_seed: int, slice_index: int, chunk_index: int) -> int:
    if slice_index < 0 or chunk_index < 0:
        raise ValueError("slice_index and chunk_index must be non-negative")
    return base_seed + 10_000 * slice_index + chunk_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pt-low", type=int, required=True)
    parser.add_argument("--pt-high", type=int, required=True)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=310_000_000)
    parser.add_argument("--slice-index", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--shower", choices=("Dire", "CSS"), default="Dire")
    args = parser.parse_args()

    if args.pt_low >= args.pt_high:
        raise SystemExit("pt-low must be smaller than pt-high")
    if args.events <= 0:
        raise SystemExit("events must be positive")

    seed = deterministic_seed(args.base_seed, args.slice_index, args.chunk_index)
    rendered = args.template.read_text()
    replacements = {
        "__NEVENTS__": str(args.events),
        "__SEED__": str(seed),
        "__SHOWER__": args.shower,
        "__PTLOW__": str(args.pt_low),
        "__PTHIGH__": str(args.pt_high),
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    if "__" in rendered:
        raise SystemExit("unresolved template placeholder remains")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    print(json.dumps({"card": str(args.output), "seed": seed, "sha256": digest}))


if __name__ == "__main__":
    main()
