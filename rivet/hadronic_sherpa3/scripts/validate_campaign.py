#!/usr/bin/env python3
"""Check a set of completed chunks against the campaign gates.

Run this before harvesting.  It answers the questions that make the difference
between a combinable set of chunks and a silently wrong one: did every chunk
finish, did the selection gate pass, did chunks of a configuration share an
integration grid, and do their cross sections agree.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Sherpa re-estimates the cross section per sample, so chunks on one grid
# scatter within their quoted errors (10.6% between two 200-event chunks,
# under 1 sigma).  Test the pull, not equality.
MAX_XSEC_PULL = 5.0


def check(report: dict, name: str, ok: bool, detail: str = "") -> bool:
    report["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
    if not ok:
        report["failures"].append(name)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--expected-chunks", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict = {"checks": [], "failures": [], "configurations": {}}
    manifests = []
    for d in args.run_dirs:
        path = d / "completion_manifest.json"
        if not path.is_file():
            check(report, f"manifest-present:{d.name}", False, str(path))
            continue
        manifests.append(json.loads(path.read_text()))

    check(report, "any-manifests", bool(manifests), f"{len(manifests)} found")
    if not manifests:
        emit(report, args.output)

    if args.expected_chunks is not None:
        check(
            report,
            "chunk-count",
            len(manifests) == args.expected_chunks,
            f"{len(manifests)} of {args.expected_chunks}",
        )

    seeds = [m.get("seed") for m in manifests]
    check(report, "seeds-unique", len(set(seeds)) == len(seeds),
          f"{len(set(seeds))} unique of {len(seeds)}")

    for m in manifests:
        tag = f"s{m['slice_index']}_q{m['qcut_GeV']:g}_c{m.get('chunk_index')}"
        check(report, f"status-complete:{tag}", m.get("status") == "complete",
              str(m.get("status")))
        sel = m.get("selection_validation", {})
        check(report, f"selection-gate:{tag}", sel.get("status") == "pass",
              json.dumps(sel))
        check(
            report,
            f"events-generated:{tag}",
            m.get("generated_events") == m.get("requested_events"),
            f"{m.get('generated_events')} of {m.get('requested_events')}",
        )
        xs = m.get("cross_section_pb")
        check(report, f"xsec-finite:{tag}",
              isinstance(xs, (int, float)) and xs > 0, str(xs))

    # Chunks may only be combined within one (slice, Qcut, shower); each such
    # configuration is one physical cross section built on one grid.
    groups = defaultdict(list)
    for m in manifests:
        groups[(m["slice_index"], m["qcut_GeV"], m["shower"])].append(m)

    for key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        tag = f"s{key[0]}_q{key[1]:g}_{key[2]}"
        grids = {m.get("grid_in_sha256") for m in group}
        check(report, f"one-grid:{tag}", len(grids) == 1,
              f"{len(grids)} distinct grid hashes: {sorted(map(str, grids))}")

        xs = [m["cross_section_pb"] for m in group]
        n = [m.get("generated_events") or 0 for m in group]
        errs = [m.get("cross_section_error_pb") or 0.0 for m in group]
        mean = sum(x * c for x, c in zip(xs, n)) / sum(n) if sum(n) else 0.0
        spread = max(abs(x - mean) for x in xs) / mean if mean else float("inf")
        pulls = [abs(x - mean) / e for x, e in zip(xs, errs) if e > 0]
        max_pull = max(pulls) if pulls else None
        check(
            report, f"xsec-consistent:{tag}",
            max_pull is None or max_pull <= MAX_XSEC_PULL,
            f"max pull {max_pull if max_pull is None else round(max_pull, 2)} "
            f"sigma, relative spread {spread:.3g}",
        )

        report["configurations"][tag] = {
            "chunks": len(group),
            "generated_events": sum(m.get("generated_events", 0) for m in group),
            "cross_section_pb": mean,
            "cross_section_relative_spread": spread,
            "cross_section_max_pull": max_pull,
            "grid_in_sha256": sorted(map(str, grids)),
        }

    emit(report, args.output)


def emit(report: dict, output: Path | None) -> None:
    report["status"] = "pass" if not report["failures"] else "fail"
    payload = json.dumps(report, indent=2) + "\n"
    if output:
        output.write_text(payload)
    # Keep stdout readable: the full check list goes to the file.
    print(json.dumps(
        {k: report[k] for k in ("status", "failures", "configurations")},
        indent=2,
    ))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
