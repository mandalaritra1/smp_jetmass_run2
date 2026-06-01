#!/usr/bin/env python
"""Headless runner for all three channels (zjet / dijet / trijet).

Reads a JSON config (see configs/*.json for examples), builds the fileset for
the requested channel/dataset/era, runs the matching processor via
notebook_utils.run_from_config, and writes pickle outputs under `outputs/`.

    python scripts/run_analysis_cli.py --config configs/dijet_pythia_2018.json
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import smp_jetmass_run2.notebook_utils as nbutils  # noqa: E402

_flush_print = functools.partial(print, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to JSON config (see configs/*.json).",
    )
    args = parser.parse_args(argv)

    cfg = nbutils.validate_analysis_config(json.loads(args.config.read_text()))
    outputs, _ = nbutils.run_from_config(cfg, repo_root=REPO_ROOT, log=_flush_print)

    print("---OUTPUT-FILES---", flush=True)
    for path in outputs:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
