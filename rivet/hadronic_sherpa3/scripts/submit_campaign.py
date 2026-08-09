#!/usr/bin/env python3
"""Submit one campaign phase to LPC HTCondor, behind the approval gate.

Canary phases (``qcut_scan``) submit freely.  The 2.45 M-event production wave
does not: it requires ``full_production_approved`` in the campaign config AND
an explicit acknowledgement on the command line, because CAMPAIGN_PLAN.md
section 12 makes production conditional on a human review of the pinned
identity, the Dire card, selection parity, the Qcut plots, slice closure and
the measured runtime.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

CANARY_PHASES = {"qcut_scan"}
GATED_PHASES = {"production"}

REVIEW_TOKEN = "reviewed-canary-and-approve-production"


def ssh(host: str, command: str, check: bool = True) -> str:
    """Run a command on LPC through the login-shell proxy environment."""
    wrapped = f"zsh -ic {shlex.quote('ssh -o BatchMode=yes ' + host + ' ' + shlex.quote(command))}"
    result = subprocess.run(
        wrapped, shell=True, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise SystemExit(
            f"FATAL: remote command failed ({result.returncode}): {command}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--stage", required=True, choices=("integrate", "chunks"))
    parser.add_argument("--work-dir", required=True, help="remote work directory")
    parser.add_argument("--host", default="amandal2@cmslpc303.fnal.gov")
    parser.add_argument("--approve-production", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.campaign.read_text())

    if args.phase in GATED_PHASES:
        if not cfg.get("full_production_approved", False):
            raise SystemExit(
                "REFUSED: full_production_approved is false in "
                f"{args.campaign}.  Production is gated on the canary review "
                "in CAMPAIGN_PLAN.md section 12."
            )
        if args.approve_production != REVIEW_TOKEN:
            raise SystemExit(
                "REFUSED: production also needs --approve-production "
                f"{REVIEW_TOKEN}, confirming a human reviewed the canary "
                "outputs.  Submission alone is not completion."
            )
    elif args.phase not in CANARY_PHASES:
        raise SystemExit(f"REFUSED: unknown phase {args.phase!r}")

    # A missing proxy is the single most common submit failure here, and its
    # error message ("unable to read proxy file") does not say what to do.
    proxy = ssh(args.host, "voms-proxy-info -timeleft 2>/dev/null || echo 0", check=False)
    if not proxy.strip().isdigit() or int(proxy.strip()) < 3600:
        raise SystemExit(
            "FATAL: no usable grid proxy on LPC (needs > 1 h left).\n"
            "Run this yourself on cmslpc -- it needs your certificate "
            "passphrase:\n    voms-proxy-init -voms cms -valid 192:00"
        )

    submit_file = f"{args.stage}.sub"
    command = f"cd {shlex.quote(args.work_dir)} && condor_submit {submit_file}"
    if args.dry_run:
        print(f"[dry-run] would run: {command}")
        return
    print(ssh(args.host, command))


if __name__ == "__main__":
    main()
