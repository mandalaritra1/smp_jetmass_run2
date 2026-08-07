"""Provenance stamp for working plots (mandatory — see cms-plot-style skill).

Every working/internal figure gets a small bottom-right line:

    YYYY-MM-DD  |  smp_jetmass_run2 <git describe --tags --always --dirty>  |  inputs: <tag>

Strip it only for publication-final figures: pass enabled=False or set
SMP_NO_STAMP=1.
"""
from __future__ import annotations

import datetime
import os
import subprocess


def repo_version() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, cwd=here, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def stamp(fig, inputs: str | None = None, enabled: bool = True) -> None:
    """Add the provenance line to a matplotlib Figure (no-op when disabled)."""
    if not enabled or os.environ.get("SMP_NO_STAMP") == "1":
        return
    txt = f"{datetime.date.today().isoformat()}  |  smp_jetmass_run2 {repo_version()}"
    if inputs:
        txt += f"  |  inputs: {inputs}"
    #### reserve a strip below the axes so the stamp never sits on the x-axis
    #### label (constrained layout otherwise pulls the label to the figure edge)
    #### rect is (left, bottom, WIDTH, HEIGHT): bottom 0.03 + height 0.97 keeps
    #### the top edge at 1.0 -- height 1 would push the CMS label off-canvas
    try:
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    except Exception:
        pass
    fig.text(0.99, 0.002, txt, ha="right", va="bottom",
             fontsize=7, color="0.45", family="monospace")
