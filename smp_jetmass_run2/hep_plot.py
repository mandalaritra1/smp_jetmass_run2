# hep_plot.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplhep as hep
from cycler import cycler

# ----------------------------
# User-configurable globals
# ----------------------------
ERA = "UL2018"
OUTDIR = Path("outputs/plots")
DEFAULT_FORMATS = ("pdf",)

CMS_COLORS = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]

CMS_LUMI_RUN2 = {
    "2016APV": 19.52,
    "2016": 16.81,
    "2017": 41.48,
    "2018": 59.83,
    "all" : 137.6
}

#CMS_LUMI_RUN2["all"] = sum(CMS_LUMI_RUN2.values())

# Internal state
_INITIALIZED = False
_CURRENT_PLOT_NAME: Optional[str] = None


def setup(
    era: str = "UL2018",
    outdir: str | Path = "outputs/plots",
    formats: tuple[str, ...] = ("pdf",),
    use_mplhep: bool = True,
) -> None:
    """
    Set CMS-like plotting style, color cycle, and default output settings.
    Call once near the top of the notebook.
    """
    global ERA, OUTDIR, DEFAULT_FORMATS, _INITIALIZED

    ERA = era
    OUTDIR = Path(outdir)
    DEFAULT_FORMATS = formats

    if use_mplhep:
        plt.style.use(hep.style.CMS)

    plt.rcParams["axes.prop_cycle"] = cycler(color=CMS_COLORS)

    # Optional nice defaults
    # plt.rcParams["figure.dpi"] = 110
    # plt.rcParams["savefig.bbox"] = "tight"
    # plt.rcParams["axes.labelsize"] = 14
    # plt.rcParams["axes.titlesize"] = 16
    # plt.rcParams["legend.fontsize"] = 12

    OUTDIR.mkdir(parents=True, exist_ok=True)
    _INITIALIZED = True


def set_plot_name(name: str) -> None:
    """
    Set a temporary plot name for the next save/show call.
    """
    global _CURRENT_PLOT_NAME
    _CURRENT_PLOT_NAME = _sanitize_name(name)


def save(
    plot_name: Optional[str] = None,
    era: Optional[str] = None,
    formats: Optional[tuple[str, ...]] = None,
    fig=None,
    close: bool = False,
) -> None:
    """
    Save the current figure (or provided figure) as:
        {plot_name}_{era}.{ext}

    Example:
        save("muon_pt")
    """
    _ensure_setup()

    if fig is None:
        fig = plt.gcf()

    name = plot_name or _CURRENT_PLOT_NAME
    if not name:
        name = input("Plot name: ").strip() or "plot"

    name = _sanitize_name(name)
    tag = era or ERA
    exts = formats or DEFAULT_FORMATS

    for ext in exts:
        path = OUTDIR / f"{name}_{tag}.{ext}"
        fig.savefig(path)
        print(f"Saved -> {path}")

    _clear_plot_name()

    if close:
        plt.close(fig)


def show(
    plot_name: Optional[str] = None,
    era: Optional[str] = None,
    formats: Optional[tuple[str, ...]] = None,
    fig=None,
    close: bool = False,
) -> None:
    """
    Save first, then show.
    Handy replacement for plain plt.show().
    """
    if fig is None:
        fig = plt.gcf()

    save(plot_name=plot_name, era=era, formats=formats, fig=fig, close=False)
    plt.show()

    if close:
        plt.close(fig)


def savefig(*args, **kwargs) -> None:
    """
    Alias for save(), so it feels familiar.
    """
    save(*args, **kwargs)


def quick_label(
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    cms_text: str = "Preliminary",
    lumi: Optional[float] = None,
    com: int = 13,
    data: bool = False,
    loc: int = 0,
) -> None:
    """
    Convenience helper for labels and CMS text.
    Automatically uses CMS Run2 lumi based on ERA.
    """

    ax = plt.gca()

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    # If lumi not given, determine from ERA
    if lumi is None:
        era_key = str(ERA).replace("UL", "")  # allow UL2018, etc.
        if era_key in CMS_LUMI_RUN2:
            lumi = CMS_LUMI_RUN2[era_key]
        elif era_key.lower() == "all":
            lumi = CMS_LUMI_RUN2["all"]

    if lumi is not None:
        hep.cms.label(
            cms_text,
            data=data,
            year = ERA,
            lumi=lumi,
            com=com,
            loc=loc,
            ax=ax,
            fontsize = 20
        )
    else:
        hep.cms.text(cms_text, loc=loc, ax=ax)


def _sanitize_name(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _clear_plot_name() -> None:
    global _CURRENT_PLOT_NAME
    _CURRENT_PLOT_NAME = None


def _ensure_setup() -> None:
    if not _INITIALIZED:
        setup()