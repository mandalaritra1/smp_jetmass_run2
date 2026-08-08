"""Cross-channel histogram parity audit.

The three channel processors (zjet / dijet / trijet) must register the same
histograms in the modes they share: a hist added to one channel and forgotten
in the others is exactly the drift this test exists to catch. Intentionally
channel-specific hists are declared in CHANNEL_SPECIFIC below with a reason --
that dict doubles as the documentation of every allowed asymmetry.

For hists that ARE shared, the axis-name sets must also agree (ignoring the
structural `channel`/`jk` axes, which legitimately differ between the zjet and
hadronic layouts). Axis ORDER and binning are allowed to differ per channel.

Run standalone (no pytest needed):
    PYTHONPATH=. <venv>/bin/python tests/test_hist_parity.py
"""
import sys

import hist

from smp_jetmass_run2.zjet_processor import QJetMassProcessor
from smp_jetmass_run2.dijet_processor import DijetProcessor
from smp_jetmass_run2.trijet_processor import TrijetProcessor

PROCESSORS = {
    "zjet": QJetMassProcessor,
    "dijet": DijetProcessor,
    "trijet": TrijetProcessor,
}

# mode -> channels compared in that mode. zjet's validation/full are entirely
# different diagnostics (leptons/MET/HEM) and are not comparable to the
# hadronic ones, so those modes are only checked dijet <-> trijet.
# reweight_pythia_rho is not available for trijet (no Herwig trijet sample).
COMPARISONS = [
    ("minimal", ("zjet", "dijet", "trijet")),
    ("minimal_rho", ("zjet", "dijet", "trijet")),
    ("mass_jk", ("zjet", "dijet", "trijet")),
    ("rho_jk", ("zjet", "dijet", "trijet")),
    ("reweight_pythia_rho", ("zjet", "dijet")),
    ("validation", ("dijet", "trijet")),
    ("full", ("dijet", "trijet")),
]

# hist name -> (channels allowed to have it exclusively, reason).
# Consulted only when a hist is missing from some compared channel: the
# channels that DO have it must be a subset of the allowed set.
CHANNEL_SPECIFIC = {
    "ptjet_rhojet_g_reco_flav": (
        {"dijet", "trijet"},
        "hadronic q/g content per measurement bin (matched jets, partonFlavour "
        "categories); zjet port undecided",
    ),
    "ptjet_rhojet_g_reco_mfloor2": (
        {"dijet", "trijet"},
        "hadronic-only full-stat diagnostic for the historical m_g > 2 GeV floor",
    ),
    "ptjet_rhojet_g_gen_mfloor2": (
        {"dijet", "trijet"},
        "hadronic-only full-stat diagnostic for the historical m_g > 2 GeV floor",
    ),
    "response_matrix_rho_g_mfloor2": (
        {"dijet", "trijet"},
        "hadronic-only full-stat diagnostic for the historical m_g > 2 GeV floor",
    ),
    "reco_cov_rho_g_mfloor2": (
        {"dijet"},
        "dijet two-jet event covariance for the hadronic mass-floor diagnostic",
    ),
    "ptz_mz_reco": (
        {"zjet"},
        "Z pT/mass validation -- no Z in the hadronic channels",
    ),
    "m_g_over_m_u_reco": (
        {"zjet"},
        "groomed/ungroomed ratio diagnostic; zjet-only port candidate",
    ),
    "m_g_vs_m_u_reco": (
        {"zjet"},
        "groomed vs ungroomed diagnostic; zjet-only port candidate",
    ),
    "m_g_over_m_u_raw_reco": (
        {"zjet"},
        "raw-jet ratio diagnostic; zjet-only port candidate",
    ),
    "m_g_vs_m_u_raw_reco": (
        {"zjet"},
        "raw-jet 2D diagnostic; zjet-only port candidate",
    ),
    "m_u_jet_reco_over_gen": (
        {"zjet"},
        "rho_jk closure; hadronic channels fill it in validation/full",
    ),
    "m_g_jet_reco_over_gen": (
        {"zjet"},
        "rho_jk closure; hadronic channels fill it in validation/full",
    ),
    "ptreco_mreco_fine_u": (
        {"trijet"},
        "fine mass diagnostic inherited by trijet only",
    ),
    "ptreco_mreco_fine_g": (
        {"trijet"},
        "fine groomed-mass diagnostic inherited by trijet only",
    ),
    "ptjet_rhojet_u_reco_chan": (
        {"zjet"},
        "zjet-only data lepton-channel split; hadronic hists have no channel axis",
    ),
    "ptjet_rhojet_g_reco_chan": (
        {"zjet"},
        "zjet-only data lepton-channel split; hadronic hists have no channel axis",
    ),
    "reco_cov_rho_u": (
        {"dijet"},
        "two measured jets require event-clustered covariance; one-jet channels are diagonal",
    ),
    "reco_cov_rho_g": (
        {"dijet"},
        "two measured jets require event-clustered covariance; one-jet channels are diagonal",
    ),
}

# Axes that encode channel structure, not physics.
STRUCTURAL_AXES = {"channel", "jk"}


def collect(channel, mode, do_gen):
    """Registered hists of one processor: {name: frozenset(axis names)}."""
    proc = PROCESSORS[channel](do_gen=do_gen, mode=mode)
    return {
        name: frozenset(axis.name for axis in obj.axes)
        for name, obj in proc.hists.items()
        if isinstance(obj, hist.Hist)
    }


def audit():
    problems = []
    for mode, channels in COMPARISONS:
        for do_gen in (True, False):
            registered = {ch: collect(ch, mode, do_gen) for ch in channels}
            union = set().union(*registered.values())
            for name in sorted(union):
                have = {ch for ch in channels if name in registered[ch]}
                missing = set(channels) - have
                if missing:
                    allowed, _reason = CHANNEL_SPECIFIC.get(name, (set(), None))
                    if not have <= allowed:
                        problems.append(
                            f"[{mode}, do_gen={do_gen}] '{name}' registered in "
                            f"{sorted(have)} but missing in {sorted(missing)} -- "
                            "register it there too or document the asymmetry in "
                            f"CHANNEL_SPECIFIC in {__file__}."
                        )
                    continue
                axis_sets = {
                    ch: registered[ch][name] - STRUCTURAL_AXES for ch in channels
                }
                if len(set(axis_sets.values())) > 1:
                    detail = "; ".join(
                        f"{ch}: {sorted(axes)}" for ch, axes in axis_sets.items()
                    )
                    problems.append(
                        f"[{mode}, do_gen={do_gen}] '{name}' has diverging axis "
                        f"names across channels: {detail}"
                    )
    return problems


def test_hist_parity():
    problems = audit()
    assert not problems, (
        f"{len(problems)} cross-channel histogram parity violation(s):\n"
        + "\n".join(problems)
    )


if __name__ == "__main__":
    found = audit()
    if found:
        print(f"FAIL: {len(found)} parity violation(s)")
        for problem in found:
            print(" -", problem)
        sys.exit(1)
    print(
        "OK: histogram registration is in parity across channels "
        f"({len(COMPARISONS)} mode comparisons, do_gen both ways)."
    )
