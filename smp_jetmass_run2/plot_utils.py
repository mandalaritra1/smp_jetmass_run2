import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import hist
from . import hep_plot as hplot
import matplotlib.patches as patches

HT_BINS = [
    "HT-100to200",
    "HT-200to400",
    "HT-400to600",
    "HT-600to800",
    "HT-800to1200",
    "HT-1200to2500",
    "HT-2500toInf",
]

ERAS = ["2016APV", "2016", "2017", "2018"]

datasets = {
    "2016": [
        "SingleElectron_UL2016",
        "SingleMuon_UL2016",
    ],
    "2016APV": [
        "SingleElectron_UL2016APV",
        "SingleMuon_UL2016APV",
    ],
    "2017": [
        "SingleElectron_UL2017",
        "SingleMuon_UL2017",
    ],
    "2018": [
        "SingleMuon_UL2018",
        "EGamma_UL2018",
    ],
}

def load_out(era, template="outputs/ht_validation_pythia_{era}.pkl"):
    with open(template.format(era=era), "rb") as f:
        return pkl.load(f)


def plot_ht(era, template="outputs/ht_validation_pythia_{era}.pkl"):
    out = load_out(era, template)

    hplot.setup(era=era)
    hplot.quick_label(
        data=False,
        xlabel=r"$H_T$ [GeV]",
        ylabel=r"#Events",
    )
    hplot.set_plot_name("ht")

    out["ht"].project("ht_bin", "pt")[HT_BINS, hist.loc(0):].plot(
        stack=False,
        histtype="fill",
    )

    plt.yscale("log")
    plt.xlim(100, 4000)
    plt.legend()
    hplot.show()


def plot_ht_all_eras(template="outputs/ht_validation_pythia_{era}.pkl"):
    for era in ERAS:
        plt.figure()
        plot_ht(era, template=template)

def plot_etaphijet_data(era, template = "outputs/etaphijet_validation_data.pkl"):
    with open(template, "rb") as f:
        out = pkl.load(f)

    hplot.setup(era = era)
    hplot.set_plot_name("etaphi_jet0")
    hplot.quick_label(data = True)




    out["eta_phi_jet_reco"][datasets[era], ...].project('eta', 'phi').plot2d(norm = 'log')

    if era == '2018':
        # HEM veto region (approximate CMS convention)
        hem_eta_min, hem_eta_max = -3.0, -1.3
        hem_phi_min, hem_phi_max = -1.57, -0.87
        
        ax = plt.gca()
        
        # draw rectangle around HEM veto area
        hem_box = patches.Rectangle(
            (hem_eta_min, hem_phi_min),                  # bottom-left corner
            hem_eta_max - hem_eta_min,                   # width in eta
            hem_phi_max - hem_phi_min,                   # height in phi
            linewidth=2.5,
            edgecolor='red',
            facecolor='none',
            linestyle='--'
        )
        
        ax.add_patch(hem_box)
    plt.ylim(-3.2, 3.2)
    cbar = plt.gcf().axes[-1]   # last axis is the colorbar
    cbar.set_ylabel("# Events")
    hplot.show()


def plot_etaphijet_data_all(template="outputs/etaphijet_validation_data.pkl"):
    for era in ERAS:
        try:
            print(f"Plotting era {era}")
            plt.figure()
            plot_etaphijet_data(era, template=template)
        except Exception as e:
            print(f"Skipping {era}: {e}")


def plot_etaphijet_mc(era, template = "outputs/etaphijet_validation_pythia_{era}.pkl" ):
    out = load_out(era, template)
    hplot.setup(era = era)
    hplot.set_plot_name("etaphi_jet0_mc")
    hplot.quick_label(data = False)

    out["eta_phi_jet_reco"].project('eta', 'phi').plot2d(norm = 'log')
    plt.ylim(-3.2, 3.2)
    cbar = plt.gcf().axes[-1]   # last axis is the colorbar
    cbar.set_ylabel("# Events")
    hplot.show()

def plot_etaphijet_mc_all(template = "outputs/etaphijet_validation_pythia_{era}.pkl" ):
    for era in ERAS:
        try:
            print(f"Plotting era {era}")
            plt.figure()
            plot_etaphijet_mc(era, template=template)
        except Exception as e:
            print(f"Skipping {era}: {e}")


def _select_if_present(h, **selectors):
    selection = {}
    axis_names = set(h.axes.name)
    for axis_name, value in selectors.items():
        if value is not None and axis_name in axis_names:
            selection[axis_name] = value
    if not selection:
        return h
    return h[selection]


def plot_raw_mass_groomed_vs_ungroomed(
    out,
    dataset=None,
    channel=None,
    systematic="nominal",
    era="2018",
    data=False,
    hist_name="m_g_vs_m_u_raw_reco",
):
    if hist_name not in out:
        raise KeyError(f"Output is missing '{hist_name}'. Re-run with the raw mass diagnostics.")

    hplot.setup(era=era)
    hplot.set_plot_name("raw_groomed_vs_ungroomed_mass")

    h = _select_if_present(
        out[hist_name],
        dataset=dataset,
        channel=channel,
        systematic=systematic,
    ).project("m_u_reco", "m_g_reco")

    h.plot2d(norm="log")
    ax = plt.gca()
    ax.plot([0, 200], [0, 200], color="white", linestyle="--", linewidth=1.0)
    hplot.quick_label(
        data=data,
        xlabel=r"Raw ungroomed jet mass [GeV]",
        ylabel=r"Raw groomed jet mass from subjets [GeV]",
        cms_text="Simulation Internal" if not data else "Preliminary",
    )
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("# Events")
    hplot.show()


def plot_reco_jet_ntuple_raw_mass_map(
    out,
    era="2018",
    data=False,
    bins=40,
    mass_range=(0, 200),
):
    if "reco_jet_ntuple" not in out:
        raise KeyError("Output is missing 'reco_jet_ntuple'. Run mass_diagnostic_ntuple mode.")

    hplot.setup(era=era)
    hplot.set_plot_name("ntuple_raw_groomed_vs_ungroomed_mass")

    ntuple = out["reco_jet_ntuple"]
    ungroomed = ntuple["mass_raw"].value
    groomed = ntuple["msoftdrop_raw"].value
    weight = ntuple["weight"].value

    fig, ax = plt.subplots()
    counts, xedges, yedges, image = ax.hist2d(
        ungroomed,
        groomed,
        bins=bins,
        range=[mass_range, mass_range],
        weights=weight,
        norm=LogNorm(),
    )
    ax.plot(mass_range, mass_range, color="white", linestyle="--", linewidth=1.0)
    hplot.quick_label(
        data=data,
        xlabel=r"Raw ungroomed jet mass [GeV]",
        ylabel=r"Raw groomed jet mass from subjets [GeV]",
        cms_text="Simulation Internal" if not data else "Preliminary",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("# Events")
    hplot.show(fig=fig)


def plot_raw_mass_overlay(
    out,
    dataset=None,
    channel=None,
    systematic="nominal",
    era="2018",
    data=False,
    density=True,
    hist_name="m_g_vs_m_u_raw_reco",
):
    if hist_name not in out:
        raise KeyError(f"Output is missing '{hist_name}'. Re-run with the raw mass diagnostics.")

    hplot.setup(era=era)
    hplot.set_plot_name("raw_mass_groomed_ungroomed_overlay")

    h = _select_if_present(
        out[hist_name],
        dataset=dataset,
        channel=channel,
        systematic=systematic,
    )

    h.project("m_u_reco").plot(label="Ungroomed raw", density=density)
    h.project("m_g_reco").plot(label="Groomed raw", density=density)
    hplot.quick_label(
        data=data,
        xlabel=r"Raw jet mass [GeV]",
        ylabel="Normalized events" if density else "# Events",
        cms_text="Simulation Internal" if not data else "Preliminary",
    )
    plt.legend()
    hplot.show()


# HEM sector in phi (approximate CMS convention): the failing HCAL endcap minus
# region spans phi in [-1.57, -0.87]. Lost energy there fakes MET pointing away
# from the hole, i.e. an excess near phi ~ +1.2. If the 2018 HEM treatment
# (flat lumi-fraction weight for MC, hard veto for data) is applied correctly,
# data and MC MET(phi) agree and no unmodelled excess remains.
_HEM_PHI_MIN, _HEM_PHI_MAX = -1.57, -0.87

_MET_VARS = {
    "met_phi":         ("phi", r"$\phi$(MET)"),
    "met_pt":          ("pt",  r"MET $p_T$ [GeV]"),
    "met_phi_xy":      ("phi", r"$\phi$(MET), xy-corrected"),
    "met_pt_xy":       ("pt",  r"MET $p_T$ [GeV], xy-corrected"),
    "met_phi_ak4veto": ("phi", r"$\phi$(MET), AK4-HEM veto"),
    "met_pt_ak4veto":  ("pt",  r"MET $p_T$ [GeV], AK4-HEM veto"),
}

# phi variables that should shade the HEM sector band
_PHI_HEM_VARS = {"met_phi", "met_phi_xy", "met_phi_ak4veto"}


def _density_max(h):
    """Peak height of ``h`` as drawn with ``density=True`` (unit-area normalized),
    so we can size the y-axis headroom without clipping."""
    import numpy as np

    vals = h.values()
    widths = np.diff(h.axes[0].edges)
    integral = (vals * widths).sum()
    if integral <= 0:
        return float(vals.max()) if len(vals) else 1.0
    return float((vals / integral).max())


def plot_met(out, var="met_phi", era="2018", data=False, dataset=None,
             systematic="nominal", density=True):
    """Single-source MET pt/phi (validation mode). Shades the HEM phi band for
    ``met_phi``. ``out`` is one loaded pickle (data or MC)."""
    if var not in _MET_VARS:
        raise ValueError(f"var must be one of {list(_MET_VARS)}, got {var!r}")
    if var not in out:
        raise KeyError(f"Output is missing {var!r}. Re-run the processor in 'validation' mode.")
    axis_name, xlabel = _MET_VARS[var]

    hplot.setup(era=era)
    hplot.set_plot_name(f"{var}_hem")
    # constrained layout keeps the CMS<->lumi label spacing and stops the axis
    # labels being clipped; no forced figsize (CMS style drives the proportions).
    fig, ax = plt.subplots(layout="constrained")

    h = _select_if_present(out[var], dataset=dataset, systematic=systematic).project(axis_name)
    h.plot(ax=ax, histtype="errorbar" if data else "fill", density=density,
           color="black" if data else None, label="Data" if data else "MC")

    if var in _PHI_HEM_VARS:
        ax.axvspan(_HEM_PHI_MIN, _HEM_PHI_MAX, color="red", alpha=0.15,
                   label="HEM sector")

    # legend headroom: raise the top to ~1.5x the data peak, never clip (rules 5/6)
    ax.set_ylim(0, (_density_max(h) if density else float(h.values().max())) * 1.5)

    hplot.quick_label(
        data=data,
        xlabel=xlabel,
        ylabel="Normalized events" if density else "# Events",
        cms_text="Simulation Internal" if not data else "Preliminary",
    )
    ax.legend(loc="upper right", framealpha=0.0)
    hplot.show(fig=fig)


def _data_mc_ratio(h_data, h_mc, xlabel, era, plot_name,
                   density=True, ratio=True, hem_band=False,
                   label_data="Data", label_mc="MC", ratio_label="Data/MC",
                   cms_data=True):
    """Overlay two already-projected 1D hists (``h_data`` points, ``h_mc`` fill)
    with an optional ratio panel below (points stat error + reference stat band
    around 1). ``hem_band`` shades the HEM phi sector. Labels are overridable so
    the same layout serves e.g. a MET<50/>50 shape comparison. Shared by the MET,
    AK4 HEM, and mass-split validation plots."""
    import numpy as np

    hplot.setup(era=era)
    hplot.set_plot_name(plot_name)
    if ratio:
        fig, (ax, rax) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]},
            layout="constrained")
    else:
        fig, ax = plt.subplots(layout="constrained")
        rax = None

    h_mc.plot(ax=ax, histtype="fill", density=density, alpha=0.6, label=label_mc)
    h_data.plot(ax=ax, histtype="errorbar", density=density, color="black", label=label_data)

    if hem_band:
        ax.axvspan(_HEM_PHI_MIN, _HEM_PHI_MAX, color="red", alpha=0.15,
                   label="HEM sector")

    peak = max(_density_max(h_mc), _density_max(h_data)) if density \
        else max(float(h_mc.values().max()), float(h_data.values().max()))
    ax.set_ylim(0, peak * 1.5)

    # CMS label goes on the top (main) axes; xlabel only on the bottom-most axes
    plt.sca(ax)
    hplot.quick_label(
        data=cms_data,
        xlabel=None if ratio else xlabel,
        ylabel="Normalized events" if density else "# Events",
        cms_text="Preliminary" if cms_data else "Simulation Preliminary",
    )
    ax.legend(loc="upper right", framealpha=0.0)

    if rax is not None:
        ax.tick_params(labelbottom=False)  # hide top-panel x labels (shared x)
        ax.set_xlabel("")                   # drop the auto axis-name label from hist.plot
        edges = h_data.axes[0].edges
        centers = h_data.axes[0].centers
        widths = np.diff(edges)
        mc_v = h_mc.values().astype(float)
        dt_v = h_data.values().astype(float)
        dt_e = np.sqrt(h_data.variances())
        mc_e = np.sqrt(h_mc.variances())
        if density:
            mc_norm = (mc_v * widths).sum() or 1.0
            dt_norm = (dt_v * widths).sum() or 1.0
        else:
            mc_norm = dt_norm = 1.0
        mc_d = mc_v / mc_norm
        good = mc_d > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (dt_v / dt_norm) / mc_d          # data stat error on the points
            r_err = (dt_e / dt_norm) / mc_d
            mc_rel = np.where(good, mc_e / mc_v, 0.0)  # MC stat, as a band around 1
        # MC stat uncertainty: shaded step band centred on 1
        rax.stairs(1.0 + mc_rel, edges, baseline=1.0 - mc_rel, fill=True,
                   color="gray", alpha=0.30, label="MC stat")
        rax.errorbar(centers[good], r[good], yerr=r_err[good], fmt="o",
                     color="black", ms=4, lw=1, label=label_data)
        rax.axhline(1.0, color="gray", ls="--", lw=1)
        if hem_band:
            rax.axvspan(_HEM_PHI_MIN, _HEM_PHI_MAX, color="red", alpha=0.15)
        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel(ratio_label)
        rax.set_xlabel(xlabel)

    hplot.show(fig=fig)


def plot_met_data_mc(data_out, mc_out, var="met_phi", era="2018",
                     data_dataset=None, systematic="nominal", density=True,
                     ratio=True):
    """Overlay 2018 data vs MC MET pt/phi to show the HEM implementation is
    correct: with HEM applied, the two agree across the HEM phi sector.

    ``data_out``/``mc_out`` are the loaded data and MC pickles.
    ``data_dataset`` selects the data dataset(s) (defaults to ``datasets[era]``).
    ``ratio=True`` adds a Data/MC ratio panel below.
    """
    if var not in _MET_VARS:
        raise ValueError(f"var must be one of {list(_MET_VARS)}, got {var!r}")
    for name, o in (("data", data_out), ("MC", mc_out)):
        if var not in o:
            raise KeyError(f"{name} output is missing {var!r}. Re-run in 'validation' mode.")
    axis_name, xlabel = _MET_VARS[var]
    if data_dataset is None:
        data_dataset = datasets.get(era)
    # keep only data streams actually present (a single-channel run may have just
    # SingleMuon or just EGamma); selecting a missing name is a hard KeyError.
    if data_dataset is not None:
        present = set(data_out[var].axes["dataset"])
        data_dataset = [d for d in data_dataset if d in present] or None

    h_mc = _select_if_present(mc_out[var], systematic=systematic).project(axis_name)
    h_data = _select_if_present(data_out[var], dataset=data_dataset,
                                systematic=systematic).project(axis_name)
    _data_mc_ratio(h_data, h_mc, xlabel, era, f"{var}_data_mc_hem",
                   density=density, ratio=ratio,
                   hem_band=var in _PHI_HEM_VARS)


def plot_validation_data_mc(data_out, mc_out, hist_name, axis_name, xlabel,
                            era="2018", data_dataset=None, systematic="nominal",
                            density=True, ratio=True, hem_band=False, plot_name=None):
    """Generic data-vs-MC overlay + ratio for any 1D validation hist (used e.g.
    for the jet-mass before/after the AK4-HEM veto: hist_name='mass_jet0' vs
    'mass_jet0_ak4veto', axis_name='mass')."""
    for name, o in (("data", data_out), ("MC", mc_out)):
        if hist_name not in o:
            raise KeyError(f"{name} output is missing {hist_name!r}. Re-run in 'validation' mode.")
    if data_dataset is None:
        data_dataset = datasets.get(era)
    if data_dataset is not None:
        present = set(data_out[hist_name].axes["dataset"])
        data_dataset = [d for d in data_dataset if d in present] or None

    h_mc = _select_if_present(mc_out[hist_name], systematic=systematic).project(axis_name)
    h_data = _select_if_present(data_out[hist_name], dataset=data_dataset,
                                systematic=systematic).project(axis_name)
    _data_mc_ratio(h_data, h_mc, xlabel, era, plot_name or f"{hist_name}_data_mc",
                   density=density, ratio=ratio, hem_band=hem_band)


def plot_ak4_phi_hem(data_out, mc_out, era="2018", data_dataset=None,
                     density=True, ratio=True):
    """Data vs MC phi of AK4 jets that fall in the HEM eta strip
    (-3.0 < eta < -1.3), with a Data/MC ratio. Answers the ARC ask "how big is
    the AK4-in-HEM effect": in 2018 data the failing HEM sector
    (phi in [-1.57, -0.87]) loses jet energy, so AK4 jets there are suppressed
    -> a data deficit / ratio dip in the band that MC (no HEM) does not show.
    The AK8 measurement jet is HEM-vetoed, but AK4 jets are not, so this is the
    residual check requested."""
    var = "ak4_phi_hemeta"
    for name, o in (("data", data_out), ("MC", mc_out)):
        if var not in o:
            raise KeyError(f"{name} output is missing {var!r}. Re-run in 'validation' mode.")
    if data_dataset is None:
        data_dataset = datasets.get(era)
    if data_dataset is not None:
        present = set(data_out[var].axes["dataset"])
        data_dataset = [d for d in data_dataset if d in present] or None

    h_mc = _select_if_present(mc_out[var]).project("phi")
    h_data = _select_if_present(data_out[var], dataset=data_dataset).project("phi")
    _data_mc_ratio(h_data, h_mc, r"AK4 jet $\phi$ (HEM $\eta$ strip)", era,
                   "ak4_phi_hem_data_mc", density=density, ratio=ratio,
                   hem_band=True)


def plot_ak4_etaphi(out, era="2018", data=False, dataset=None):
    """2D eta-phi occupancy of AK4 jets (validation mode), with the HEM veto box
    overlaid for 2018. Data shows the depleted HEM sector; MC (no HEM) does not."""
    var = "ak4_eta_phi_reco"
    if var not in out:
        raise KeyError(f"Output is missing {var!r}. Re-run in 'validation' mode.")
    if dataset is None and data:
        dataset = datasets.get(era)
    if dataset is not None:
        present = set(out[var].axes["dataset"])
        dataset = [d for d in dataset if d in present] or None

    hplot.setup(era=era)
    hplot.set_plot_name("ak4_etaphi" + ("" if data else "_mc"))
    fig, ax = plt.subplots(layout="constrained")

    _select_if_present(out[var], dataset=dataset).project("eta", "phi").plot2d(
        ax=ax, norm="log")

    if era == "2018":
        # HEM veto box (approx CMS convention): eta in [-3.0, -1.3], phi in [-1.57, -0.87]
        ax.add_patch(patches.Rectangle(
            (-3.0, _HEM_PHI_MIN), 3.0 - 1.3, _HEM_PHI_MAX - _HEM_PHI_MIN,
            linewidth=2.5, edgecolor="red", facecolor="none", linestyle="--"))
    ax.set_xlabel(r"AK4 jet $\eta$")
    ax.set_ylabel(r"AK4 jet $\phi$")
    plt.sca(ax)
    hplot.quick_label(data=data, cms_text="Preliminary" if data else "Simulation Internal")
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("# Jets")
    hplot.show(fig=fig)


def plot_met_vs_jetpt(data_out, mc_out, era="2018", data_dataset=None,
                      systematic="nominal", hist_name="met_ptjet_reco"):
    """Profile of <MET> vs leading-jet pT, data vs MC. If the MET tail is
    jet-resolution driven, <MET> rises with jet pT and MC tracks it -- the
    justification that high-MET events are resolution, not anomalies."""
    import numpy as np
    for name, o in (("data", data_out), ("MC", mc_out)):
        if hist_name not in o:
            raise KeyError(f"{name} output is missing {hist_name!r}. Re-run in 'validation' mode.")
    if data_dataset is None:
        data_dataset = datasets.get(era)
    if data_dataset is not None:
        present = set(data_out[hist_name].axes["dataset"])
        data_dataset = [d for d in data_dataset if d in present] or None

    h_mc = _select_if_present(mc_out[hist_name], systematic=systematic).project("ptreco", "pt")
    h_data = _select_if_present(data_out[hist_name], dataset=data_dataset,
                                systematic=systematic).project("ptreco", "pt")
    met_c = h_mc.axes["pt"].centers
    ptj_c = h_mc.axes["ptreco"].centers

    def profile(h):
        v = h.values().astype(float)              # (nptreco, nmet)
        n = v.sum(axis=1)
        good = n > 0
        mean = np.where(good, (v * met_c).sum(axis=1) / np.where(good, n, 1), np.nan)
        var = np.where(good, (v * met_c ** 2).sum(axis=1) / np.where(good, n, 1) - mean ** 2, np.nan)
        err = np.sqrt(np.clip(var, 0, None) / np.where(good, n, 1))
        return mean, err, good

    m_mc, e_mc, g_mc = profile(h_mc)
    m_dt, e_dt, g_dt = profile(h_data)

    hplot.setup(era=era)
    hplot.set_plot_name("met_vs_jetpt")
    fig, ax = plt.subplots(layout="constrained")
    ax.errorbar(ptj_c[g_mc], m_mc[g_mc], yerr=e_mc[g_mc], fmt="s", color="C0",
                ms=5, lw=1.2, label="MC")
    ax.errorbar(ptj_c[g_dt], m_dt[g_dt], yerr=e_dt[g_dt], fmt="o", color="black",
                ms=5, lw=1.2, label="Data")
    ax.axhline(50.0, color="red", ls=":", lw=1)  # the MET>50 line under discussion
    ax.set_xlabel(r"Leading jet $p_T$ [GeV]")
    ax.set_ylabel(r"$\langle$MET$\rangle$ [GeV]")
    ax.set_ylim(bottom=0)
    plt.sca(ax)
    hplot.quick_label(data=True, cms_text="Preliminary")
    ax.legend(loc="upper left", framealpha=0.0)
    hplot.show(fig=fig)


def plot_mass_metsplit(out, era="2018", data=True, dataset=None, met_cut=50.0,
                       hist_name="mass_met_reco", systematic="nominal",
                       density=True, ratio=True):
    """Jet-mass shape for MET < cut vs MET > cut (same source). If high-MET
    events don't distort the observable the two overlap and the ratio is flat
    -- the justification for not cutting on MET."""
    if hist_name not in out:
        raise KeyError(f"Output is missing {hist_name!r}. Re-run in 'validation' mode.")
    if dataset is None and data:
        dataset = datasets.get(era)
    if dataset is not None:
        present = set(out[hist_name].axes["dataset"])
        dataset = [d for d in dataset if d in present] or None

    h = _select_if_present(out[hist_name], dataset=dataset, systematic=systematic)
    h_lo = h[{"pt": slice(None, hist.loc(met_cut))}].project("mass")
    h_hi = h[{"pt": slice(hist.loc(met_cut), None)}].project("mass")
    _data_mc_ratio(
        h_hi, h_lo, r"Ungroomed jet mass [GeV]", era,
        f"mass_metsplit_{int(met_cut)}", density=density, ratio=ratio,
        label_mc=f"MET < {int(met_cut)} GeV", label_data=f"MET > {int(met_cut)} GeV",
        ratio_label=f">{int(met_cut)} / <{int(met_cut)}", cms_data=data)
