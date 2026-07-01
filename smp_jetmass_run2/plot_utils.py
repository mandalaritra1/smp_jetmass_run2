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
    "met_phi": ("phi", r"$\phi$(MET)"),
    "met_pt":  ("pt",  r"MET $p_T$ [GeV]"),
}


def plot_met(out, var="met_phi", era="2018", data=False, dataset=None,
             systematic="nominal", density=True):
    """Single-source MET pt/phi (validation mode). Shades the HEM phi band for
    ``met_phi``. ``out`` is one loaded pickle (data or MC)."""
    if var not in _MET_VARS:
        raise ValueError(f"var must be one of {list(_MET_VARS)}, got {var!r}")
    if var not in out:
        raise KeyError(f"Output is missing {var!r}. Re-run the processor in 'validation' mode.")
    axis_name, xlabel = _MET_VARS[var]

    plt.figure()
    hplot.setup(era=era)
    hplot.set_plot_name(f"{var}_hem")

    h = _select_if_present(out[var], dataset=dataset, systematic=systematic).project(axis_name)
    h.plot(histtype="fill" if not data else "errorbar", density=density,
           label="Data" if data else "MC")

    if var == "met_phi":
        plt.axvspan(_HEM_PHI_MIN, _HEM_PHI_MAX, color="red", alpha=0.15,
                    label="HEM sector")

    hplot.quick_label(
        data=data,
        xlabel=xlabel,
        ylabel="Normalized events" if density else "# Events",
        cms_text="Simulation Internal" if not data else "Preliminary",
    )
    plt.legend()
    hplot.show()


def plot_met_data_mc(data_out, mc_out, var="met_phi", era="2018",
                     data_dataset=None, systematic="nominal", density=True):
    """Overlay 2018 data vs MC MET pt/phi to show the HEM implementation is
    correct: with HEM applied, the two agree across the HEM phi sector.

    ``data_out``/``mc_out`` are the loaded data and MC pickles.
    ``data_dataset`` selects the data dataset(s) (defaults to ``datasets[era]``).
    """
    if var not in _MET_VARS:
        raise ValueError(f"var must be one of {list(_MET_VARS)}, got {var!r}")
    for name, o in (("data", data_out), ("MC", mc_out)):
        if var not in o:
            raise KeyError(f"{name} output is missing {var!r}. Re-run in 'validation' mode.")
    axis_name, xlabel = _MET_VARS[var]
    if data_dataset is None:
        data_dataset = datasets.get(era)

    plt.figure()
    hplot.setup(era=era)
    hplot.set_plot_name(f"{var}_data_mc_hem")

    h_mc = _select_if_present(mc_out[var], systematic=systematic).project(axis_name)
    h_data = _select_if_present(data_out[var], dataset=data_dataset,
                                systematic=systematic).project(axis_name)

    h_mc.plot(histtype="fill", density=density, alpha=0.6, label="MC")
    h_data.plot(histtype="errorbar", density=density, color="black", label="Data")

    if var == "met_phi":
        plt.axvspan(_HEM_PHI_MIN, _HEM_PHI_MAX, color="red", alpha=0.15,
                    label="HEM sector")

    hplot.quick_label(
        data=True,
        xlabel=xlabel,
        ylabel="Normalized events" if density else "# Events",
        cms_text="Preliminary",
    )
    plt.legend()
    hplot.show()
