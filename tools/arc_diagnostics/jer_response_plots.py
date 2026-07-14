"""AK8 jet-pT response distributions per analysis pT bin.

Shows the *actual* shape of R = pt_reco / pt_gen (gen-matched jets) so the
resolution is not assumed Gaussian. Each panel overlays a Gaussian with the same
median and robust width (half of the 16-84% interval) to expose the non-Gaussian
low-side tail. Emits one CMS-styled square panel per pT bin (linear + log-y).

Input: ~/Downloads/mass_diagnostic_ntuple_pythia_2018.pkl  (key reco_jet_ntuple)
Output: tools/arc_diagnostics/figs/jer_response_pt{1,2,3}.png
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.style.CMS)

D = os.path.expanduser("~/Downloads")
NT = "mass_diagnostic_ntuple_pythia_2018.pkl"
OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

EDGES = [200, 290, 400, 1e9]
LABELS = ["200 < $p_{T}^{gen}$ < 290 GeV",
          "290 < $p_{T}^{gen}$ < 400 GeV",
          "$p_{T}^{gen}$ > 400 GeV"]
TAGS = ["pt1", "pt2", "pt3"]


def load():
    o = pickle.load(open(os.path.join(D, NT), "rb"))["reco_jet_ntuple"]
    a = lambda k: np.asarray(o[k].value)
    pt, gpt, w = a("pt"), a("gen_pt"), a("weight")
    m = a("has_gen_match").astype(bool)
    sel = m & np.isfinite(gpt) & (gpt > 0) & np.isfinite(pt) & (pt > 0)
    return pt[sel] / gpt[sel], gpt[sel], w[sel]


def wquant(x, w, q):
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    cw = (np.cumsum(ws) - 0.5 * ws) / np.sum(ws)
    return np.interp(q, cw, xs)


def main():
    r, gpt, w = load()
    bins = np.linspace(0.6, 1.5, 91)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    for lo, hi, lab, tag in zip(EDGES[:-1], EDGES[1:], LABELS, TAGS):
        m = (gpt >= lo) & (gpt < hi)
        rr, ww = r[m], w[m]
        q16, q50, q84 = wquant(rr, ww, [0.16, 0.5, 0.84])
        sig = 0.5 * (q84 - q16)
        mean = np.average(rr, weights=ww)
        rms = np.sqrt(np.average((rr - mean) ** 2, weights=ww))
        # tail fractions outside median +/- 2 sigma_robust
        lo2, hi2 = q50 - 2 * sig, q50 + 2 * sig
        f_lo = ww[rr < lo2].sum() / ww.sum()
        f_hi = ww[rr > hi2].sum() / ww.sum()

        h, _ = np.histogram(rr, bins=bins, weights=ww, density=True)
        gauss = np.exp(-0.5 * ((ctr - q50) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

        for scale in ("lin", "log"):
            fig, ax = plt.subplots(layout="constrained")
            hep.cms.label("Preliminary", data=False, loc=0, ax=ax,
                          rlabel="2018 (13 TeV)")
            ax.grid(alpha=0.25)
            ax.stairs(h, bins, lw=2.2, color="black",
                      label="Response (Pythia8)")
            ax.plot(ctr, gauss, lw=2.0, ls="--", color="tab:red",
                    label="Gaussian (same median, robust $\\sigma$)")
            ax.axvline(1.0, color="0.5", lw=1.2, ls=":")
            ax.set_xlabel(r"$p_{T}^{\,reco}\,/\,p_{T}^{\,gen}$")
            ax.set_ylabel("Probability density")
            ax.set_xlim(0.6, 1.5)
            txt = (f"{lab}\n"
                   f"median = {q50:.3f}\n"
                   f"robust $\\sigma$ = {sig:.3f}  ({sig/q50*100:.1f}%)\n"
                   f"RMS/mean = {rms/mean*100:.1f}%\n"
                   f"tails >2$\\sigma$: {f_lo*100:.1f}% lo / {f_hi*100:.1f}% hi")
            if scale == "lin":
                ax.set_ylim(0, h.max() * 1.55)
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
                        ha="left", fontsize=16,
                        bbox=dict(fc="white", ec="0.7", alpha=0.85))
                ax.legend(loc="upper right", fontsize=15)
            else:
                pos = h[h > 0].min()
                ax.set_yscale("log")
                ax.set_ylim(pos * 0.5, h.max() * 5)
                ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
                        ha="left", fontsize=16,
                        bbox=dict(fc="white", ec="0.7", alpha=0.85))
                ax.legend(loc="upper right", fontsize=15)
            p = os.path.join(OUT, f"jer_response_{tag}_{scale}.png")
            fig.savefig(p, dpi=120)
            plt.close(fig)
            print("wrote", p)


if __name__ == "__main__":
    main()
