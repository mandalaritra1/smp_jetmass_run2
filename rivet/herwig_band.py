#!/usr/bin/env python3
"""Pythia<->Herwig alternate-generator band vs the Pythia-internal envelope.

Herwig (MG-LO + Herwig, cluster hadronisation) is an ALTERNATIVE estimate of the
combined shower+hadronisation modelling -- it is NOT an independent source to add in
quadrature to Vincia/CR/Lund. The right question is whether the Pythia-internal
envelope (Total = sqrt(CR^2+Had^2+Shower^2)) COVERS the Pythia->Herwig difference.

Per obs/pT bin/bin:
  herwig_dev = |norm_herwig/norm_pythia - 1|, with MC-stat error (conservative,
               independent) and noise-subtracted value.
Compared against the internal Total read from out/model_unc_<PREFIX>/model_uncertainty.csv.

Outputs out/overlay_<PREFIX>/herwig_vs_internal_<obs>.png and herwig_band.csv.
Run: ~/Projects/GluonJetMass/.venv/bin/python herwig_band.py [hi]
"""
import os, re, csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "hi"
YODA = lambda name: os.path.join(HERE, "out", f"{PREFIX}_{name}.yoda")
OUTDIR = os.path.join(HERE, "out", f"overlay_{PREFIX}")
INTERNAL_CSV = os.path.join(HERE, "out", "model_unc" if PREFIX == "mglo" else f"model_unc_{PREFIX}",
                            "model_uncertainty.csv")
FAMILIES = {"rho_g": r"$\rho_g$", "mass_g": r"$m_g$ [GeV]",
            "rho_u": r"$\rho_u$", "mass_u": r"$m_u$ [GeV]"}
PT_BINS = ["pt200_290", "pt290_400", "pt400_Inf"]


def parse(path):
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = re.match(r"BEGIN YODA_ESTIMATE1D_V3 (\S+)", lines[i])
        if not m:
            i += 1; continue
        hp = m.group(1); edges = None; vals = []; errs = []; tab = False; i += 1
        while i < n and not lines[i].startswith("END YODA_ESTIMATE1D_V3"):
            ln = lines[i]
            if ln.startswith("Edges(A1):"):
                edges = [float(x) for x in re.findall(r"[-+0-9.eE]+", ln.split(":", 1)[1])]
            elif ln.startswith("# value"):
                tab = True
            elif tab:
                tok = ln.split()
                if tok:
                    if tok[0] == "nan":
                        vals.append(np.nan); errs.append(np.nan)
                    else:
                        vals.append(float(tok[0]))
                        errs.append(abs(float(tok[2])) if len(tok) > 2 and tok[2] != "---" else 0.0)
            i += 1
        if hp.startswith("/CMS_ZJET_JETMASS/") and edges is not None and len(vals) >= 2:
            v = np.nan_to_num(np.array(vals[1:-1])); e = np.nan_to_num(np.array(errs[1:-1]))
            ed = np.array(edges)
            if len(v) == len(ed) - 1:
                out[hp.split("/")[-1]] = (ed, v, e)
        i += 1
    return out


def norm(ed, v, e):
    w = np.diff(ed); S = np.sum(v * w)
    if S <= 0:
        return v * 0.0, np.zeros_like(v), np.zeros_like(v)
    nn = v / S
    rel = np.where(v > 0, e / v, 0.0)
    return nn, nn * rel, rel


def load_internal_total():
    tot = {}
    if not os.path.exists(INTERNAL_CSV):
        return tot
    with open(INTERNAL_CSV) as fh:
        r = csv.DictReader(fh)
        for row in r:
            tot[(row["observable"], row["pt_bin"], row["bin_lo"], row["bin_hi"])] = float(row["Total"])
    return tot


def main():
    py = parse(YODA("pythia")); hw = parse(YODA("herwig"))
    if not hw:
        raise SystemExit(f"no Herwig yoda at {YODA('herwig')} -- stitch it first")
    internal = load_internal_total()
    rows = [["observable", "pt_bin", "bin_lo", "bin_hi",
             "herwig_raw_%", "herwig_stat_%", "herwig_nsub_%", "internal_total_%"]]

    for fam in FAMILIES:
        fig, axes = plt.subplots(1, len(PT_BINS), figsize=(14, 3.8), sharey=False)
        drew = False
        for ax, ptb in zip(axes, PT_BINS):
            key = f"{fam}_{ptb}"
            if key not in py or key not in hw:
                ax.set_visible(False); continue
            ed, vp, ep = py[key]
            np_, _, relp = norm(ed, vp, ep)
            nh, _, relh = norm(*hw[key])
            base = np.where(np_ > 0, np_, np.nan)
            dev = np.abs(nh - np_) / base
            ratio = np.where(np_ > 0, nh / np_, 0.0)
            sdev = ratio * np.sqrt(relh**2 + relp**2)
            dev = np.nan_to_num(dev); sdev = np.nan_to_num(sdev)
            nsub = np.sqrt(np.maximum(dev**2 - sdev**2, 0.0))
            ctr = 0.5 * (ed[:-1] + ed[1:])
            itot = np.array([internal.get((fam, ptb, f"{ed[b]:.4g}", f"{ed[b+1]:.4g}"), np.nan)
                             for b in range(len(ctr))]) * 100.0

            ax.step(ctr, 100 * dev, where="mid", color="magenta", lw=1.8, label="Pythia↔Herwig")
            ax.errorbar(ctr, 100 * dev, yerr=100 * sdev, fmt="none", ecolor="magenta",
                        elinewidth=0.7, capsize=1.5, alpha=0.7)
            ax.step(ctr, itot, where="mid", color="black", lw=1.8, label="Pythia-internal Total")
            ax.set_title(ptb.replace("pt", "pT ").replace("_", "–") + " GeV", fontsize=9)
            ax.set_xlabel(FAMILIES[fam], fontsize=10); ax.grid(alpha=0.3)
            ax.set_ylim(0, min(80, np.nanmax(np.concatenate([100*dev, itot])) * 1.15 + 5))
            drew = True
            for b in range(len(ctr)):
                rows.append([fam, ptb, f"{ed[b]:.4g}", f"{ed[b+1]:.4g}",
                             f"{100*dev[b]:.2f}", f"{100*sdev[b]:.2f}", f"{100*nsub[b]:.2f}",
                             f"{itot[b]:.2f}" if not np.isnan(itot[b]) else ""])
        if not drew:
            plt.close(fig); continue
        axes[0].set_ylabel("fractional [%]"); axes[0].legend(fontsize=8, loc="upper center")
        fig.suptitle(f"Alternate generator vs internal envelope — {fam}  "
                     f"(Pythia↔Herwig band vs sqrt(CR²+Had²+Shower²), 50k)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(OUTDIR, f"herwig_vs_internal_{fam}.png"), dpi=130)
        plt.close(fig)
        print("wrote", os.path.join(OUTDIR, f"herwig_vs_internal_{fam}.png"))

    with open(os.path.join(OUTDIR, "herwig_band.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("wrote", os.path.join(OUTDIR, "herwig_band.csv"))


if __name__ == "__main__":
    main()
