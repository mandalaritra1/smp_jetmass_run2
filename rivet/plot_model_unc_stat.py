#!/usr/bin/env python3
"""Stat-aware modelling uncertainty -- 'uncertainty of the uncertainty'.

For each measured observable / pT bin, every variation is area-normalised (shape),
then the fractional deviation from nominal is computed WITH its MC statistical error:

    dev_i      = norm_member_i / norm_nominal_i - 1
    sig(dev)_i = (nm/nn) * sqrt( (e_mem/v_mem)^2 + (e_nom/v_nom)^2 )_i

The per-bin stat errors come from the yoda 'stats' ErrorLabels (errUp column).
NOTE: nominal & each variation shower the SAME LHE per slice, so the noise on the
*difference* is smaller than this independent propagation -> these bars are a
CONSERVATIVE upper bound. If |dev| exceeds its bar, the modelling effect is real.

Per source we report:
    env        = max_member |dev|                         (raw envelope, biased up by noise)
    env_staterr= sig(dev) of the argmax member            (its statistical error)
    env_nsub   = sqrt(max(env^2 - env_staterr^2, 0))      (noise-subtracted estimate)

Outputs out/model_unc_stat/: overlay PNGs (each variation's dev +/- stat band per
pT panel) for rho_g & mass_g, and model_unc_stat.csv with every number.
Run: ~/Projects/GluonJetMass/.venv/bin/python plot_model_unc_stat.py [mglo]
"""
import os, re, csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "mglo"
OUTDIR = os.path.join(HERE, "out", "model_unc_stat" if PREFIX == "mglo" else f"model_unc_stat_{PREFIX}")
YODA = lambda name: os.path.join(HERE, "out", f"{PREFIX}_{name}.yoda")
NOMINAL = "pythia"
SOURCES = {"CR": ["pythia_cr1", "pythia_cr2"],
           "Hadronization": ["pythia_fragsoft", "pythia_fraghard"],
           "Shower": ["pythia_vincia"]}
STYLE = {"pythia_cr1": ("CR1 (QCD)", "tab:blue"),
         "pythia_cr2": ("CR2 (move)", "tab:cyan"),
         "pythia_fragsoft": ("frag-soft", "tab:orange"),
         "pythia_fraghard": ("frag-hard", "tab:red"),
         "pythia_vincia": ("vincia", "tab:green")}
FAMILIES = {"rho_g": r"$\rho_g$", "mass_g": r"$m_g$ [GeV]",
            "rho_u": r"$\rho_u$", "mass_u": r"$m_u$ [GeV]"}
PT_BINS = ["pt200_290", "pt290_400", "pt400_Inf"]
PLOT_FAMS = ["rho_g", "mass_g"]


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
                        errs.append(abs(float(tok[2])) if len(tok) > 2 and tok[2] != "---" else np.nan)
            i += 1
        if hp.startswith("/CMS_ZJET_JETMASS/") and edges is not None and len(vals) >= 2:
            v = np.array(vals[1:-1]); e = np.array(errs[1:-1]); ed = np.array(edges)
            if len(v) == len(ed) - 1:
                out[hp.split("/")[-1]] = (ed, np.nan_to_num(v), np.nan_to_num(e))
        i += 1
    return out


def norm_shape(ed, v, e):
    w = np.diff(ed); S = np.sum(v * w)
    if S <= 0:
        return v * 0.0, np.zeros_like(v), np.zeros_like(v)
    n = v / S
    rel = np.where(v > 0, e / v, 0.0)
    return n, n * rel, rel  # normalized shape, abs stat err, per-bin rel stat err


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    nom = parse(YODA(NOMINAL))
    if not nom:
        raise SystemExit(f"no nominal in {YODA(NOMINAL)}")
    active = {s: [m for m in mem if os.path.exists(YODA(m))] for s, mem in SOURCES.items()}
    active = {s: m for s, m in active.items() if m}
    data = {m: parse(YODA(m)) for ms in active.values() for m in ms}
    print("active sources:", active)

    rows = [["observable", "pt_bin", "bin_lo", "bin_hi", "source",
             "env_raw_%", "env_staterr_%", "env_noisesub_%"]]

    for fam in FAMILIES:
        for ptb in PT_BINS:
            key = f"{fam}_{ptb}"
            if key not in nom:
                continue
            ed, vn, en = nom[key]
            nn, _, reln = norm_shape(ed, vn, en)
            base = np.where(nn > 0, nn, np.nan)
            for src, members in active.items():
                env = np.zeros_like(nn); env_se = np.zeros_like(nn)
                for m in members:
                    if key not in data[m]:
                        continue
                    nm, _, relm = norm_shape(*data[m][key])
                    dev = np.abs(nm - nn) / base
                    ratio = np.where(nn > 0, nm / nn, 0.0)
                    sdev = ratio * np.sqrt(relm**2 + reln**2)
                    dev = np.nan_to_num(dev); sdev = np.nan_to_num(sdev)
                    take = dev > env
                    env_se = np.where(take, sdev, env_se)
                    env = np.where(take, dev, env)
                nsub = np.sqrt(np.maximum(env**2 - env_se**2, 0.0))
                for b in range(len(nn)):
                    rows.append([fam, ptb, f"{ed[b]:.4g}", f"{ed[b+1]:.4g}", src,
                                 f"{100*env[b]:.2f}", f"{100*env_se[b]:.2f}", f"{100*nsub[b]:.2f}"])

    # overlay plots: each variation's dev +/- stat band, per pT panel
    for fam in PLOT_FAMS:
        fig, axes = plt.subplots(1, len(PT_BINS), figsize=(14, 4.0), sharey=False)
        drew = False
        for ax, ptb in zip(axes, PT_BINS):
            key = f"{fam}_{ptb}"
            if key not in nom:
                ax.set_visible(False); continue
            ed, vn, en = nom[key]
            nn, _, reln = norm_shape(ed, vn, en)
            base = np.where(nn > 0, nn, np.nan)
            c = 0.5 * (ed[:-1] + ed[1:])
            for m in [mm for ms in active.values() for mm in ms]:
                if key not in data[m]:
                    continue
                nm, _, relm = norm_shape(*data[m][key])
                dev = 100 * (nm - nn) / base
                ratio = np.where(nn > 0, nm / nn, 0.0)
                sdev = 100 * ratio * np.sqrt(relm**2 + reln**2)
                lbl, col = STYLE[m]
                ax.errorbar(c, np.nan_to_num(dev), yerr=np.nan_to_num(sdev), color=col,
                            label=lbl, lw=1.3, marker="o", ms=2.5, capsize=1.5, elinewidth=0.8)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_title(ptb.replace("pt", "pT ").replace("_", "-") + " GeV", fontsize=9)
            ax.set_xlabel(FAMILIES[fam], fontsize=10); ax.grid(alpha=0.3)
            drew = True
        if not drew:
            plt.close(fig); continue
        axes[0].set_ylabel("shape deviation from nominal [%]")
        axes[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"Per-variation shape deviation +/- MC stat error  -  {fam}  "
                     f"(mglo+Pythia8, 20k, area-norm; bars are conservative/uncorrelated)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(OUTDIR, f"dev_stat_{fam}.png"), dpi=130)
        plt.close(fig)
        print("wrote", os.path.join(OUTDIR, f"dev_stat_{fam}.png"))

    with open(os.path.join(OUTDIR, "model_unc_stat.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("wrote", os.path.join(OUTDIR, "model_unc_stat.csv"))


if __name__ == "__main__":
    main()
