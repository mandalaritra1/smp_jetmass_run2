#!/usr/bin/env python3
"""Bin-by-bin fractional modelling uncertainty per source, from the mglo+pythia
variation yodas produced by run_mg_local.sh.

For each measured observable (rho/mass, groomed/ungroomed, per pT bin) every
variation is area-normalised within that histogram (shape comparison -- the
measurement is normalised per pT bin), then per source:

    source_unc[bin] = max_member | norm_member[bin] - norm_nominal[bin] | / norm_nominal[bin]   (envelope)
    total[bin]      = sqrt( sum_source source_unc[bin]^2 )                                       (quadrature)

Sources (members auto-skipped if their yoda is absent):
    CR            : pythia_cr1   (QCD-inspired),   pythia_cr2 (gluon-move)
    Hadronization : pythia_fragsoft, pythia_fraghard
    Shower        : pythia_vincia            (add via run_mg_local.sh once present)

Outputs (out/model_unc/): one PNG+PDF per observable family (3 pT-bin panels)
and model_uncertainty.csv with every number.

Pure stdlib + numpy + matplotlib; parses the YODA3 text directly (no yoda module).
Run on the host venv:
    /Users/aritra/Projects/GluonJetMass/.venv/bin/python plot_model_uncertainty.py
or inside the rivet image.
"""
import os, re, csv, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# yoda family prefix: "mglo" (single-mult, default) or "mlm" (MLM-merged).
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "mglo"
OUTDIR = os.path.join(HERE, "out", "model_unc" if PREFIX == "mglo" else f"model_unc_{PREFIX}")
YODA = lambda name: os.path.join(HERE, "out", f"{PREFIX}_{name}.yoda")

NOMINAL = "pythia"
SOURCES = {                       # source label -> list of variation members
    "CR":            ["pythia_cr1", "pythia_cr2"],
    "Hadronization": ["pythia_fragsoft", "pythia_fraghard"],
    "Shower":        ["pythia_vincia"],
}
SRC_COLOR = {"CR": "tab:blue", "Hadronization": "tab:orange",
             "Shower": "tab:green", "Total": "black"}
# measured observables: (family key, pretty x-label)
FAMILIES = {
    "rho_g": r"$\rho_g = 2\log_{10}(m_g/(p_T R))$",
    "rho_u": r"$\rho_u$",
    "mass_g": r"$m_g$ [GeV]",
    "mass_u": r"$m_u$ [GeV]",
}
PT_BINS = ["pt200_290", "pt290_400", "pt400_Inf"]


def parse_yoda(path):
    """Return {histo_path: (edges[np], values[np])} for finalized ESTIMATE1D_V3
    objects under /CMS_ZJET_JETMASS/ (drops the /RAW/ fillable copies)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        lines = fh.readlines()
    i, n = 0, len(lines)
    while i < n:
        m = re.match(r"BEGIN YODA_ESTIMATE1D_V3 (\S+)", lines[i])
        if not m:
            i += 1; continue
        hpath = m.group(1)
        edges, vals, in_table = None, [], False
        i += 1
        while i < n and not lines[i].startswith("END YODA_ESTIMATE1D_V3"):
            ln = lines[i]
            if ln.startswith("Edges(A1):"):
                edges = [float(x) for x in re.findall(r"[-+0-9.eE]+", ln.split(":", 1)[1])]
            elif ln.startswith("# value"):
                in_table = True
            elif in_table:
                tok = ln.split()
                if tok:
                    vals.append(float("nan") if tok[0] == "nan" else float(tok[0]))
            i += 1
        if hpath.startswith("/CMS_ZJET_JETMASS/") and edges is not None and len(vals) >= 2:
            # value table = [underflow, bin_0..bin_{N-1}, overflow]; drop the flow rows
            body = np.array(vals[1:-1], dtype=float)
            ed = np.array(edges, dtype=float)
            if len(body) == len(ed) - 1:
                out[hpath.split("/")[-1]] = (ed, np.nan_to_num(body))
        i += 1
    return out


def area_normalise(edges, vals):
    w = np.diff(edges)
    integral = np.sum(vals * w)
    return vals / integral if integral > 0 else vals


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    nom = parse_yoda(YODA(NOMINAL))
    if not nom:
        raise SystemExit(f"no nominal histos in {YODA(NOMINAL)} -- run run_mg_local.sh first")

    # load only the sources whose member yodas exist
    active = {}
    for src, members in SOURCES.items():
        present = [mname for mname in members if os.path.exists(YODA(mname))]
        if present:
            active[src] = {mname: parse_yoda(YODA(mname)) for mname in present}
    print("active sources:", {s: list(m.keys()) for s, m in active.items()})

    csv_rows = [["observable", "pt_bin", "bin_lo", "bin_hi"] +
                list(active.keys()) + ["Total"]]

    for fam, xlabel in FAMILIES.items():
        fig, axes = plt.subplots(1, len(PT_BINS), figsize=(13, 3.6), sharey=True)
        any_panel = False
        for ax, ptb in zip(axes, PT_BINS):
            key = f"{fam}_{ptb}"
            if key not in nom:
                ax.set_visible(False); continue
            edges, nv = nom[key]
            nvn = area_normalise(edges, nv)
            centers = 0.5 * (edges[:-1] + edges[1:])
            with np.errstate(divide="ignore", invalid="ignore"):
                base = np.where(nvn > 0, nvn, np.nan)
            totsq = np.zeros_like(nvn)
            per_src = {}
            for src, members in active.items():
                env = np.zeros_like(nvn)
                for mname, hd in members.items():
                    if key not in hd:
                        continue
                    mvn = area_normalise(hd[key][0], hd[key][1])
                    dev = np.abs(mvn - nvn) / base
                    env = np.fmax(env, np.nan_to_num(dev))
                per_src[src] = env
                totsq += env ** 2
                ax.step(centers, 100 * env, where="mid", color=SRC_COLOR[src], label=src, lw=1.6)
            total = np.sqrt(totsq)
            ax.step(centers, 100 * total, where="mid", color="black", label="Total", lw=2.0)
            ax.set_title(ptb.replace("pt", "pT ").replace("_", "-") + " GeV", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.grid(alpha=0.3)
            any_panel = True
            for b in range(len(centers)):
                csv_rows.append([fam, ptb, f"{edges[b]:.4g}", f"{edges[b+1]:.4g}"] +
                                [f"{per_src[s][b]:.4f}" for s in active.keys()] +
                                [f"{total[b]:.4f}"])
        if not any_panel:
            plt.close(fig); continue
        axes[0].set_ylabel("fractional uncertainty [%]")
        axes[0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"Modelling uncertainty per source  -  {fam}  (mglo+Pythia8, area-normalised)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUTDIR, f"model_unc_{fam}.{ext}"), dpi=130)
        plt.close(fig)
        print("wrote", os.path.join(OUTDIR, f"model_unc_{fam}.png"))

    with open(os.path.join(OUTDIR, "model_uncertainty.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)
    print("wrote", os.path.join(OUTDIR, "model_uncertainty.csv"))


if __name__ == "__main__":
    main()
