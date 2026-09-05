#!/usr/bin/env python3
"""Build 24-bin CR/hadronization weighters from a pythia_var gen-rho cache.

The output uses the PtVarWeighter format and is written alongside the Vincia
weighter by default::

    python scripts/build_pythia_var_weighters.py /path/to/pythia_var_cache.npz
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

DEFAULT_FINE_INPUT = (
    Path.home()
    / "Projects/unfold/inputs/zjet/rho/finebins/minimal_rho_fine_pythia_2018.pkl"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "smp_jetmass_run2/corrections"
PT = np.array([200., 290., 400., 13000.])
SOURCES = ["cr1", "cr2", "fraghard", "fragsoft"]


def shape(cache, variation, mode, pt_index, edges):
    rho = cache[f"{variation}_rho_{mode}"]
    weight = cache[f"{variation}_weight"]
    jet_pt = cache[f"{variation}_jet_pt"]
    m = (
        (jet_pt >= PT[pt_index])
        & (jet_pt < PT[pt_index + 1])
        & np.isfinite(rho)
    )
    h, _ = np.histogram(rho[m], bins=edges, weights=weight[m])
    c, _ = np.histogram(rho[m], bins=edges)
    return h, c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="pythia_var gen-rho cache (.npz)")
    parser.add_argument(
        "--fine-input",
        type=Path,
        default=DEFAULT_FINE_INPUT,
        help="fine-bin nominal-rho pickle used to derive the 24-bin edges",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for the PtVarWeighter .npz files",
    )
    args = parser.parse_args(argv)

    with args.fine_input.open("rb") as source:
        fine_histograms = pickle.load(source)
    # Merge the 48-bin gen axis in pairs, matching the Vincia weighter.
    fine_edges = np.asarray(fine_histograms["ptjet_rhojet_g_gen"].axes[2].edges)
    rho_edges = fine_edges[::2]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.cache) as cache:
        for variation in SOURCES:
            for mode, tag in (("g", "groomed"), ("u", "ungroomed")):
                rho_grids = np.empty(4, dtype=object)
                weight_grids = np.empty(4, dtype=object)
                rho_grids[0] = np.array([rho_edges[0], rho_edges[-1]])
                weight_grids[0] = np.array([1.0, 1.0])
                for pt_index in range(3):
                    nominal, _ = shape(cache, "nom", mode, pt_index, rho_edges)
                    varied, varied_counts = shape(
                        cache, variation, mode, pt_index, rho_edges
                    )
                    nominal_fraction = nominal / nominal.sum()
                    varied_fraction = varied / varied.sum()
                    weights = np.ones_like(nominal_fraction)
                    good = (nominal_fraction > 1e-4) & (varied_counts >= 25)
                    weights[good] = np.clip(
                        varied_fraction[good] / nominal_fraction[good], 0.2, 5.0
                    )
                    rho_grids[pt_index + 1] = np.repeat(rho_edges, 2)[1:-1]
                    weight_grids[pt_index + 1] = np.repeat(weights, 2)

                output = args.output_dir / f"{variation}_rho_reweight_{tag}.npz"
                np.savez(
                    output,
                    pt_edges=np.array([0.0, 200.0, 290.0, 400.0, 13000.0]),
                    rho_grids=rho_grids,
                    w_grids=weight_grids,
                    source=f"{variation}_over_nom_pythia_var_24bin",
                )
                print("wrote", output)


if __name__ == "__main__":
    main()
