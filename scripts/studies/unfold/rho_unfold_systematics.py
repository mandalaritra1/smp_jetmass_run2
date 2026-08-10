#!/usr/bin/env python
"""Propagate JER/JMS/JMR through the adopted hadronic groomed-rho unfolds.

Inputs are the NPZ files produced by ``rho_unfold_inputs_from_hists.py``:

    <channel>_nominal.npz
    <channel>_{JER,JMS,JMR}{Up,Down}.npz

For each variation the same data covariance is unfolded through the shifted
response, with that variation's inclusive reco/gen marginals used to rebuild
fakes and misses. Detector sources are combined in quadrature. Both the
symmetrized half-difference and the more conservative up/down envelope are
reported. Statistical covariances are propagated through the per-pT
normalization Jacobian before comparison with shape systematics.

ROOT/TUnfold must be importable in the selected Python environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make ``python scripts/rho_unfold_systematics.py ...`` work from the checkout
# without requiring callers to remember a PYTHONPATH override.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rho_unfold_stability import gen_map, tunfold


SOURCES = ("JER", "JMS", "JMR")
FLOOR = -4.0
SPECS = {
    "dijet": {
        "rho_groups": [
            [0], [1, 2], [3, 4], [5], [6], [7], [8], [9], [10], [11], [12]
        ],
        "norm": (-2.85, -0.55),
    },
    "trijet": {
        "rho_groups": [[0], [1, 2], [3], [4], [5], [6], [7]],
        "norm": (-3.0, -0.7),
    },
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return {key: value for key, value in np.load(path).items()}


def unfold_variation(inputs, rho_groups, tag):
    """Run one tau=0 unfold, rebuilding fakes/misses from this variation."""
    n_pt = len(inputs["pt_edges"]) - 1
    n_rho = len(inputs["rho_edges"]) - 1
    mapping = gen_map(
        n_pt, n_rho, [[index] for index in range(n_pt)], rho_groups
    )
    response = inputs["A"] @ mapping
    response_w2 = inputs["A_w2"] @ mapping
    gen = inputs["gen"] @ mapping
    gen_w2 = inputs["gen_w2"] @ mapping
    reco = inputs["reco"]

    fakes = reco - response.sum(axis=1)
    misses = gen - response.sum(axis=0)
    misses_w2 = np.maximum(gen_w2 - response_w2.sum(axis=0), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fake_fraction = np.where(reco > 0.0, fakes / reco, 0.0)
    survival = 1.0 - np.clip(fake_fraction, 0.0, 1.0)
    data = inputs["data_reco"] * survival
    data_covariance = inputs["data_V"] * np.outer(survival, survival)

    result = tunfold(
        response,
        response_w2,
        misses,
        misses_w2,
        data,
        data_covariance,
        tag,
    )
    return result, gen, response


def normalized_shape(values, n_pt, n_rho, in_window):
    values_2d = values.reshape(n_pt, n_rho)
    totals = values_2d[:, in_window].sum(axis=1, keepdims=True)
    return (values_2d / totals).reshape(-1)


def normalized_covariance(values, covariance, n_pt, n_rho, in_window):
    """Apply y_i=x_i/sum_window(x) independently in every pT slice."""
    values_2d = values.reshape(n_pt, n_rho)
    jacobian = np.zeros((values.size, values.size))
    window = in_window.astype(float)
    for pt_index in range(n_pt):
        offset = pt_index * n_rho
        total = values_2d[pt_index, in_window].sum()
        for rho_index in range(n_rho):
            row = offset + rho_index
            jacobian[row, offset : offset + n_rho] = (
                -values_2d[pt_index, rho_index] * window / total**2
            )
            jacobian[row, row] += 1.0 / total
    return jacobian @ covariance @ jacobian.T


def relative(delta, nominal):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(nominal > 0.0, delta / nominal, np.nan)


def summarize(values, mask):
    return {
        "median": float(np.nanmedian(values[mask])),
        "max": float(np.nanmax(values[mask])),
    }


def analyse_channel(channel: str, input_dir: Path):
    spec = SPECS[channel]
    groups = spec["rho_groups"]
    nominal_inputs = load_npz(input_dir / f"{channel}_nominal.npz")
    rho_edges = nominal_inputs["rho_edges"]
    gen_edges = np.r_[[rho_edges[group[0]] for group in groups], rho_edges[-1]]
    n_rho = len(groups)
    n_pt = len(nominal_inputs["pt_edges"]) - 1
    in_window = (
        (gen_edges[:-1] >= spec["norm"][0] - 1e-9)
        & (gen_edges[1:] <= spec["norm"][1] + 1e-9)
    )

    nominal_result, gen, nominal_response = unfold_variation(
        nominal_inputs, groups, f"_{channel}_nominal"
    )
    nominal = nominal_result["x"]
    nominal_shape = normalized_shape(nominal, n_pt, n_rho, in_window)
    stat_shape_covariance = normalized_covariance(
        nominal, nominal_result["Ein"], n_pt, n_rho, in_window
    )
    response_shape_covariance = normalized_covariance(
        nominal, nominal_result["Esys"], n_pt, n_rho, in_window
    )

    shown = (
        np.tile(gen_edges[:-1] >= FLOOR, (n_pt, 1)).reshape(-1)
        & (gen > 0.0)
        & (nominal_response.sum(axis=0) > 0.0)
    )
    core = shown & np.tile(in_window, n_pt)

    relative_half = {}
    relative_envelope = {}
    non_bracketing = {}
    for source in SOURCES:
        shifted_shapes = {}
        for direction in ("Up", "Down"):
            shifted_inputs = load_npz(
                input_dir / f"{channel}_{source}{direction}.npz"
            )
            shifted_result, _, _ = unfold_variation(
                shifted_inputs, groups, f"_{channel}_{source}{direction}"
            )
            shifted_shapes[direction] = normalized_shape(
                shifted_result["x"], n_pt, n_rho, in_window
            )

        half_difference = 0.5 * np.abs(
            shifted_shapes["Up"] - shifted_shapes["Down"]
        )
        envelope = np.maximum(
            np.abs(shifted_shapes["Up"] - nominal_shape),
            np.abs(shifted_shapes["Down"] - nominal_shape),
        )
        relative_half[source] = relative(half_difference, nominal_shape)
        relative_envelope[source] = relative(envelope, nominal_shape)
        same_side = (
            (shifted_shapes["Up"] - nominal_shape)
            * (shifted_shapes["Down"] - nominal_shape)
            > 0.0
        )
        non_bracketing[source] = int((same_side & core).sum())

    total_half = np.sqrt(
        sum(relative_half[source] ** 2 for source in SOURCES)
    )
    total_envelope = np.sqrt(
        sum(relative_envelope[source] ** 2 for source in SOURCES)
    )
    relative_stat = relative(
        np.sqrt(np.maximum(np.diag(stat_shape_covariance), 0.0)),
        nominal_shape,
    )
    relative_response = relative(
        np.sqrt(np.maximum(np.diag(response_shape_covariance), 0.0)),
        nominal_shape,
    )

    output = {
        "gen_rho_edges": gen_edges.tolist(),
        "pt_edges": nominal_inputs["pt_edges"].tolist(),
        "n_pt": n_pt,
        "n_rho": n_rho,
        "shown": shown.tolist(),
        "core": core.tolist(),
        "in_window": in_window.tolist(),
        "relative_half_difference": {
            source: values.tolist() for source, values in relative_half.items()
        },
        "relative_envelope": {
            source: values.tolist()
            for source, values in relative_envelope.items()
        },
        "total_half_difference": total_half.tolist(),
        "total_envelope": total_envelope.tolist(),
        "relative_data_stat": relative_stat.tolist(),
        "relative_response_stat": relative_response.tolist(),
        "non_bracketing_core_bins": non_bracketing,
        "core_summary": {
            "half_difference": summarize(total_half, core),
            "envelope": summarize(total_envelope, core),
            "data_stat": summarize(relative_stat, core),
            "response_stat": summarize(relative_response, core),
        },
    }

    summary = output["core_summary"]
    print(
        f"{channel}: {int(shown.sum())} reported bins, {int(core.sum())} core\n"
        f"  half-difference quadrature "
        f"{summary['half_difference']['median']:.2%} median / "
        f"{summary['half_difference']['max']:.2%} max\n"
        f"  envelope quadrature        "
        f"{summary['envelope']['median']:.2%} median / "
        f"{summary['envelope']['max']:.2%} max\n"
        f"  normalized data stat       "
        f"{summary['data_stat']['median']:.2%} median / "
        f"{summary['data_stat']['max']:.2%} max\n"
        f"  non-bracketing core bins   {non_bracketing}"
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="directory containing <channel>_<variation>.npz inputs",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--channel",
        action="append",
        choices=tuple(SPECS),
        help="channel to run; repeat for both (default: both)",
    )
    args = parser.parse_args(argv)

    channels = args.channel or list(SPECS)
    output = {
        channel: analyse_channel(channel, args.input_dir)
        for channel in channels
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
