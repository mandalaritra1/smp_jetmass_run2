#!/usr/bin/env python
"""Headless runner for all three channels (zjet / dijet / trijet).

Reads a JSON config (see configs/*.json for examples), builds the fileset for
the requested channel/dataset/era, runs the matching processor via
notebook_utils helpers, and writes pickle outputs under `outputs/`.

    python scripts/run_analysis_cli.py --config configs/dijet_pythia_2018.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dask

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from coffea.nanoevents import NanoAODSchema  # noqa: E402

import smp_jetmass_run2.notebook_utils as nbutils  # noqa: E402


def _flush_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _run_and_save(
    *,
    fileset,
    index,
    cfg,
    paths,
    client,
):
    out = nbutils.run_once(
        fileset,
        client=client,
        test=cfg["test"],
        data=cfg["dataset"] == "data",
        mode=cfg["mode"],
        channel=cfg["channel"],
        systematic_profile=cfg["systematic_profile"],
        chunksize=cfg["chunksize"],
        chunksize_test=cfg["chunksize_test"],
        executor_mode=cfg["executor_mode"],
    )
    tag = nbutils.get_group_tag(index, cfg["era"], cfg["group_mode"])
    fout = nbutils.make_output_filename(
        data=cfg["dataset"] == "data",
        dataset=cfg["dataset"],
        tag=tag,
        mode=cfg["mode"],
        channel=cfg["channel"],
        test=cfg["test"],
        output_dir=paths.repo_root / "outputs",
    )
    nbutils.save_output(out, fout)
    _flush_print(f"[{index + 1}] Saved: {fout}")
    return fout


def run(cfg: dict) -> list[str]:
    paths = nbutils.get_analysis_paths(REPO_ROOT)
    samplePath = nbutils.SamplePath(cfg["era"])
    prependstr = cfg["prependstr"]

    NanoAODSchema.warn_missing_crossrefs = False
    dask.config.set({
        "distributed.logging.distributed": "error",
        "distributed.logging.bokeh": "error",
        "distributed.logging.tornado": "error",
    })

    _flush_print("Configuration:")
    for key in sorted(cfg):
        _flush_print(f"  {key} = {cfg[key]}")

    client = nbutils.ensure_client(
        casa=cfg["casa"],
        test=cfg["test"],
        useDefault=cfg["useDefault"],
        executor_mode=cfg["executor_mode"],
    )
    nbutils.upload_package_if_casa(client, casa=cfg["casa"])

    outputs: list[str] = []
    dataset = cfg["dataset"]
    group_mode = cfg["group_mode"]

    try:
        if cfg["channel"] in ("dijet", "trijet"):
            fileset = nbutils.build_hadronic_fileset(
                paths.samples_hadronic_dir,
                dataset=dataset,
                era=cfg["era"],
                redirector=cfg["redirector"],
                prepend=prependstr,
            )
            outputs.append(_run_and_save(
                fileset=fileset, index=0, cfg=cfg, paths=paths, client=client,
            ))
        elif dataset == "data":
            for i, group in enumerate(nbutils.iter_groups(samplePath.data, group_mode)):
                fileset = nbutils.build_fileset_from_txts(
                    group, paths.samples_data_dir, prependstr, split_ht=False,
                )
                outputs.append(_run_and_save(
                    fileset=fileset, index=i, cfg=cfg, paths=paths, client=client,
                ))
        elif dataset == "pythia":
            for i, group in enumerate(nbutils.iter_groups(samplePath.pythia, group_mode)):
                fileset = nbutils.build_fileset_from_txts(
                    group, paths.samples_mc_dir, prependstr,
                    split_ht=True, ht_bins=nbutils.HT_BINS,
                )
                outputs.append(_run_and_save(
                    fileset=fileset, index=i, cfg=cfg, paths=paths, client=client,
                ))
        elif dataset == "pythia_local":
            fileset = nbutils.build_local_pythia_fileset(
                paths.samples_mc_local_dir, cfg["era"],
            )
            outputs.append(_run_and_save(
                fileset=fileset, index=0, cfg=cfg, paths=paths, client=client,
            ))
        elif dataset == "pythia2":
            fileset = nbutils.build_fileset_from_txts(
                ["inclusive_UL16NanoAODv9.txt"], paths.samples_mc_dir, prependstr,
                split_ht=False,
            )
            outputs.append(_run_and_save(
                fileset=fileset, index=0, cfg=cfg, paths=paths, client=client,
            ))
        elif dataset == "herwig":
            for i, group in enumerate(nbutils.iter_groups(samplePath.herwig, group_mode)):
                fileset = nbutils.build_fileset_from_txts(
                    group, paths.samples_mc_dir, prependstr, split_ht=False,
                )
                outputs.append(_run_and_save(
                    fileset=fileset, index=i, cfg=cfg, paths=paths, client=client,
                ))
        elif dataset == "powheg":
            fileset = nbutils.build_fileset_from_txts(
                ["powheg_UL18NanoAODv9_inclusive.txt"], paths.samples_mc_dir, prependstr,
                split_ht=False,
            )
            outputs.append(_run_and_save(
                fileset=fileset, index=0, cfg=cfg, paths=paths, client=client,
            ))
        elif dataset == "st":
            fileset = nbutils.build_fileset_from_txts(
                nbutils.ST_FILES, paths.samples_mc_dir, prependstr, split_ht=False,
            )
            outputs.append(_run_and_save(
                fileset=fileset, index=0, cfg=cfg, paths=paths, client=client,
            ))
        elif dataset == "backgrounds":
            fileset = nbutils.build_backgrounds_fileset(
                paths.samples_bkg_dir, prependstr,
            )
            outputs.append(_run_and_save(
                fileset=fileset, index=0, cfg=cfg, paths=paths, client=client,
            ))
        else:
            _flush_print(f"Dataset is {dataset} and it is not in the list")
    finally:
        if client is not None:
            client.close()

    _flush_print(f"Number of group outputs: {len(outputs)}")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to JSON config (same schema as .analysis_widget_config.json).",
    )
    args = parser.parse_args(argv)

    raw_cfg = json.loads(args.config.read_text())
    cfg = nbutils.validate_analysis_config(raw_cfg)
    outputs = run(cfg)

    print("---OUTPUT-FILES---", flush=True)
    for path in outputs:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
