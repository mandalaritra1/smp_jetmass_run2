#!/usr/bin/env python
"""Assert `nEvents initial` in hadronic production pickles against DAS.

WHY THIS EXISTS (found 2026-08-08, auditing the 2018 LHE-basis archive):
coffea can silently lose whole chunks -- preprocessing enumerates fewer chunks
than a file contains, the run reports "Processing 100%" (of the chunks it
*discovered*), exits 0, and the logs contain no skip or failure message at
all. 10 of 28 files in the old 2018 archive were missing 6.2M events (0.15%)
this way, and 4 of 12 channel-eras of the fresh campaign show the same
signature at the 0.02-0.05% level. The only reliable detection is ground
truth: the processor's `nEvents initial` cutflow counter (every event handed
to `process()`) must equal the DAS event count of the input dataset. Run this
on every downloaded output before accepting a production.

WHAT IT CHECKS, per pickle:
- dijet outputs name their inputs: `nEvents initial /DAS/dataset/path` keys.
  Each dataset is checked individually.
- trijet outputs carry a single anonymous `nEvents initial`; the expected
  datasets are resolved from the filename (era + HT/Pt/Run token) against
  `samples/hadronic/fileset_*.json` and compared as one summed total.
- Legacy-mode trijet outputs (pre pair_split, GluonJetMass-inherited) counted
  `nEvents initial` TWICE per chunk; pair_split outputs count once. If the
  raw count is ~2x the expectation the tool halves it and says so -- a real
  chunk loss then still shows up as a nonzero delta on the halved value.

Any nonzero delta prints a loud FAIL line and the exit code is 1
(2 if some expectation could not be fetched at all). `--allow-frac X`
downgrades |delta|/expected < X to WARN for deltas that were reviewed and
accepted.

DAS BACKENDS (tried in order under the default `--backend auto`):
- cached value in `--cache` (DAS counts for a fixed dataset never change);
- `lxplus`: `dasgoclient` through the authenticated lxplus SSH master;
- `dbs`: DBS `filesummaries` on cmsweb with a valid local grid proxy
  (`voms-proxy-init -voms cms`; on macOS `X509_CERT_DIR` defaults to
  `~/.globus/certificates`, see tools/compute_sumw_qcd_mg.py);
- `mcm`: McM public REST (`completed_events` of the producing request).
  No auth needed, but MC only -- data (JetHT) needs one of the first two.

USAGE (repo venv -- unpickling the outputs imports coffea/hist):

    .venv/bin/python tools/check_das_event_counts.py \
        "/path/to/archive/2018/dijet_mc" outputs/some_skim.pkl

Arguments are pickle files or directories (searched recursively for *.pkl).
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "samples" / "hadronic"
DEFAULT_CACHE = SAMPLES / "das_nevents_cache.json"

#### filename token -> fileset holding its DAS dataset names
FILESETS = [
    ("data", SAMPLES / "fileset_JetHT.json"),
    ("mg_pythia8", SAMPLES / "fileset_MG_pythia8.json"),
    ("herwig", SAMPLES / "fileset_HERWIG_wRedirs.json"),
    ("pythia", SAMPLES / "fileset_QCD.json"),
]

DBS_URL = ("https://cmsweb.cern.ch/dbs/prod/global/DBSReader/"
           "filesummaries?dataset={ds}")
MCM_URL = ("https://cms-pdmv-prod.web.cern.ch/mcm/public/restapi/"
           "requests/produces{ds}")


def era_of_dataset(das: str) -> str:
    """Era of a DAS dataset name, MC and data conventions both."""
    if "APV" in das or "HIPM" in das:
        return "2016APV"
    if "UL18" in das or "UL2018" in das:
        return "2018"
    if "UL17" in das or "UL2017" in das:
        return "2017"
    return "2016"


def era_of_filename(name: str) -> str | None:
    m = re.search(r"(?<=_)(2016APV|2016|2017|2018)(?=[_.])", name)
    return m.group(1) if m else None


def resolve_datasets(pkl_path: Path) -> list[str]:
    """DAS datasets an anonymous (trijet-style) pickle was produced from.

    Filename convention: <mode>_<channel>_<sample>_<era>[_<bin-or-run>].pkl.
    No bin/run token means an era-wide (all_syst) production over every
    dataset of that era in the fileset.
    """
    name = pkl_path.name
    era = era_of_filename(name)
    if era is None:
        raise ValueError(f"no era token in filename: {name}")
    for token, fs_path in FILESETS:
        if token in name:
            fileset = json.load(fs_path.open())
            break
    else:
        raise ValueError(f"no sample token ({'/'.join(t for t, _ in FILESETS)}) "
                         f"in filename: {name}")
    datasets = [ds for group in fileset.values() for ds in group
                if era_of_dataset(ds) == era]
    binm = re.search(r"(HT|Pt_)\d+to(?:\d+|Inf)", name)
    runm = re.search(r"Run201\d[A-H](?:-\w+)?", name)
    if binm:
        datasets = [ds for ds in datasets if binm.group(0) + "_" in ds]
    elif runm:
        datasets = [ds for ds in datasets
                    if ds.split("/")[2].startswith(runm.group(0))]
    if not datasets:
        raise ValueError(f"no {era} datasets match {name!r} in {fs_path.name}")
    return datasets


def extract_reads(out: dict) -> tuple[dict[str, int], int | None]:
    """(per-dataset counts from dijet-style keys, anonymous trijet total)."""
    per_dataset: dict[str, int] = {}
    anonymous = None
    for group in out.get("cutflow", {}).values():
        if not hasattr(group, "items"):
            continue
        for key, val in group.items():
            if key == "nEvents initial":
                anonymous = (anonymous or 0) + int(val)
            elif key.startswith("nEvents initial /"):
                das = key[len("nEvents initial "):]
                per_dataset[das] = per_dataset.get(das, 0) + int(val)
    return per_dataset, anonymous


#### ---------------------------------------------------------------- DAS ----

class BackendDown(Exception):
    """Infrastructure failure (expired auth, no network) -- distinct from a
    backend that answered but has nothing for this particular dataset."""


def _curl(url: str, extra: list[str], timeout: int) -> str:
    cmd = ["curl", "-sS", "--fail", "--max-time", str(timeout)] + extra + [url]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout + 15)
    except subprocess.TimeoutExpired as exc:
        raise BackendDown("curl timeout") from exc
    if res.returncode != 0:
        raise BackendDown(res.stderr.strip()[:120])
    return res.stdout


def query_lxplus(dataset: str, timeout: int) -> int | None:
    inner = f"dasgoclient -query 'summary dataset={dataset}' -json"
    try:
        #### BatchMode and closed stdin make an expired SSH master fail fast
        #### instead of waiting for a password or OTP prompt.
        res = subprocess.run(["ssh", "-o", "BatchMode=yes", "lxplus", inner],
                             capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as exc:
        raise BackendDown("lxplus SSH timeout") from exc
    if res.returncode != 0:
        raise BackendDown(res.stderr.strip().splitlines()[-1][:120]
                          if res.stderr.strip() else "lxplus SSH failed")
    try:
        recs = json.loads(res.stdout)
        for rec in recs:
            for summ in rec.get("summary", []):
                if "nevents" in summ:
                    return int(summ["nevents"])
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def query_dbs(dataset: str, timeout: int) -> int | None:
    import os
    proxy = os.environ.get("X509_USER_PROXY", f"/tmp/x509up_u{os.getuid()}")
    capath = os.environ.get("X509_CERT_DIR",
                            str(Path.home() / ".globus" / "certificates"))
    if not Path(proxy).exists():
        raise BackendDown(f"no grid proxy at {proxy}")
    text = _curl(DBS_URL.format(ds=dataset),
                 ["--cert", proxy, "--key", proxy, "--capath", capath],
                 timeout)
    try:
        return int(json.loads(text)[0]["num_event"])
    except (json.JSONDecodeError, LookupError, TypeError, ValueError):
        return None


def is_data_dataset(dataset: str) -> bool:
    """Data era campaigns look like Run2018B-UL2018_...; MC campaigns like
    RunIISummer20UL18... -- only the era pattern means real data."""
    return re.match(r"Run20\d{2}[A-H]", dataset.split("/")[2]) is not None


def query_mcm(dataset: str, timeout: int) -> int | None:
    if is_data_dataset(dataset):
        return None  # McM only knows MC requests
    #### UL16(nonAPV)v9 MC is published under /NANOAOD in DBS, but McM
    #### registered the request output as /NANOAODSIM -- try both tiers
    tries = [dataset]
    if dataset.endswith("/NANOAOD"):
        tries.append(dataset + "SIM")
    for ds in tries:
        text = _curl(MCM_URL.format(ds=ds), [], timeout)
        try:
            res = json.loads(text)["results"]
            if res and res.get("status") == "done":
                return int(res["completed_events"])
        except (json.JSONDecodeError, LookupError, TypeError, ValueError):
            pass
    return None


BACKENDS = {"lxplus": query_lxplus, "dbs": query_dbs, "mcm": query_mcm}


class DASCounts:
    def __init__(self, cache_path: Path, backend: str, timeout: int,
                 refresh: bool):
        self.cache_path = cache_path
        self.order = list(BACKENDS) if backend == "auto" else [backend]
        self.timeout = timeout
        self.cache: dict[str, dict] = {}
        self.misses: set[str] = set()
        #### a backend that is DOWN (BackendDown) 3 datasets in a row stays
        #### disabled for the run -- stop paying its timeout every dataset.
        #### An answered "nothing for this dataset" (None) does not count.
        self.strikes = {name: 0 for name in BACKENDS}
        if cache_path.exists() and not refresh:
            self.cache = json.load(cache_path.open())

    def get(self, dataset: str) -> int | None:
        if dataset in self.cache:
            return int(self.cache[dataset]["nevents"])
        if dataset in self.misses:
            return None
        for name in self.order:
            if self.strikes[name] >= 3:
                continue
            try:
                nev = BACKENDS[name](dataset, self.timeout)
            except BackendDown as exc:
                self.strikes[name] += 1
                if self.strikes[name] == 3:
                    print(f"   (backend {name} disabled for this run: {exc})")
                continue
            self.strikes[name] = 0
            if nev is not None:
                self.cache[dataset] = {"nevents": nev, "backend": name}
                self.cache_path.write_text(
                    json.dumps(self.cache, indent=1, sort_keys=True) + "\n")
                return nev
        self.misses.add(dataset)
        return None


#### ------------------------------------------------------------- report ----

def check_pickle(path: Path, das: DASCounts, allow_frac: float) -> str:
    """Check one pickle; prints its report; returns worst status."""
    with path.open("rb") as fh:
        out = pickle.load(fh)
    per_dataset, anonymous = extract_reads(out)

    rows = []  # (label, expected|None, read, note)
    if per_dataset:
        for ds_name, read in sorted(per_dataset.items()):
            rows.append((ds_name, das.get(ds_name), read, ""))
    if anonymous is not None:
        try:
            datasets = resolve_datasets(path)
        except ValueError as exc:
            rows.append(("<unresolved inputs>", None, anonymous, str(exc)))
            datasets = []
        if datasets:
            expected_each = [das.get(ds_name) for ds_name in datasets]
            expected = (sum(expected_each)
                        if None not in expected_each else None)
            read, note = anonymous, ""
            if expected and read / expected > 1.5:
                read, note = read // 2, "legacy 2x nEvents initial, halved"
            rows.append((f"sum of {len(datasets)} dataset(s)",
                         expected, read, note))
    if not rows:
        rows.append(("<no nEvents initial in cutflow>", None, 0, ""))

    worst = "OK"
    print(f"== {path}")
    for label, expected, read, note in rows:
        note = f"   [{note}]" if note else ""
        if expected is None:
            status = "NO-REF"
            print(f"   NO-REF {label}  read={read:,}  "
                  f"(no DAS backend could answer){note}")
        else:
            delta = read - expected
            frac = delta / expected if expected else 0.0
            status = ("OK" if delta == 0
                      else "WARN" if abs(frac) < allow_frac else "FAIL")
            print(f"   {status:6s} {label}  expected={expected:,} "
                  f"read={read:,}  delta={delta:+,} ({frac:+.4%}){note}")
        worst = max(worst, status, key=["OK", "WARN", "NO-REF", "FAIL"].index)
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path,
                    help="production pickles and/or directories of them")
    ap.add_argument("--backend", choices=["auto"] + list(BACKENDS),
                    default="auto")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                    help=f"DAS-count cache (default {DEFAULT_CACHE})")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cached counts, re-query DAS")
    ap.add_argument("--allow-frac", type=float, default=0.0,
                    help="downgrade FAIL to WARN below this |delta|/expected "
                         "(for reviewed-and-accepted deltas; default 0)")
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args(argv)

    pickles = []
    for p in args.paths:
        if p.is_dir():
            pickles += sorted(p.rglob("*.pkl"))
        elif p.exists():
            pickles.append(p)
        else:
            print(f"missing path: {p}", file=sys.stderr)
            return 2
    das = DASCounts(args.cache, args.backend, args.timeout, args.refresh)

    statuses = [check_pickle(p, das, args.allow_frac) for p in pickles]
    n = {s: statuses.count(s) for s in ("OK", "WARN", "NO-REF", "FAIL")}
    print(f"\n{len(statuses)} pickle(s): {n['OK']} OK, {n['WARN']} WARN, "
          f"{n['NO-REF']} NO-REF, {n['FAIL']} FAIL")
    if n["FAIL"]:
        print("FAIL: at least one production read fewer/more events than DAS "
              "-- silently dropped chunks; re-produce before using.")
        return 1
    return 2 if n["NO-REF"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
