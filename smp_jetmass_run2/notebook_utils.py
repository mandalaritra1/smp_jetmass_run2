import os
import json
import pickle
import shutil
import time
import logging
from dataclasses import dataclass
from pathlib import Path

from coffea import processor
from coffea.nanoevents import NanoAODSchema


HT_BINS = [
    "100to200",
    "200to400",
    "400to600",
    "600to800",
    "800to1200",
    "1200to2500",
    "2500toInf",
]

DATASET_OPTIONS = [
    "data",
    "pythia",
    "mg_pythia8",
    "pythia_local",
    "pythia2",
    "nlo",
    "nlo_ptz",
    "herwig",
    "st",
    "powheg",
    "backgrounds",
]

ERA_OPTIONS = ["2016", "2016APV", "2017", "2018", "all"]
CHANNEL_OPTIONS = ["zjet", "dijet", "trijet"]

RHO_MODE_OPTIONS = [
    "minimal_rho",
    "minimal_rho_fine",
    "minimal_rho_fine_split",
    "rho_jk",
    "reweight_pythia_rho",
    "reweight_data_prior_rho",
]
MASS_MODE_OPTIONS = [
    "minimal",
    "mass",
    "mass_reweight",
    "mass_jk",
    "mass_jk_mc",
    "mass_jk_data",
    "mass_diagnostic_ntuple",
]
MODE_OPTIONS = ["validation", "full"] + RHO_MODE_OPTIONS + MASS_MODE_OPTIONS
SYSTEMATIC_PROFILE_OPTIONS = ["all_syst", "minimal_syst", "no_syst"]
EXECUTOR_MODE_OPTIONS = [
    "futures",
    "dask-local",
    "dask-casa",
    "dask-lpc",
]
REDIRECTOR_PREPENDS = {
    "local": "",
    "lpc": "root://cmsxrootd.fnal.gov/",
    "casa": "root://xcache/",
}
REDIRECTOR_OPTIONS = list(REDIRECTOR_PREPENDS)

OUTPUT_MODE_TAGS = {
    # Shorter filename tag for the opt-in data-prior rho closure/stress test.
    "reweight_data_prior_rho": "data_prior_weighted",
}

_MASS_MODE_ALIASES = {
    "minimal",
    "reweight_pythia",
    "jk_mc",
    "jk_data",
    *MASS_MODE_OPTIONS,
}

ST_FILES = [
    "st_tW_antitop_UL16NanoAODv9.txt",
    "st_tW_antitop_UL16NanoAODAPVv9.txt",
    "st_tW_antitop_UL17NanoAODv9.txt",
    "st_tW_antitop_UL18NanoAODv9.txt",
    "st_tW_top_UL16NanoAODv9.txt",
    "st_tW_top_UL16NanoAODAPVv9.txt",
    "st_tW_top_UL17NanoAODv9.txt",
    "st_tW_top_UL18NanoAODv9.txt",
    "ST_t-channel_antitop_4f_InclusiveDecays_UL16NanoAODv9.txt",
    "ST_t-channel_antitop_4f_InclusiveDecays_UL16NanoAODAPVv9.txt",
    "ST_t-channel_antitop_4f_InclusiveDecays_UL17NanoAODv9.txt",
    "ST_t-channel_antitop_4f_InclusiveDecays_UL18NanoAODv9.txt",
    "ST_t-channel_top_4f_InclusiveDecays_UL16NanoAODv9.txt",
    "ST_t-channel_top_4f_InclusiveDecays_UL16NanoAODAPVv9.txt",
    "ST_t-channel_top_4f_InclusiveDecays_UL17NanoAODv9.txt",
    "ST_t-channel_top_4f_InclusiveDecays_UL18NanoAODv9.txt",
]


def resolve_redirector_prepend(redirector: str) -> str:
    if redirector not in REDIRECTOR_PREPENDS:
        raise ValueError(
            f"Unknown redirector '{redirector}'. "
            f"Choose from {', '.join(REDIRECTOR_OPTIONS)}."
        )
    return REDIRECTOR_PREPENDS[redirector]


def infer_redirector_from_prepend(prepend: str) -> str:
    for name, value in REDIRECTOR_PREPENDS.items():
        if prepend == value:
            return name
    return "casa"

MINIMAL_JET_SYSTEMATICS = ["nominal", "JERUp", "JERDown"]
NO_SYST_SYSTEMATICS = ["nominal"]


ANALYSIS_CONFIG_DEFAULTS = {
    "casa": True,
    "test": False,
    "useDefault": False,
    "executor_mode": "futures",
    "mode": "validation",
    "channel": "zjet",
    "era": "2016",
    "dataset": "pythia",
    "chunksize": 50000,
    "chunksize_test": 200000,
    "group_mode": "all_in_one",
    "redirector": "casa",
    "prependstr": "root://xcache/",
    "systematic_profile": "all_syst",
    # optional DAS-name substring filter (hadronic): e.g. "HT1000to1500" to point a
    # quick test at a high-HT bin (low-HT bins select nothing in dijet/trijet).
    "dataset_filter": "",
}


def validate_analysis_config(cfg: dict) -> dict:
    """Merge user config onto defaults and normalize stale option values.

    Mirrors the shape-checks the notebook performs after loading
    `.analysis_widget_config.json`: unknown enum values fall back to the
    default for that key, and the redirector is re-derived from prependstr
    when missing or stale.
    """
    merged = ANALYSIS_CONFIG_DEFAULTS.copy()
    merged.update(cfg or {})

    if merged["dataset"] not in DATASET_OPTIONS:
        merged["dataset"] = ANALYSIS_CONFIG_DEFAULTS["dataset"]
    if merged["era"] not in ERA_OPTIONS:
        merged["era"] = ANALYSIS_CONFIG_DEFAULTS["era"]
    if merged["mode"] not in MODE_OPTIONS:
        merged["mode"] = ANALYSIS_CONFIG_DEFAULTS["mode"]
    if merged.get("channel") not in CHANNEL_OPTIONS:
        merged["channel"] = ANALYSIS_CONFIG_DEFAULTS["channel"]
    if merged.get("systematic_profile") not in SYSTEMATIC_PROFILE_OPTIONS:
        merged["systematic_profile"] = ANALYSIS_CONFIG_DEFAULTS["systematic_profile"]
    if merged.get("executor_mode") not in EXECUTOR_MODE_OPTIONS:
        merged["executor_mode"] = ANALYSIS_CONFIG_DEFAULTS["executor_mode"]
    if merged.get("redirector") not in REDIRECTOR_OPTIONS:
        merged["redirector"] = infer_redirector_from_prepend(
            merged.get("prependstr", ANALYSIS_CONFIG_DEFAULTS["prependstr"])
        )

    merged["prependstr"] = resolve_redirector_prepend(merged["redirector"])
    return merged


@dataclass(frozen=True)
class AnalysisPaths:
    repo_root: Path
    samples_data_dir: Path
    samples_mc_dir: Path
    samples_bkg_dir: Path
    samples_mc_local_dir: Path
    samples_hadronic_dir: Path


# PtZ-binned NLO (amcatnloFXFX, MatchEWPDG20) DY. Bins are stitched by normalizing
# each to its own XSDB cross section in the processor postprocess (each bin is a
# separate `dataset` axis entry -> summing them gives the ptZ spectrum). Lists for
# all six bins are built by samples/zjet/mc/make_nlo_ptz_lists.sh (dasgoclient).
#
# The analysis selects jet pT > 200 GeV (first ptgen/ptreco bin is 200-290). In the
# balanced Z+jet topology jet pT ~ ptZ, so the low-ptZ bins contribute ~nothing:
# ptZ<100 cannot recoil against a 200 GeV jet. We therefore process only the bins
# that overlap the phase space. ptZ-100To250 is KEPT (its 200-250 part populates the
# first 200-290 jet-pT bin); 0To50 and 50To100 are dropped (pure wasted compute, the
# two largest samples). To use all six (e.g. a full-spectrum cross-check), set
# NLO_PTZ_BINS = NLO_PTZ_BINS_ALL.
NLO_PTZ_BINS_ALL = ["0To50", "50To100", "100To250", "250To400", "400To650", "650ToInf"]
NLO_PTZ_BINS = ["100To250", "250To400", "400To650", "650ToInf"]


def _nlo_ptz_lists(era_tag: str) -> list[str]:
    return [f"ptz_{b}_{era_tag}.txt" for b in NLO_PTZ_BINS]


class SamplePath:
    """Hold list-of-lists so the notebook can run per-group or all-in-one."""

    def __init__(self, era: str):
        self.era = era

        if era == "all":
            self.data = [
                ["SingleMuon_UL2018.txt", "EGamma_UL2018.txt"],
                ["SingleMuon_UL2017.txt", "SingleElectron_UL2017.txt"],
                ["SingleMuon_UL2016APV.txt", "SingleElectron_UL2016APV.txt"],
                ["SingleMuon_UL2016.txt", "SingleElectron_UL2016.txt"],
            ]
            self.pythia = [
                ["pythia_UL16NanoAODAPVv9.txt"],
                ["pythia_UL16NanoAODv9.txt"],
                ["pythia_UL17NanoAODv9.txt"],
                ["pythia_UL18NanoAODv9.txt"],
            ]
            self.herwig = [
                ["herwig7_UL16NanoAODAPVv9_inclusive.txt"],
                ["herwig7_UL16NanoAODv9_inclusive.txt"],
                ["herwig7_UL17NanoAODv9_inclusive.txt"],
                ["herwig7_UL18NanoAODv9_inclusive.txt"],
            ]
            # NLO (amcatnloFXFX) DY. Order matches get_group_tag's era_tags
            # (["2016","2016APV","2017","2018"]) so per_group output tags are
            # correct. The non-2016 lists are produced by samples/zjet/mc/
            # make_nlo_lists.sh (dasgoclient) before running era="all".
            self.nlo = [
                ["inclusive_UL16NanoAODv9.txt"],
                ["inclusive_UL16NanoAODAPVv9.txt"],
                ["inclusive_UL17NanoAODv9.txt"],
                ["inclusive_UL18NanoAODv9.txt"],
            ]
            # PtZ-binned NLO: one group per era, each holding the 6 ptZ bins
            # (same era order as self.nlo so per_group tags line up).
            self.nlo_ptz = [
                _nlo_ptz_lists("UL16NanoAODv9"),
                _nlo_ptz_lists("UL16NanoAODAPVv9"),
                _nlo_ptz_lists("UL17NanoAODv9"),
                _nlo_ptz_lists("UL18NanoAODv9"),
            ]
        elif era == "2018":
            self.data = [["SingleMuon_UL2018.txt", "EGamma_UL2018.txt"]]
            self.pythia = [["pythia_UL18NanoAODv9.txt"]]
            self.herwig = [["herwig7_UL18NanoAODv9_inclusive.txt"]]
            self.nlo = [["inclusive_UL18NanoAODv9.txt"]]
            self.nlo_ptz = [_nlo_ptz_lists("UL18NanoAODv9")]
        elif era == "2017":
            self.data = [["SingleMuon_UL2017.txt", "SingleElectron_UL2017.txt"]]
            self.pythia = [["pythia_UL17NanoAODv9.txt"]]
            self.herwig = [["herwig7_UL17NanoAODv9_inclusive.txt"]]
            self.nlo = [["inclusive_UL17NanoAODv9.txt"]]
            self.nlo_ptz = [_nlo_ptz_lists("UL17NanoAODv9")]
        elif era == "2016APV":
            self.data = [["SingleMuon_UL2016APV.txt", "SingleElectron_UL2016APV.txt"]]
            self.pythia = [["pythia_UL16NanoAODAPVv9.txt"]]
            self.herwig = [["herwig7_UL16NanoAODAPVv9_inclusive.txt"]]
            self.nlo = [["inclusive_UL16NanoAODAPVv9.txt"]]
            self.nlo_ptz = [_nlo_ptz_lists("UL16NanoAODAPVv9")]
        elif era == "2016":
            self.data = [["SingleMuon_UL2016.txt", "SingleElectron_UL2016.txt"]]
            self.pythia = [["pythia_UL16NanoAODv9.txt"]]
            self.herwig = [["herwig7_UL16NanoAODv9_inclusive.txt"]]
            self.nlo = [["inclusive_UL16NanoAODv9.txt"]]
            self.nlo_ptz = [_nlo_ptz_lists("UL16NanoAODv9")]
        else:
            raise ValueError(f"Unknown era: {era}")


def resolve_repo_root(start: Path | None = None) -> Path:
    repo_root = (start or Path.cwd()).resolve()
    if (repo_root / "smp_jetmass_run2").exists() or (repo_root / "src" / "smp_jetmass_run2").exists():
        return repo_root
    if not (repo_root / "smp_jetmass_run2").exists():
        repo_root = repo_root.parent
    return repo_root


def get_analysis_paths(repo_root: Path | None = None) -> AnalysisPaths:
    root = resolve_repo_root(repo_root)
    return AnalysisPaths(
        repo_root=root,
        samples_data_dir=root / "samples" / "zjet" / "data",
        samples_mc_dir=root / "samples" / "zjet" / "mc",
        samples_bkg_dir=root / "samples" / "zjet" / "mc" / "backgrounds",
        samples_mc_local_dir=root / "samples" / "zjet" / "mc" / "files",
        samples_hadronic_dir=root / "samples" / "hadronic",
    )


def format_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def iter_groups(list_of_lists, mode: str):
    """Yield groups based on the notebook's intended semantics."""
    if mode == "per_group":
        for group in list_of_lists:
            yield group
    elif mode == "all_in_one":
        flat = []
        for group in list_of_lists:
            flat.extend(group)
        yield flat
    else:
        raise ValueError(f"Unknown group_mode: {mode}")


def read_txt_lines(txt_file: str | os.PathLike[str]) -> list[str]:
    with open(txt_file) as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def build_fileset_from_txts(
    txt_files: list[str],
    base_dir: str | os.PathLike[str],
    prepend: str,
    split_ht: bool = False,
    ht_bins: list[str] | None = None,
) -> dict[str, list[str]]:
    fileset = {}

    for filename in txt_files:
        sample = filename.split(".")[0]
        fullpath = os.path.join(base_dir, filename)
        lines = read_txt_lines(fullpath)

        if split_ht:
            if not ht_bins:
                raise ValueError("split_ht=True requires ht_bins")
            for ht_bin in ht_bins:
                files = [prepend + line for line in lines if ht_bin in line]
                fileset[f"{sample}_HT-{ht_bin}"] = files
        else:
            fileset[sample] = [prepend + line for line in lines]

    return {key: value for key, value in fileset.items() if value}


def build_backgrounds_fileset(
    directory: str | os.PathLike[str],
    prepend: str,
) -> dict[str, list[str]]:
    fileset = {}
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        core = filename[:-4]
        index = core.find("UL")
        if index == -1:
            print(f"Warning: 'UL' not found in {core}, skipping")
            continue
        sample = core[:index]
        era_version = core[index:]
        key = f"{sample}_{era_version}"
        lines = read_txt_lines(os.path.join(directory, filename))
        fileset[key] = [prepend + line for line in lines]
    return fileset


def build_local_pythia_fileset(
    directory: str | os.PathLike[str],
    era: str,
) -> dict[str, list[str]]:
    era_map = {
        "2016": "UL16NanoAODv9",
        "2016APV": "UL16NanoAODAPVv9",
        "2017": "UL17NanoAODv9",
        "2018": "UL18NanoAODv9",
    }
    if era == "all":
        raise ValueError("pythia_local requires a specific era, not era='all'.")
    if era not in era_map:
        raise ValueError(f"Unsupported era for pythia_local: {era}")

    fileset = {}
    for path in sorted(Path(directory).rglob("*.root")):
        stem = path.stem
        ht_source = stem if "HT" in stem else str(path)
        if "HT" not in ht_source:
            print(f"Warning: could not infer HT bin from local file {path.name}, skipping")
            continue

        ht_token = ht_source.split("HT", 1)[1].split("_")[0].lstrip("_-")
        dataset_name = f"pythia_{era_map[era]}_HT-{ht_token}"
        fileset[dataset_name] = [str(path)]

    if not fileset:
        raise FileNotFoundError(f"No local ROOT files found in {directory}")
    return fileset


HADRONIC_ERAS_MC = {
    "2016APV": "UL16NanoAODAPV",
    "2016": "UL16NanoAODv9",
    "2017": "UL17NanoAODv9",
    "2018": "UL18NanoAODv9",
}
HADRONIC_ERAS_DATA = {
    "2016APV": "HIPM_UL2016",
    "2016": "-UL2016",
    "2017": "UL2017",
    "2018": "UL2018",
}
HADRONIC_FILESETS = {
    "pythia": {"default": "fileset_QCD_wRedirs.json", "casa": "fileset_QCD.json"},
    "mg_pythia8": {"default": "fileset_MG_pythia8_wRedirs.json", "casa": "fileset_MG_pythia8.json"},
    "herwig": {"default": "fileset_HERWIG_wRedirs.json"},
    "data": {"default": "fileset_JetHT_wRedirs.json", "casa": "fileset_JetHT.json"},
}
HADRONIC_MODE_ALIASES = {
    "mass": "minimal",
    "mass_reweight": "minimal",
    "mass_jk_mc": "mass_jk",
    "mass_jk_data": "mass_jk",
}
HADRONIC_MODES = {"minimal", "minimal_rho", "validation", "full", "mass_jk", "rho_jk", "reweight_pythia_rho"}
# Modes only available for some hadronic channels. reweight_pythia_rho needs a
# Herwig sample to build the h/p splines; that exists for dijet but not trijet.
HADRONIC_CHANNEL_EXCLUDED_MODES = {"trijet": {"reweight_pythia_rho"}}


def normalize_mode_for_channel(mode: str, channel: str) -> str:
    if channel not in ("dijet", "trijet"):
        return mode
    resolved = HADRONIC_MODE_ALIASES.get(mode, mode)
    excluded = HADRONIC_CHANNEL_EXCLUDED_MODES.get(channel, set())
    if resolved not in HADRONIC_MODES or resolved in excluded:
        allowed = sorted(HADRONIC_MODES - excluded)
        raise ValueError(
            f"Hadronic channel '{channel}' does not support mode '{mode}'. "
            f"Choose from {', '.join(allowed)}."
        )
    return resolved


def build_hadronic_fileset(
    directory: str | os.PathLike[str],
    *,
    dataset: str,
    era: str,
    redirector: str = "casa",
    prepend: str = "root://xcache/",
    name_filter: str | None = None,
) -> dict[str, list[str]]:
    """Build a hadronic fileset from samples/hadronic/*.json.

    `name_filter`, if given, keeps only datasets whose DAS name contains that
    substring (e.g. "HT1000to1500") — useful to point a quick test at a high-HT
    bin, since the low-HT bins (HT200to300, ...) yield no events passing the
    dijet/trijet pt>200 / 3-jet selection.
    """
    if dataset not in HADRONIC_FILESETS:
        raise ValueError(
            f"Unsupported hadronic dataset '{dataset}'. "
            f"Choose from {', '.join(sorted(HADRONIC_FILESETS))}."
        )

    selector = HADRONIC_FILESETS[dataset]
    filename = selector.get("casa") if redirector == "casa" else None
    filename = filename or selector["default"]
    json_path = Path(directory) / filename
    if not json_path.exists():
        raise FileNotFoundError(f"Missing hadronic fileset: {json_path}")

    qualifier = None
    if era != "all":
        qualifier = (HADRONIC_ERAS_DATA if dataset == "data" else HADRONIC_ERAS_MC).get(era)
        if qualifier is None:
            raise ValueError(f"Unsupported hadronic era '{era}'")

    with open(json_path) as handle:
        top = json.load(handle)

    fileset = {}
    for _group, datasets in top.items():
        if not isinstance(datasets, dict):
            continue
        for das, files in datasets.items():
            if qualifier and qualifier not in das:
                continue
            if name_filter and name_filter not in das:
                continue
            urls = []
            for path in files:
                if path.startswith("root://"):
                    urls.append(path)
                elif path.startswith("/store/"):
                    urls.append(prepend.rstrip("/") + "/" + path)
                else:
                    urls.append(path)
            if urls:
                fileset[das] = urls

    if not fileset:
        raise FileNotFoundError(
            f"No hadronic files selected from {json_path.name} for era={era}"
        )
    return fileset


def get_processor_class(mode: str, channel: str = "zjet"):
    """Select the processor class for a given channel.

    channel="zjet"   -> QJetMassProcessor (Z+jet, default; mass modes route through
                        the thin zjet_processor_mass wrapper)
    channel="dijet"  -> DijetProcessor   (hadronic dijet, ported from GluonJetMass)
    channel="trijet" -> TrijetProcessor  (hadronic trijet, ported from GluonJetMass)
    """
    mode = normalize_mode_for_channel(mode, channel)
    if channel == "dijet":
        from .dijet_processor import DijetProcessor

        return DijetProcessor

    if channel == "trijet":
        from .trijet_processor import TrijetProcessor

        return TrijetProcessor

    if channel != "zjet":
        raise ValueError(f"Unknown channel '{channel}'. Expected zjet, dijet, or trijet.")

    if mode in _MASS_MODE_ALIASES:
        from .zjet_processor_mass import QJetMassProcessor

        return QJetMassProcessor

    from .zjet_processor import QJetMassProcessor

    return QJetMassProcessor


def resolve_systematics(profile: str):
    if profile == "all_syst":
        return None, None
    if profile == "minimal_syst":
        return NO_SYST_SYSTEMATICS.copy(), MINIMAL_JET_SYSTEMATICS.copy()
    if profile == "no_syst":
        return NO_SYST_SYSTEMATICS.copy(), NO_SYST_SYSTEMATICS.copy()
    raise ValueError(f"Unknown systematic profile: {profile}")


def make_runner(
    use_dask: bool = False,
    client=None,
    workers: int = 1,
    chunksize: int = 200_000,
    maxchunks: int | None = 1,
    skipbadfiles: bool = False,
):
    if use_dask:
        if client is None:
            raise ValueError("use_dask=True but no Dask client provided.")
        executor = processor.DaskExecutor(
            client=client,
            status=True,
            retries=10,
            treereduction=10,
        )
    else:
        executor = processor.FuturesExecutor(
            workers=workers,
            status=True,
            compression=None,
        )

    return processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=chunksize,
        maxchunks=maxchunks,
        skipbadfiles=skipbadfiles,
        xrootdtimeout=120,
    )


def _resolve_executor_mode(
    *,
    executor_mode: str | None,
    casa: bool,
) -> str:
    if executor_mode is not None:
        if executor_mode not in EXECUTOR_MODE_OPTIONS:
            raise ValueError(
                f"Unknown executor_mode '{executor_mode}'. "
                f"Choose from {', '.join(EXECUTOR_MODE_OPTIONS)}."
            )
        return executor_mode
    return "dask-casa" if casa else "futures"


def ensure_client(
    casa: bool,
    test: bool,
    useDefault: bool,
    executor_mode: str | None = None,
):
    from dask.distributed import Client
    from dask.distributed import LocalCluster
    resolved_mode = _resolve_executor_mode(executor_mode=executor_mode, casa=casa)

    if test:
        print("Running locally with 1-2 files (test=True)")

    if resolved_mode == "futures":
        print("Using FuturesExecutor without a Dask client.")
        return None

    if resolved_mode == "dask-local":
        cluster = LocalCluster(
            n_workers=1 if test else 4,
            threads_per_worker=1,
            processes=True,
            silence_logs=logging.ERROR,
        )
        client = Client(cluster)
        print("Created local Dask client.")
        return client

    if resolved_mode == "dask-casa":
        if useDefault:
            client = Client("tls://localhost:8786")
            print("Connected to existing Dask client.")
            return client

        from coffea_casa import CoffeaCasaCluster

        cluster = CoffeaCasaCluster(memory="6 GiB", cores=1)
        # Keep a worker floor: with minimum=0 the cluster scales to zero during the
        # idle gap between preprocessing and processing (heavy/many-file datasets like
        # data), which retires the workers holding the preprocessed data and drops the
        # scheduler<->client comm ("CommClosedError: Scheduler->Client already closed").
        cluster.adapt(minimum=2, maximum=300)
        client = Client(cluster)
        print("Created CoffeaCasaCluster client.")
        return client

    if resolved_mode == "dask-lpc":
        try:
            from lpcjobqueue import LPCCondorCluster
        except ImportError as exc:
            raise ImportError(
                "executor_mode='dask-lpc' requires the lpcjobqueue package. "
                "Use this mode from an LPC environment, or choose "
                "'dask-casa', 'dask-local', or 'futures'."
            ) from exc

        zip_path = make_package_archive()
        cluster = LPCCondorCluster(
            memory="6 GiB",
            transfer_input_files=[str(zip_path)],
            scheduler_options={"dashboard_address": ":8787"},
        )
        cluster.adapt(minimum=1, maximum=100)
        client = Client(cluster)
        print("Created LPCCondorCluster client.")
        return client

    raise ValueError(f"Unsupported executor_mode '{resolved_mode}'")


def make_package_archive(package_dir: Path | None = None) -> Path:
    pkg_dir = package_dir or Path(__file__).resolve().parent
    zip_path = Path("/tmp/smp_jetmass_run2.zip")
    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(zip_path.with_suffix(""), "zip", pkg_dir.parent, pkg_dir.name)
    return zip_path


def upload_package_if_casa(client, casa: bool, package_dir: Path | None = None):
    if client is None:
        return

    zip_path = make_package_archive(package_dir=package_dir)
    client.upload_file(str(zip_path))
    print("Uploaded smp_jetmass_run2.zip to workers.")


def dump_dask_worker_logs(client, *, log=print, max_entries: int = 200):
    """Print recent worker/nanny logs without masking the processing exception."""
    if client is None:
        return

    log("\n===== Dask worker logs after failed test run =====")
    for source, nanny in (("worker", False), ("nanny", True)):
        try:
            logs_by_worker = client.get_worker_logs(n=max_entries, nanny=nanny)
        except Exception as exc:
            log(f"[diagnostic] Could not retrieve Dask {source} logs: {exc}")
            continue

        if not logs_by_worker:
            log(f"[diagnostic] No Dask {source} logs were returned.")
            continue

        for worker, entries in logs_by_worker.items():
            log(f"\n--- {source}: {worker} ---")
            # Dask returns newest entries first; chronological order is easier
            # to read when following the exception and its traceback.
            for entry in reversed(entries):
                if isinstance(entry, (tuple, list)) and len(entry) == 2:
                    level, message = entry
                    log(f"{level}: {message}")
                else:
                    log(str(entry))
    log("===== End Dask worker logs =====\n")


def run_once(
    fileset: dict[str, list[str]],
    *,
    client,
    test: bool,
    data: bool,
    mode: str,
    channel: str = "zjet",
    systematic_profile: str = "all_syst",
    chunksize: int = 100_000,
    chunksize_test: int = 100_000,
    executor_mode: str | None = None,
):
    print("Running over:", list(fileset.keys())[:10], "..." if len(fileset) > 10 else "")
    mode = normalize_mode_for_channel(mode, channel)
    systematics, jet_systematics = resolve_systematics(systematic_profile)
    if executor_mode is None:
        use_dask = client is not None
    else:
        resolved_mode = _resolve_executor_mode(executor_mode=executor_mode, casa=False)
        use_dask = resolved_mode != "futures"

    if test:
        first_key = list(fileset.keys())[0]
        fileset = {first_key: [fileset[first_key][0]]}
        print("Running over test files:", list(fileset.keys()))
        run = make_runner(
            use_dask=use_dask,
            client=client,
            chunksize=chunksize_test,
            maxchunks=1,
        )
        debug = True
    else:
        print("Running over full dataset")
        run = make_runner(
            use_dask=use_dask,
            client=client,
            chunksize=chunksize,
            maxchunks=None,
        )
        debug = False

    processor_cls = get_processor_class(mode, channel=channel)

    start = time.time()
    out = run(
        fileset,
        processor_cls(
            do_gen=not data,
            debug=debug,
            systematics=systematics,
            jet_systematics=jet_systematics,
            mode=mode,
        ),
        treename="Events",
    )
    print(f"Done. time taken {format_time(time.time() - start)}")
    return out


def save_output(out, fout: str | os.PathLike[str]):
    output_path = Path(fout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(out, handle)
    size = output_path.stat().st_size
    unit = "kB" if size < 1e6 else "MB"
    value = size / (1e3 if unit == "kB" else 1e6)
    print(f"Output written to {output_path} with size {value:.1f} {unit}")


def default_output_dir(repo_root: Path | None = None) -> Path:
    return get_analysis_paths(repo_root).repo_root / "outputs"


def make_output_filename(
    data: bool,
    dataset: str,
    tag: str,
    mode: str | None = None,
    channel: str | None = None,
    test: bool = False,
    output_dir: str | os.PathLike[str] | None = None,
) -> str:
    base = "data" if data else dataset
    if channel and channel != "zjet":
        base = f"{channel}_{base}"
    mode_token = ""
    if mode:
        output_mode = OUTPUT_MODE_TAGS.get(mode, mode)
        safe_mode = "".join(ch if ch.isalnum() else "_" for ch in output_mode).strip("_")
        if safe_mode:
            mode_token = f"{safe_mode}_"
    filename = f"{mode_token}{base}_{tag}{'_TEST' if test else ''}.pkl"
    base_output_dir = Path(output_dir) if output_dir is not None else default_output_dir()
    return str(base_output_dir / filename)


def get_group_tag(index: int, era: str, group_mode: str) -> str:
    if group_mode == "per_group":
        if era == "all":
            era_tags = ["2016", "2016APV", "2017", "2018"]
            return era_tags[index] if index < len(era_tags) else f"group{index}"
        return era
    return era if era != "all" else "all"


def run_from_config(cfg, *, client=None, repo_root=None, log=print):
    """Build fileset(s) for cfg, run the matching channel processor, save pickles
    under outputs/, and return (output_paths, last_out).

    Single source of truth shared by scripts/run_analysis_cli.py and
    notebooks/run_analysis.ipynb. `cfg` should already be validated
    (validate_analysis_config). If `client` is None one is created from cfg and
    closed at the end; if a client is passed it is reused and left open (so a
    notebook can keep one dask client across runs).
    """
    import dask

    cfg = validate_analysis_config(cfg)
    paths = get_analysis_paths(repo_root)
    samplePath = SamplePath(cfg["era"])
    prependstr = cfg["prependstr"]
    dataset = cfg["dataset"]
    group_mode = cfg["group_mode"]

    NanoAODSchema.warn_missing_crossrefs = False
    dask.config.set({
        "distributed.logging.distributed": "error",
        "distributed.logging.bokeh": "error",
        "distributed.logging.tornado": "error",
    })

    log("Configuration:")
    for key in sorted(cfg):
        log(f"  {key} = {cfg[key]}")

    own_client = client is None
    if own_client:
        client = ensure_client(
            casa=cfg["casa"], test=cfg["test"],
            useDefault=cfg["useDefault"], executor_mode=cfg["executor_mode"],
        )
        upload_package_if_casa(client, casa=cfg["casa"])

    outputs: list[str] = []
    last_out = None

    def _run_and_save(fileset, index):
        nonlocal last_out
        out = run_once(
            fileset, client=client, test=cfg["test"], data=dataset == "data",
            mode=cfg["mode"], channel=cfg["channel"],
            systematic_profile=cfg["systematic_profile"],
            chunksize=cfg["chunksize"], chunksize_test=cfg["chunksize_test"],
            executor_mode=cfg["executor_mode"],
        )
        tag = get_group_tag(index, cfg["era"], group_mode)
        fout = make_output_filename(
            data=dataset == "data", dataset=dataset, tag=tag, mode=cfg["mode"],
            channel=cfg["channel"], test=cfg["test"],
            output_dir=paths.repo_root / "outputs",
        )
        save_output(out, fout)
        log(f"[{index + 1}] Saved: {fout}")
        last_out = out
        outputs.append(fout)

    try:
        if cfg["channel"] in ("dijet", "trijet"):
            fileset = build_hadronic_fileset(
                paths.samples_hadronic_dir, dataset=dataset, era=cfg["era"],
                redirector=cfg["redirector"], prepend=prependstr,
                name_filter=(cfg.get("dataset_filter") or None),
            )
            _run_and_save(fileset, 0)
        elif dataset == "data":
            for i, group in enumerate(iter_groups(samplePath.data, group_mode)):
                _run_and_save(build_fileset_from_txts(
                    group, paths.samples_data_dir, prependstr, split_ht=False), i)
        elif dataset == "pythia":
            for i, group in enumerate(iter_groups(samplePath.pythia, group_mode)):
                _run_and_save(build_fileset_from_txts(
                    group, paths.samples_mc_dir, prependstr,
                    split_ht=True, ht_bins=HT_BINS), i)
        elif dataset == "pythia_local":
            _run_and_save(build_local_pythia_fileset(
                paths.samples_mc_local_dir, cfg["era"]), 0)
        elif dataset in ("nlo", "pythia2"):
            # NLO (amcatnloFXFX) DY inclusive, era-aware like pythia/herwig.
            for i, group in enumerate(iter_groups(samplePath.nlo, group_mode)):
                _run_and_save(build_fileset_from_txts(
                    group, paths.samples_mc_dir, prependstr, split_ht=False), i)
        elif dataset == "nlo_ptz":
            # PtZ-binned NLO (amcatnloFXFX, MatchEWPDG20) DY; each of the 6 ptZ
            # bins becomes its own dataset, normalized to its own xs in postprocess.
            for i, group in enumerate(iter_groups(samplePath.nlo_ptz, group_mode)):
                _run_and_save(build_fileset_from_txts(
                    group, paths.samples_mc_dir, prependstr, split_ht=False), i)
        elif dataset == "herwig":
            for i, group in enumerate(iter_groups(samplePath.herwig, group_mode)):
                _run_and_save(build_fileset_from_txts(
                    group, paths.samples_mc_dir, prependstr, split_ht=False), i)
        elif dataset == "powheg":
            _run_and_save(build_fileset_from_txts(
                ["powheg_UL18NanoAODv9_inclusive.txt"], paths.samples_mc_dir,
                prependstr, split_ht=False), 0)
        elif dataset == "st":
            _run_and_save(build_fileset_from_txts(
                ST_FILES, paths.samples_mc_dir, prependstr, split_ht=False), 0)
        elif dataset == "backgrounds":
            _run_and_save(build_backgrounds_fileset(paths.samples_bkg_dir, prependstr), 0)
        else:
            log(f"Dataset is {dataset} and it is not in the list")
    except Exception:
        resolved_mode = _resolve_executor_mode(
            executor_mode=cfg["executor_mode"],
            casa=cfg["casa"],
        )
        if cfg["test"] and resolved_mode != "futures":
            dump_dask_worker_logs(client, log=log)
        raise
    finally:
        if own_client and client is not None:
            client.close()

    log(f"Number of group outputs: {len(outputs)}")
    return outputs, last_out


def event_id_set(out):
    """Set of (run, lumi, event) tuples logged for finally-selected DATA events.

    The dijet/trijet (and, when added, zjet) data processors record an `event_id`
    accumulator. Returns an empty set if the output has none (e.g. an MC run).
    """
    eid = out.get("event_id") if hasattr(out, "get") else None
    if eid is None:
        return set()
    run = eid["run"].value
    lumi = eid["luminosityBlock"].value
    evt = eid["event"].value
    return set(zip(run.tolist(), lumi.tolist(), evt.tolist()))


def channel_overlap(outs: dict):
    """Pairwise orthogonality report for DATA outputs.

    `outs` maps a channel label -> processor output. Returns a dict with the
    per-channel final-event counts and the size of each pairwise intersection
    (0 everywhere == fully orthogonal selections).
    """
    import itertools
    sets = {k: event_id_set(v) for k, v in outs.items()}
    report = {f"n[{k}]": len(s) for k, s in sets.items()}
    for a, b in itertools.combinations(sets, 2):
        report[f"overlap[{a}&{b}]"] = len(sets[a] & sets[b])
    return report
