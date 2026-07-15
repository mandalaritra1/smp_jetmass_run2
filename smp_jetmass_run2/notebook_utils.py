import os
import json
import pickle
import shutil
import time
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

# LCG_110's jupyter_client signs every kernel message with datetime.utcnow(),
# which python 3.13 deprecates -- one warning PER MESSAGE floods notebook cell
# output. Harness noise, not analysis code; silence it.
warnings.filterwarnings(
    "ignore", message=".*utcnow.*", category=DeprecationWarning
)
# The filter alone does not stick (libraries in the run path reset warning
# filters), so also fix the emitter: replace jupyter_client's deprecated
# utcnow() with the timezone-aware equivalent it is warning about.
try:
    import datetime as _datetime
    import jupyter_client.session as _jc_session

    def _utcnow_tz_aware():
        return _datetime.datetime.now(_datetime.timezone.utc)

    _jc_session.utcnow = _utcnow_tz_aware
except Exception:  # no jupyter_client outside notebook kernels -- fine
    pass

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
    "dask-lxplus",
    # Purdue AF via Dask Gateway (see ensure_client): login works, but the
    # worker software env is unresolved -- NOT yet usable, kept as a stub.
    "dask-purdue",
]
REDIRECTOR_PREPENDS = {
    "local": "",
    "lpc": "root://cmsxrootd.fnal.gov/",
    "casa": "root://xcache/",
    # CERN/lxplus: CMS global redirector. Swap for the closer Europe redirector
    # root://xrootd-cms.infn.it/ if most inputs live on EU sites.
    "lxplus": "root://cms-xrd-global.cern.ch/",
    # AAA fallbacks for facilities without a local xcache (Purdue/INFN AF):
    # global redirector, and the EU redirector which is closer to INFN sites.
    "global": "root://cms-xrd-global.cern.ch/",
    "eu": "root://xrootd-cms.infn.it/",
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

MINIMAL_JET_SYSTEMATICS = ["nominal", "JERUp", "JERDown",
                           "JMSUp", "JMSDown", "JMRUp", "JMRDown"]
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
        return [
            line.strip()
            for line in handle.readlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]


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
    treereduce: bool = False,
    worker_merge: int = 1,
):
    if use_dask:
        if client is None:
            raise ValueError("use_dask=True but no Dask client provided.")
        if treereduce:
            # coffea DaskExecutor merges chunk outputs on the workers (tree
            # reduction) -- the partial sums of many-axis histograms can OOM
            # small condor workers at the end of a run.
            executor = processor.DaskExecutor(
                client=client,
                status=True,
                retries=10,
                treereduction=10,
            )
        else:
            # Default: stream each finished chunk back and merge on the client
            # (peak = running total + one chunk; no worker-side merge spike).
            # The client still has to hold the full accumulated output.
            from .streaming_executor import StreamingDaskExecutor

            executor = StreamingDaskExecutor(
                client=client,
                status=True,
                retries=10,
                # >1 merges that many consecutive chunks into one worker task,
                # cutting client-bound traffic by the same factor (big win for
                # all-syst runs whose per-chunk hist output is large).
                worker_merge=worker_merge,
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


def normalize_worker_memory(worker_memory):
    """A bare number ('4', 4, 4.0) means GiB. Without this, dask parses a
    unitless string as BYTES -- CernCluster(memory='4') spawns workers with
    `--memory-limit 4.0B`, which pause instantly and hang the whole run."""
    if worker_memory is None:
        return None
    if isinstance(worker_memory, (int, float)):
        return f"{worker_memory} GiB"
    s = str(worker_memory).strip()
    try:
        return f"{float(s)} GiB"
    except ValueError:
        return s


def ensure_client(
    casa: bool,
    test: bool,
    useDefault: bool,
    executor_mode: str | None = None,
    worker_memory: str | None = None,
    n_workers: int | None = None,
):
    """worker_memory overrides the per-worker memory request (default 6 GiB)
    on the casa/LPC/lxplus clusters. With the streaming executor (no
    worker-side merges) workers only need chunk-processing headroom, so
    e.g. "2 GiB" is viable for no-syst runs.

    n_workers requests a FIXED pool (cluster.scale) on the lxplus/LPC batch
    clusters instead of adaptive scaling -- recommended for production:
    HTCondor's slow worker startup makes adapt flap and can leave a
    single-worker straggler tail."""
    worker_memory = normalize_worker_memory(worker_memory)
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

        cluster = CoffeaCasaCluster(memory=worker_memory or "6 GiB", cores=1)
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
            memory=worker_memory or "6 GiB",
            transfer_input_files=[str(zip_path)],
            scheduler_options={"dashboard_address": ":8787"},
        )
        if n_workers:
            cluster.scale(n_workers)
        else:
            cluster.adapt(minimum=1, maximum=100)
        client = Client(cluster)
        print("Created LPCCondorCluster client.")
        return client

    if resolved_mode == "dask-lxplus":
        try:
            from dask_lxplus import CernCluster
        except ImportError as exc:
            raise ImportError(
                "executor_mode='dask-lxplus' requires the dask-lxplus package "
                "(pip install dask-lxplus). Use this mode from an lxplus/CERN "
                "HTCondor submit node, or choose 'dask-casa', 'dask-local', or "
                "'futures'."
            ) from exc

        import socket

        # lcg=True ships the *submitting* environment (the LCG view you sourced
        # on lxplus, which must provide coffea/dask/awkward) to the batch workers;
        # container_runtime='none' means run directly in that view rather than a
        # Singularity/Docker image. The analysis package itself is shipped
        # separately via client.upload_file (see upload_package_if_casa).
        log_dir = Path(
            os.environ.get(
                "DASK_LXPLUS_LOG_DIR", str(Path.home() / "dask_lxplus_logs")
            )
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        job_flavour = os.environ.get("DASK_LXPLUS_JOB_FLAVOUR", "longlunch")

        cluster = CernCluster(
            cores=1,
            memory=worker_memory or "6 GiB",
            disk="10 GiB",
            death_timeout="60",
            lcg=True,
            nanny=False,
            container_runtime="none",
            log_directory=str(log_dir),
            scheduler_options={
                # HTCondor workers dial back to the scheduler, so it must bind to
                # a routable host/port (not localhost). gethostname() resolves to
                # the lxplus node's reachable name inside the CERN batch network.
                "port": 8786,
                "host": socket.gethostname(),
                "dashboard_address": ":8787",
            },
            job_extra={
                # HTCondor max walltime tier: espresso(20m)/microcentury(1h)/
                # longlunch(2h)/workday(8h)/tomorrow(1d)/testmatch(3d)/nextweek(1w).
                "MY.JobFlavour": f'"{job_flavour}"',
                # Ship the grid proxy to each worker and export X509_USER_PROXY in
                # its env. Without this the worker boots with no proxy, so it can't
                # authenticate to any xrootd server and every /store read fails with
                # "[3011] No servers are available to read the file" (which looks
                # like a redirector problem but is really missing auth). Requires a
                # valid proxy at submit time: voms-proxy-init -voms cms.
                "use_x509userproxy": "true",
            },
        )
        # Adaptive scaling flaps on HTCondor (minutes of worker startup
        # latency + eager retirement as the queue drains -> single-worker
        # tails). For production, request a fixed pool: n_workers config key
        # (the notebook's "lx workers" widget) or DASK_LXPLUS_WORKERS env.
        n_fixed = n_workers or os.environ.get("DASK_LXPLUS_WORKERS")
        if n_fixed:
            cluster.scale(int(n_fixed))
            scale_msg = f"fixed scale {n_fixed} workers"
        else:
            cluster.adapt(
                minimum=int(os.environ.get("DASK_LXPLUS_MIN_WORKERS", "1")),
                maximum=int(os.environ.get("DASK_LXPLUS_MAX_WORKERS", "100")),
            )
            scale_msg = "adaptive 1-100 workers"
        client = Client(cluster)
        print(
            f"Created CernCluster (lxplus) client "
            f"[JobFlavour={job_flavour}, {scale_msg}, logs={log_dir}]."
        )
        return client

    if resolved_mode == "dask-purdue":
        # Purdue Analysis Facility (cms.geddes.rcac.purdue.edu), Dask Gateway
        # backend. STATUS 2026-07-15: login works (CERN account) and this code
        # follows their docs, but the mode is NOT yet usable: the facility's
        # global pixi env did not actually provide coffea when we tried it, so
        # the worker software environment still has to be figured out --
        # probably a project pixi env in this repo pinning coffea==2026.5.0,
        # passed via PURDUE_AF_PIXI_PROJECT. Left in place as the starting
        # point for that work. Docs:
        # https://purdue-af.readthedocs.io/en/latest/guide-dask-gateway.html
        try:
            from dask_gateway import Gateway
        except ImportError as exc:
            raise ImportError(
                "executor_mode='dask-purdue' requires the dask-gateway package "
                "and is meant to run inside a Purdue AF JupyterLab session "
                "(https://cms.geddes.rcac.purdue.edu/hub)."
            ) from exc

        from dask.utils import parse_bytes

        # Inside the AF, Gateway() picks up address/auth from the session env.
        gateway = Gateway()
        cluster_kwargs = {
            "worker_cores": 1,
            # Gateway expects a number in GB, not a "6 GiB" string.
            "worker_memory": parse_bytes(worker_memory or "4 GiB") / 2**30,
            # Forward the session environment (tokens, X509 proxy path).
            "env": dict(os.environ),
        }
        # Workers import software from a shared pixi/conda env, NOT from an
        # image. The global env (/work/pixi/global) had coffea 2025.12.0 when
        # this was written -- older than the 2026.5.0 this repo is validated
        # on -- so prefer pointing PURDUE_AF_PIXI_PROJECT at a project env
        # that pins coffea. The analysis package itself still arrives via
        # client.upload_file (upload_package_if_casa).
        pixi_project = os.environ.get("PURDUE_AF_PIXI_PROJECT")
        conda_env = os.environ.get("PURDUE_AF_CONDA_ENV")
        if pixi_project:
            cluster_kwargs["pixi_project"] = pixi_project
        elif conda_env:
            cluster_kwargs["conda_env"] = conda_env
        cluster = gateway.new_cluster(**cluster_kwargs)
        # Same worker floor rationale as dask-casa (see above).
        cluster.adapt(minimum=2, maximum=int(os.environ.get("PURDUE_AF_MAX_WORKERS", "200")))
        client = cluster.get_client()
        print("Created Purdue AF Dask Gateway client.")
        return client

    # (An INFN CMS AF backend via dask_remote_jobqueue existed briefly; removed
    # because the facility requires registration we don't have. See git history
    # -- "Add dask-purdue and dask-infn executor modes" -- if it becomes
    # relevant again.)

    raise ValueError(f"Unsupported executor_mode '{resolved_mode}'")


def make_package_archive(package_dir: Path | None = None) -> Path:
    pkg_dir = package_dir or Path(__file__).resolve().parent
    zip_path = Path("/tmp/smp_jetmass_run2.zip")
    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(zip_path.with_suffix(""), "zip", pkg_dir.parent, pkg_dir.name)
    return zip_path


def upload_package_if_casa(client, casa: bool, package_dir: Path | None = None):
    # Ships the analysis package zip to every worker over the scheduler
    # (client.upload_file adds it to each worker's sys.path). This covers
    # dask-casa and dask-lxplus (CernCluster), neither of which has lpcjobqueue's
    # per-job transfer_input_files hook; dask-lpc additionally transfers it via
    # the cluster, but the redundant upload here is harmless.
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
    reweight_source: str = "herwig",
    rho_refine: int | None = None,
    worker_merge: int = 1,
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
            skipbadfiles=True,
            worker_merge=worker_merge,
        )
        debug = True
    else:
        print("Running over full dataset")
        run = make_runner(
            use_dask=use_dask,
            client=client,
            chunksize=chunksize,
            maxchunks=None,
            skipbadfiles=True,
            worker_merge=worker_merge,
        )
        debug = False

    processor_cls = get_processor_class(mode, channel=channel)

    proc_kwargs = dict(
        do_gen=not data,
        debug=debug,
        systematics=systematics,
        jet_systematics=jet_systematics,
        mode=mode,
    )
    # reweight_source / rho_refine are only zjet knobs
    if channel == "zjet":
        proc_kwargs["reweight_source"] = reweight_source
        proc_kwargs["rho_refine"] = rho_refine

    start = time.time()
    out = run(
        fileset,
        processor_cls(**proc_kwargs),
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
    reweight_source: str = "herwig",
    rho_refine: int | None = None,
) -> str:
    base = "data" if data else dataset
    if channel and channel != "zjet":
        base = f"{channel}_{base}"
    # keep alternate-generator reweights in distinct files (Herwig is the default)
    if reweight_source and reweight_source != "herwig":
        base = f"{base}_{reweight_source}"
    # distinct file per rho-axis refinement (2x -> _r2, 4x -> _r4)
    if rho_refine and rho_refine != 1:
        base = f"{base}_r{rho_refine}"
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
            worker_memory=cfg.get("worker_memory"),
            n_workers=int(cfg.get("n_workers") or 0) or None,
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
            reweight_source=cfg.get("reweight_source", "herwig"),
            rho_refine=cfg.get("rho_refine", None),
            worker_merge=cfg.get("worker_merge", 1),
        )
        tag = get_group_tag(index, cfg["era"], group_mode)
        fout = make_output_filename(
            data=dataset == "data", dataset=dataset, tag=tag, mode=cfg["mode"],
            channel=cfg["channel"], test=cfg["test"],
            output_dir=paths.repo_root / "outputs",
            reweight_source=cfg.get("reweight_source", "herwig"),
            rho_refine=cfg.get("rho_refine", None),
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
            # client.close() only disconnects -- the cluster (and its claimed
            # batch slots, e.g. a fixed n_workers pool on lxplus) survives in
            # this process until closed. client.cluster is None for clients
            # connected by address (useDefault -> casa's prespawned cluster,
            # which must NOT be torn down here).
            cluster = getattr(client, "cluster", None)
            client.close()
            if cluster is not None:
                cluster.close()

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
