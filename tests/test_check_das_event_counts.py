"""Offline checks of the DAS event-count gate: filename -> dataset
resolution against the shipped filesets, and cutflow extraction for both the
dijet (per-dataset keys) and trijet (anonymous counter) conventions.
SSH responses are mocked; these tests never connect to CERN.
"""

from collections import defaultdict
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from tools.check_das_event_counts import (BackendDown, era_of_dataset, extract_reads,
                                          is_data_dataset, query_lxplus,
                                          resolve_datasets)


def test_data_vs_mc_campaign_classification():
    # every MC campaign ALSO starts with "Run" (RunIISummer20...) -- only the
    # era pattern Run20XX[A-H] means real data
    assert is_data_dataset(
        "/JetHT/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD")
    assert is_data_dataset(
        "/JetHT/Run2016B-ver1_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
    assert not is_data_dataset(
        "/QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8/"
        "RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/"
        "NANOAODSIM")
    assert not is_data_dataset(
        "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/"
        "RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAOD")


def test_dijet_style_cutflow_names_its_datasets():
    das = "/QCD_HT300to500_X/RunIISummer20UL18NanoAODv9-v1/NANOAODSIM"
    out = {"cutflow": {"QCD_pythiaMG2018": {
        f"nEvents initial {das}": 100,
        "nEvents initial (all datasets)": 100,
        "nEvents initial nominal": 100,
    }}}
    per_dataset, anonymous = extract_reads(out)
    assert per_dataset == {das: 100}
    assert anonymous is None


def test_trijet_style_cutflow_is_anonymous_and_summed_across_groups():
    out = {"cutflow": {
        "pythiaMG2018": defaultdict(int, {"nEvents initial": 70}),
        "pythiaMG2017": defaultdict(int, {"nEvents initial": 30}),
        "trigger_init": defaultdict(int),
    }}
    per_dataset, anonymous = extract_reads(out)
    assert per_dataset == {}
    assert anonymous == 100


@pytest.mark.parametrize("name, era, expect", [
    ("rho_skim_trijet_mg_pythia8_2018_HT300to500.pkl", "2018",
     ["/QCD_HT300to500_"]),
    ("rho_skim_trijet_mg_pythia8_2016APV_HT100to200.pkl", "2016APV",
     ["/QCD_HT100to200_"]),
    ("rho_skim_trijet_data_2016_Run2016F-UL2016.pkl", "2016",
     ["/JetHT/Run2016F-UL2016_"]),
    ("rho_skim_trijet_data_2016APV_Run2016B-ver1.pkl", "2016APV",
     ["/JetHT/Run2016B-ver1_"]),
])
def test_binned_filename_resolves_to_single_dataset(name, era, expect):
    datasets = resolve_datasets(Path(name))
    assert len(datasets) == 1
    assert datasets[0].startswith(expect[0])
    assert era_of_dataset(datasets[0]) == era


def test_era_wide_filename_resolves_to_whole_era():
    datasets = resolve_datasets(Path("minimal_rho_trijet_mg_pythia8_2018.pkl"))
    assert len(datasets) == 8
    assert all(era_of_dataset(ds) == "2018" for ds in datasets)
    datasets = resolve_datasets(Path("minimal_rho_trijet_data_2016APV.pkl"))
    assert len(datasets) == 6
    assert all("HIPM" in ds for ds in datasets)


def test_unresolvable_filenames_raise():
    with pytest.raises(ValueError):
        resolve_datasets(Path("no_era_token_here.pkl"))
    with pytest.raises(ValueError):
        resolve_datasets(Path("minimal_rho_trijet_sherpa_2018.pkl"))


def test_lxplus_count_uses_noninteractive_ssh():
    dataset = "/QCD_X/RunIISummer20UL18_X/NANOAODSIM"
    response = subprocess.CompletedProcess(
        [], 0, '[{"summary": [{"nevents": 123}]}]', ""
    )
    with patch("tools.check_das_event_counts.subprocess.run", return_value=response) as run:
        assert query_lxplus(dataset, timeout=7) == 123
    command = run.call_args.args[0]
    assert command[:4] == ["ssh", "-o", "BatchMode=yes", "lxplus"]
    assert command[4] == f"dasgoclient -query 'summary dataset={dataset}' -json"
    assert run.call_args.kwargs["stdin"] == subprocess.DEVNULL
    assert run.call_args.kwargs["timeout"] == 7


def test_lxplus_auth_failure_is_unavailable_not_zero_events():
    response = subprocess.CompletedProcess([], 255, "", "Permission denied")
    with patch("tools.check_das_event_counts.subprocess.run", return_value=response):
        with pytest.raises(BackendDown, match="Permission denied"):
            query_lxplus("/QCD_X/campaign/NANOAODSIM", timeout=7)


def test_lxplus_timeout_is_unavailable_not_zero_events():
    with patch("tools.check_das_event_counts.subprocess.run",
               side_effect=subprocess.TimeoutExpired("ssh", 7)):
        with pytest.raises(BackendDown, match="timeout"):
            query_lxplus("/QCD_X/campaign/NANOAODSIM", timeout=7)


@pytest.mark.parametrize("stdout", ["not JSON", "[]", '[{"summary": [{}]}]'])
def test_lxplus_missing_count_does_not_become_zero(stdout):
    response = subprocess.CompletedProcess([], 0, stdout, "")
    with patch("tools.check_das_event_counts.subprocess.run", return_value=response):
        assert query_lxplus("/QCD_X/campaign/NANOAODSIM", timeout=7) is None
