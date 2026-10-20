import json

from smp_jetmass_run2.notebook_utils import build_hadronic_fileset


def test_casa_rewrites_legacy_redirector_url_to_xcache(tmp_path):
    fileset = {
        "QCD_binned": {
            (
                "/QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7/"
                "RunIISummer20UL18NanoAODv9-campaign/NANOAODSIM"
            ): [
                (
                    "root://cmsxrootd.fnal.gov//store/mc/"
                    "RunIISummer20UL18NanoAODv9/file.root"
                )
            ]
        }
    }
    (tmp_path / "fileset_HERWIG_wRedirs.json").write_text(
        json.dumps(fileset)
    )

    selected = build_hadronic_fileset(
        tmp_path,
        dataset="herwig",
        era="2018",
        redirector="casa",
        prepend="root://xcache/",
    )

    assert next(iter(selected.values())) == [
        "root://xcache//store/mc/RunIISummer20UL18NanoAODv9/file.root"
    ]


def test_exclude_drops_matching_datasets(tmp_path):
    fileset = {
        "QCD_binned": {
            f"/QCD_{ht}_TuneCP5_13TeV-madgraphMLM-pythia8/"
            "RunIISummer20UL18NanoAODv9-campaign/NANOAODSIM": [f"/store/mc/{ht}.root"]
            for ht in ("HT100to200", "HT200to300")
        }
    }
    (tmp_path / "fileset_MG_pythia8.json").write_text(json.dumps(fileset))
    (tmp_path / "fileset_MG_pythia8_wRedirs.json").write_text(json.dumps(fileset))

    selected = build_hadronic_fileset(
        tmp_path, dataset="mg_pythia8", era="2018", exclude=["QCD_HT100to200"],
    )

    assert [das.split("/")[1].split("_Tune")[0] for das in selected] == ["QCD_HT200to300"]
