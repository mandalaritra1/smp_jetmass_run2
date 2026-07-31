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
