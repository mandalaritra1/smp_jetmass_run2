"""The overlap script copies each processor's _SKIM_CUTS bit order instead of
importing coffea; this pins the copy to the source of truth."""

from scripts.channel_overlap import DIJET_CUTS, TRIJET_CUTS


def test_cut_tuples_match_processors():
    from smp_jetmass_run2.dijet_processor import DijetProcessor
    from smp_jetmass_run2.trijet_processor import TrijetProcessor

    assert DIJET_CUTS == DijetProcessor._SKIM_CUTS
    assert TRIJET_CUTS == TrijetProcessor._SKIM_CUTS
