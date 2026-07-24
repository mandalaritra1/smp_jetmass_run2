import gzip
import json
from contextlib import ExitStack
from functools import lru_cache
from importlib.resources import files, as_file
from pathlib import Path
from typing import Optional

import numpy as np
import awkward as ak
import correctionlib
import tempfile
from coffea.lumi_tools import LumiMask
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools import extractor

from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from .roccor import RoccoR

# Map your IOV to the key used inside the JSON
_HNAME = {
    "2016APV": "Collisions16_UltraLegacy_goldenJSON",
    "2016"   : "Collisions16_UltraLegacy_goldenJSON",
    "2017"   : "Collisions17_UltraLegacy_goldenJSON",
    "2018"   : "Collisions18_UltraLegacy_goldenJSON",
}

_CORRLIB_NAME_MAP = {
    "2016APV": "2016preVFP_UL",
    "2016"   : "2016postVFP_UL",
    "2017"   : "2017_UL",
    "2018"   : "2018_UL",
}

_ELE_SF_YEAR_LABEL = {
    "2016APV": "2016preVFP",
    "2016"   : "2016postVFP",
    "2017"   : "2017",
    "2018"   : "2018",
}

_JME_MODE_FILE = {
    "AK8": "fatJet_jerc.json.gz",
    "AK4": "jet_jerc.json.gz",
}

_JME_AK_LABEL = {
    "AK8": "AK8PFPuppi",
    "AK4": "AK4PFchs",
}

_JEC_DATA_TAGS = {
    "2016APV": {
        "Run2016B": "Summer19UL16APV_RunBCD_V7_DATA",
        "Run2016C": "Summer19UL16APV_RunBCD_V7_DATA",
        "Run2016D": "Summer19UL16APV_RunBCD_V7_DATA",
        "Run2016E": "Summer19UL16APV_RunEF_V7_DATA",
        "Run2016F": "Summer19UL16APV_RunEF_V7_DATA",
    },
    "2016": {
        "Run2016F": "Summer19UL16_RunFGH_V7_DATA",
        "Run2016G": "Summer19UL16_RunFGH_V7_DATA",
        "Run2016H": "Summer19UL16_RunFGH_V7_DATA",
    },
    "2017": {
        "Run2017B": "Summer19UL17_RunB_V5_DATA",
        "Run2017C": "Summer19UL17_RunC_V5_DATA",
        "Run2017D": "Summer19UL17_RunD_V5_DATA",
        "Run2017E": "Summer19UL17_RunE_V5_DATA",
        "Run2017F": "Summer19UL17_RunF_V5_DATA",
    },
    "2018": {
        "Run2018A": "Summer19UL18_RunA_V5_DATA",
        "Run2018B": "Summer19UL18_RunB_V5_DATA",
        "Run2018C": "Summer19UL18_RunC_V5_DATA",
        "Run2018D": "Summer19UL18_RunD_V5_DATA",
    },
}


def _resolve_jec_tags(iov: str):
    if iov == "2018":
        return (
            "Summer19UL18_V5_MC",
            {
                "Run2018A": "Summer19UL18_RunA_V6_DATA",
                "Run2018B": "Summer19UL18_RunB_V6_DATA",
                "Run2018C": "Summer19UL18_RunC_V6_DATA",
                "Run2018D": "Summer19UL18_RunD_V6_DATA",
            },
            "Summer19UL18_JRV2_MC",
        )
    if iov == "2017":
        return (
            "Summer19UL17_V6_MC",  # V6 for all (user decision); V6 MC JEC bundled
            {
                "Run2017B": "Summer19UL17_RunB_V6_DATA",
                "Run2017C": "Summer19UL17_RunC_V6_DATA",
                "Run2017D": "Summer19UL17_RunD_V6_DATA",
                "Run2017E": "Summer19UL17_RunE_V6_DATA",
                "Run2017F": "Summer19UL17_RunF_V6_DATA",
            },
            "Summer19UL17_JRV3_MC",
        )
    if iov == "2016":
        return (
            "Summer19UL16_V7_MC",
            {
                "Run2016F": "Summer19UL16_RunFGH_V7_DATA",
                "Run2016G": "Summer19UL16_RunFGH_V7_DATA",
                "Run2016H": "Summer19UL16_RunFGH_V7_DATA",
            },
            "Summer20UL16_JRV3_MC",
        )
    if iov == "2016APV":
        return (
            "Summer19UL16APV_V7_MC",
            {
                "Run2016B": "Summer19UL16APV_RunBCD_V7_DATA",
                "Run2016C": "Summer19UL16APV_RunBCD_V7_DATA",
                "Run2016D": "Summer19UL16APV_RunBCD_V7_DATA",
                "Run2016E": "Summer19UL16APV_RunEF_V7_DATA",
                "Run2016F": "Summer19UL16APV_RunEF_V7_DATA",
            },
            "Summer20UL16APV_JRV3_MC",
        )
    raise ValueError(f"Unsupported IOV '{iov}' for jet energy corrections")

class PtVarWeighter:
    def __init__(self, npz_path: str, grid_key: str):
        dat = np.load(npz_path, allow_pickle=True)
        self.pt_edges = np.asarray(dat["pt_edges"], dtype=float)
        self.var_grids = dat[grid_key]
        self.w_grids = dat["w_grids"]

    def _pt_bin(self, pt: float) -> int:
        k = np.searchsorted(self.pt_edges, pt, side="right") - 1
        return int(np.clip(k, 0, len(self.pt_edges) - 2))

    def weight(self, pt: float, value: float) -> float:
        k = self._pt_bin(pt)
        x = np.asarray(self.var_grids[k], dtype=float)
        w = np.asarray(self.w_grids[k], dtype=float)
        return float(np.interp(value, x, w, left=w[0], right=w[-1]))

    def weight_array(self, pt_arr, value_arr):
        pt_arr = np.asarray(pt_arr, dtype=float)
        value_arr = np.asarray(value_arr, dtype=float)
        out = np.empty_like(value_arr, dtype=float)

        k_arr = np.searchsorted(self.pt_edges, pt_arr, side="right") - 1
        k_arr = np.clip(k_arr, 0, len(self.pt_edges) - 2)

        for k in range(len(self.pt_edges) - 1):
            mask = (k_arr == k)
            if not np.any(mask):
                continue
            x = np.asarray(self.var_grids[k], dtype=float)
            w = np.asarray(self.w_grids[k], dtype=float)
            out[mask] = np.interp(value_arr[mask], x, w, left=w[0], right=w[-1])

        return out


PtRhoWeighter = PtVarWeighter


class PtBinnedVarWeighter:
    def __init__(self, npz_path: str):
        dat = np.load(npz_path, allow_pickle=True)
        self.pt_edges = np.asarray(dat["pt_edges"], dtype=float)
        self.var_edges = dat["rho_edges"]
        self.w_grids = dat["w_grids"]

    def weight_array(self, pt_arr, value_arr):
        pt_arr = np.asarray(pt_arr, dtype=float)
        value_arr = np.asarray(value_arr, dtype=float)
        out = np.ones_like(value_arr, dtype=float)

        finite = np.isfinite(pt_arr) & np.isfinite(value_arr)
        if not np.any(finite):
            return out

        pt_bin = np.searchsorted(self.pt_edges, pt_arr[finite], side="right") - 1
        pt_bin = np.clip(pt_bin, 0, len(self.pt_edges) - 2)
        finite_indices = np.flatnonzero(finite)

        for k in range(len(self.pt_edges) - 1):
            mask = pt_bin == k
            if not np.any(mask):
                continue
            target = finite_indices[mask]
            edges = np.asarray(self.var_edges[k], dtype=float)
            weights = np.asarray(self.w_grids[k], dtype=float)
            value_bin = np.searchsorted(edges, value_arr[target], side="right") - 1
            value_bin = np.clip(value_bin, 0, len(weights) - 1)
            out[target] = weights[value_bin]

        return out


def _get_herwig_weight_resource(is_groomed: bool, mode: str, channel: str = "zjet") -> tuple[str, str]:
    # Splines are channel-specific (different pt binning / generators). The
    # zjet files keep their historical names; other channels get a prefix.
    # Only the zjet channel currently ships mass splines.
    prefix = "" if channel == "zjet" else f"{channel}_"
    if mode == "rho":
        filename = f"{prefix}spline_groomed.npz" if is_groomed else f"{prefix}spline_ungroomed.npz"
        return filename, "rho_grids"
    if mode == "mass":
        if channel != "zjet":
            raise ValueError(f"Mass reweighting splines are not available for channel '{channel}'.")
        filename = "mass_spline_groomed.npz" if is_groomed else "mass_spline_ungroomed.npz"
        return filename, "mass_grids"
    raise ValueError(f"Unsupported reweight mode '{mode}'. Expected 'rho' or 'mass'.")


def _get_data_prior_rho_weight_resource(is_groomed: bool) -> tuple[str, str]:
    filename = (
        "data_prior_rho_binned_groomed.npz"
        if is_groomed
        else "data_prior_rho_binned_ungroomed.npz"
    )
    return filename, "rho_edges"

@lru_cache(maxsize=None)
def get_herwig_weight_g(mode: str = "rho", channel: str = "zjet") -> PtVarWeighter:
    filename, grid_key = _get_herwig_weight_resource(is_groomed=True, mode=mode, channel=channel)
    resource = files("smp_jetmass_run2") / "corrections" / filename
    with as_file(resource) as p:          # p is a real pathlib.Path on disk
        return PtVarWeighter(p, grid_key)

@lru_cache(maxsize=None)
def get_herwig_weight_u(mode: str = "rho", channel: str = "zjet") -> PtVarWeighter:
    filename, grid_key = _get_herwig_weight_resource(is_groomed=False, mode=mode, channel=channel)
    resource = files("smp_jetmass_run2") / "corrections" / filename
    with as_file(resource) as p:
        return PtVarWeighter(p, grid_key)


def _get_model_reweight_resource(source: str, is_groomed: bool, mode: str, channel: str = "zjet") -> tuple[str, str]:
    # Generic gen-reweight source for the modelling-uncertainty skims (Vincia, CR,
    # hadronization, ...). Each `source` maps to <source>_rho_reweight_{groomed,
    # ungroomed}.npz in PtVarWeighter format (w = variation / Pythia-CP5), built
    # from the standalone-generator derivations. Same grid_key as the Herwig files.
    if mode != "rho":
        raise ValueError(f"Model reweighting is only available for mode 'rho', got '{mode}'.")
    if channel != "zjet":
        raise ValueError(f"Model reweighting is only wired for channel 'zjet', got '{channel}'.")
    g = "groomed" if is_groomed else "ungroomed"
    return f"{source}_rho_reweight_{g}.npz", "rho_grids"


@lru_cache(maxsize=None)
def get_model_reweight_g(source: str, mode: str = "rho", channel: str = "zjet") -> PtVarWeighter:
    filename, grid_key = _get_model_reweight_resource(source, is_groomed=True, mode=mode, channel=channel)
    with as_file(files("smp_jetmass_run2") / "corrections" / filename) as p:
        return PtVarWeighter(p, grid_key)


@lru_cache(maxsize=None)
def get_model_reweight_u(source: str, mode: str = "rho", channel: str = "zjet") -> PtVarWeighter:
    filename, grid_key = _get_model_reweight_resource(source, is_groomed=False, mode=mode, channel=channel)
    with as_file(files("smp_jetmass_run2") / "corrections" / filename) as p:
        return PtVarWeighter(p, grid_key)


# backward-compatible aliases (source="vincia")
def get_vincia_weight_g(mode: str = "rho", channel: str = "zjet") -> PtVarWeighter:
    return get_model_reweight_g("vincia", mode=mode, channel=channel)


def get_vincia_weight_u(mode: str = "rho", channel: str = "zjet") -> PtVarWeighter:
    return get_model_reweight_u("vincia", mode=mode, channel=channel)


@lru_cache(maxsize=None)
def get_data_prior_rho_weight_g() -> PtBinnedVarWeighter:
    filename, _ = _get_data_prior_rho_weight_resource(is_groomed=True)
    resource = files("smp_jetmass_run2") / "corrections" / filename
    with as_file(resource) as p:
        return PtBinnedVarWeighter(p)


@lru_cache(maxsize=None)
def get_data_prior_rho_weight_u() -> PtBinnedVarWeighter:
    filename, _ = _get_data_prior_rho_weight_resource(is_groomed=False)
    resource = files("smp_jetmass_run2") / "corrections" / filename
    with as_file(resource) as p:
        return PtBinnedVarWeighter(p)

@lru_cache(maxsize=None)
def get_rocc_corrections(iov=None) -> RoccoR:
    # you can map iov -> filename later if you want
    resource = (
        files("smp_jetmass_run2")
        / "corrections" / "muonSF" / "UL2016" / "RoccoR2016aUL.txt"
    )
    with as_file(resource) as p:
        return RoccoR(str(p))   # str is safe; RoccoR opens the path

@lru_cache(maxsize=None)
def _load_jme_corrections(iov: str, mode: str):
    """Return a {name: correctionlib_wrapper} dict for the requested IOV and jet mode."""
    mode_key = mode.upper()
    if mode_key not in _JME_MODE_FILE:
        raise ValueError(f"Unsupported jet mode '{mode}'. Expected one of {tuple(_JME_MODE_FILE)}.")
    if iov not in _CORRLIB_NAME_MAP:
        raise ValueError(f"Unsupported IOV '{iov}'. Expected one of {tuple(_CORRLIB_NAME_MAP)}.")

    data_dir = files("smp_jetmass_run2") / "corrections" / "JME" / _CORRLIB_NAME_MAP[iov]
    data_path = data_dir / _JME_MODE_FILE[mode_key]
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing JME correction file at {data_path}")

    cset = correctionlib.CorrectionSet.from_file(str(data_path))
    return {name: correctionlib_wrapper(cset[name]) for name in cset.keys()}

@lru_cache(maxsize=None)
def _load_cset(iov: str) -> correctionlib.CorrectionSet:
    """Load and cache the CorrectionSet from packaged resources."""
    data_path = files("smp_jetmass_run2") / "corrections" / "pu" / f"{iov}_UL" / "puWeights.json.gz"
    with data_path.open("rb") as fh:
        raw = fh.read()
    data = gzip.decompress(raw).decode("utf-8")
    return correctionlib.CorrectionSet.from_string(data)

def get_pu_weights(events, iov: str):
    """Return (puNom, puUp, puDown) numpy arrays for given events and IOV."""
    cset = _load_cset(iov)
    key = _HNAME[iov]
    corr = cset[key]

    

    # robust extraction of nTrueInt as numpy
    ntrue = ak.to_numpy(events.Pileup.nTrueInt)
    # Some JSONs accept (ntrue, "variation")
    pu_nom  = corr.evaluate(ntrue, "nominal")
    pu_up   = corr.evaluate(ntrue, "up")
    pu_down = corr.evaluate(ntrue, "down")
    return pu_nom, pu_up, pu_down


@lru_cache(maxsize=None)
def _load_ele_trig_corrections() -> correctionlib.CorrectionSet:
    """Load electron trigger efficiency corrections from packaged resources."""
    data_path = files("smp_jetmass_run2") / "corrections" / "eleSF" / "egammaEffi_EGM2D.json"
    with data_path.open("r", encoding="utf-8") as fh:
        return correctionlib.CorrectionSet.from_string(fh.read())


@lru_cache(maxsize=None)
def _load_ele_sf_corrections(iov: str) -> correctionlib.CorrectionSet:
    """Load electron identification scale factors from packaged resources."""
    period = _CORRLIB_NAME_MAP[iov]
    data_path = files("smp_jetmass_run2") / "corrections" / "EGM" / period / "electron.json.gz"
    with data_path.open("rb") as fh:
        raw = fh.read()
    return correctionlib.CorrectionSet.from_string(gzip.decompress(raw).decode("utf-8"))


@lru_cache(maxsize=None)
def _load_muon_sf_corrections(iov: str) -> correctionlib.CorrectionSet:
    """Load muon scale factor corrections from packaged resources."""
    data_path = files("smp_jetmass_run2") / "corrections" / "muonSF" / f"UL{iov}" / "muon_Z.json.gz"
    with data_path.open("rb") as fh:
        raw = fh.read()
    return correctionlib.CorrectionSet.from_string(gzip.decompress(raw).decode("utf-8"))


def GetPDFweights(df):
    """Return (pdf_nominal, pdf_up, pdf_down) arrays for PDF variations."""
    pdf = ak.ones_like(df.Pileup.nTrueInt)
    if "LHEPdfWeight" in ak.fields(df):
        pdfUnc = ak.std(df.LHEPdfWeight, axis=1) / ak.mean(df.LHEPdfWeight, axis=1)
    else:
        pdfUnc = ak.zeros_like(pdf)
    return pdf, pdf + pdfUnc, pdf - pdfUnc

def GetL1PreFiringWeight(IOV, df):
    """Return (nominal, up, down) L1 prefiring weights from the event record."""
    if "L1PreFiringWeight" not in ak.fields(df):
        ones = np.ones(len(df), dtype=np.float64)
        return ones, ones, ones

    weights = df["L1PreFiringWeight"]
    nom = ak.to_numpy(ak.fill_none(weights["Nom"], 1.0))
    up = ak.to_numpy(ak.fill_none(weights["Up"], 1.0))
    down = ak.to_numpy(ak.fill_none(weights["Dn"], 1.0))
    return nom, up, down


def GetPSweights(df, shower = "ISR"):
    """ Return nominal, up, down weights for ISR or FSR """
    if shower == "ISR":
        ones = ak.ones_like(df.event)
        try:
            return ones, df.PSWeight[:,0], df.PSWeight[:,2]
        except:
            return ones, ones, ones
    elif shower == "FSR":
        ones = ak.ones_like(df.event)
        try:
            return ones, df.PSWeight[:,1], df.PSWeight[:,3]
        except:
            return ones, ones, ones

def GetQ2weights(df, var="nominal"):
    q2 = ak.ones_like(df.event, dtype = float)
    q2_up = ak.ones_like(df.event, dtype = float)
    q2_down = ak.ones_like(df.event, dtype = float)
    if ("LHEScaleWeight" in ak.fields(df)):
        if ak.all(ak.num(df.LHEScaleWeight, axis=1) == 9):
            nom = df.LHEScaleWeight[:, 4]
            scales = df.LHEScaleWeight[:, [0, 1, 3, 5, 7, 8]]
            q2_up = ak.max(scales, axis=1) / nom
            q2_down = ak.min(scales, axis=1) / nom
        elif ak.all(ak.num(df.LHEScaleWeight, axis=1) == 8):
            scales = df.LHEScaleWeight[:, [0, 1, 3, 4, 6, 7]]
            q2_up = ak.max(scales, axis=1)
            q2_down = ak.min(scales, axis=1)

    return q2, q2_up, q2_down
    
def GetEleTrigEff(IOV, lep0pT, lep0eta):
    """Return (nominal, up, down) electron trigger efficiencies."""
    counts = ak.num(lep0pT)
    ceval = _load_ele_trig_corrections()

    flat_eta = ak.flatten(lep0eta)
    flat_pt = ak.flatten(lep0pT)

    sf_nom = ceval["pt_reweight"].evaluate(flat_eta, flat_pt)
    sf_up = ceval["pt_reweight_up"].evaluate(flat_eta, flat_pt)
    sf_down = ceval["pt_reweight_down"].evaluate(flat_eta, flat_pt)

    return (
        ak.unflatten(sf_nom, counts),
        ak.unflatten(sf_up, counts),
        ak.unflatten(sf_down, counts),
    )
    
# def getRapidity(p4):
#     return 0.5 * np.log(( p4.energy + p4.pz ) / ( p4.energy - p4.pz ))
def getRapidity(obj, eps=1e-12):
    pt, eta, m = obj.pt, obj.eta, obj.mass
    pz = pt * np.sinh(eta)
    E  = np.sqrt((pt * np.cosh(eta))**2 + m**2)
    # print("[IN FUNCTION] Computing rapidity")
    # print("[IN FUNCTION] Energy ", E)
    # print("[IN FUNCTION] pz", pz)

    num = E + pz
    den = E - pz
    # print("[IN FUNCTION] num ", num)
    # print("[IN FUNCTION] den ", den)
    den = ak.where(den > eps, den, np.nan)
    num = ak.where(num > eps, num, np.nan)
    rapidity = 0.5 * np.log(num / den)
    # print("[IN FUNCTION] Rapidity ", rapidity)
    # print("[IN FUNCTION] Eta ", eta)
    return rapidity
    




# def compute_rapidity(obj):
#     """
#     Compute rapidity for NanoAOD objects with pt, eta, phi, mass
#     Works with awkward arrays (coffea NanoEvents).
#     """

#     pt = obj.pt
#     eta = obj.eta
#     mass = obj.mass

#     pz = pt * np.sinh(eta)
#     energy = np.sqrt((pt * np.cosh(eta))**2 + mass**2)

#     y = 0.5 * np.log((energy + pz) / (energy - pz))
#     return y
def compute_rapidity(obj, eps=1e-12):
    pt = obj.pt
    eta = obj.eta
    m = obj.mass

    pz = pt * np.sinh(eta)
    E  = np.sqrt((pt * np.cosh(eta))**2 + m**2)

    num = E + pz
    den = E - pz

    # protect against den <= 0 (or tiny) from weird values / rounding
    den = ak.where(den > eps, den, np.nan)
    num = ak.where(num > eps, num, np.nan)

    return 0.5 * np.log(num / den)
def GetEleSF(IOV, wp, eta, pt):
    """Return (nominal, systup, systdown) electron identification scale factors."""
    counts = ak.num(pt)
    evaluator = _load_ele_sf_corrections(IOV)

    mask = pt > 20
    adj_pt = ak.where(mask, pt, 22)

    flat_eta = np.array(ak.flatten(eta))
    flat_pt = np.array(ak.flatten(adj_pt))

    year = _ELE_SF_YEAR_LABEL[IOV]

    sf_nom_flat = evaluator["UL-Electron-ID-SF"].evaluate(year, "sf", wp, flat_eta, flat_pt)
    sf_up_flat  = evaluator["UL-Electron-ID-SF"].evaluate(year, "sfup", wp, flat_eta, flat_pt)
    sf_down_flat= evaluator["UL-Electron-ID-SF"].evaluate(year, "sfdown", wp, flat_eta, flat_pt)

    sf_nom  = ak.unflatten(sf_nom_flat, counts)
    sf_up   = ak.unflatten(sf_up_flat, counts)
    sf_down = ak.unflatten(sf_down_flat, counts)

    return (
        ak.where(mask, sf_nom, ak.ones_like(sf_nom)),
        ak.where(mask, sf_up,  ak.ones_like(sf_up)),
        ak.where(mask, sf_down,ak.ones_like(sf_down)),
    )


def GetMuonSF(IOV, corrset, abseta, pt):
    """Return (nominal, systup, systdown) muon scale factors for the requested set."""
    counts = ak.num(pt)
    evaluator = _load_muon_sf_corrections(IOV)

    # clamp to valid phase space covered by the correction files
    adj_abseta = ak.where(abseta < 2.4, abseta, 2.39)
    adj_pt = pt

    if corrset == "RECO":
        hname = "NUM_GlobalMuons_DEN_genTracks"
        adj_pt = ak.where(adj_pt < 15, 15.1, adj_pt)
    elif corrset == "HLT":
        if IOV == "2016" or IOV == "2016APV":
            hname = "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight"
            adj_pt = ak.where(adj_pt < 26, 26.1, adj_pt)
        elif IOV == "2017":
            hname = "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight"
            adj_pt = ak.where(adj_pt < 29, 29.1, adj_pt)
        elif IOV == "2018":
            hname = "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight"
            adj_pt = ak.where(adj_pt < 26, 26.1, adj_pt)
        else:
            raise ValueError(f"Unsupported IOV '{IOV}' for HLT muon scale factors")
    elif corrset == "ID":
        hname = "NUM_TightID_DEN_TrackerMuons"
        adj_pt = ak.where(adj_pt < 15, 15.1, adj_pt)
    elif corrset == "ISO":
        hname = "NUM_TightRelIso_DEN_TightIDandIPCut"
        adj_pt = ak.where(adj_pt < 15, 15.1, adj_pt)
    else:
        raise ValueError(f"Unsupported corrset '{corrset}' for muon scale factors")

    flat_eta = np.array(ak.flatten(adj_abseta))
    flat_pt = np.array(ak.flatten(adj_pt))
    wrapper = evaluator[hname]

    sf_nom = wrapper.evaluate(flat_eta, flat_pt, "nominal")
    sf_up = wrapper.evaluate(flat_eta, flat_pt, "systup")
    sf_down = wrapper.evaluate(flat_eta, flat_pt, "systdown")

    return (
        ak.unflatten(sf_nom, counts),
        ak.unflatten(sf_up, counts),
        ak.unflatten(sf_down, counts),
    )

from contextlib import ExitStack
from importlib.resources import files, as_file
from pathlib import Path

# One global stack per worker process keeps extracted files alive.
_resource_stack = ExitStack()

def resource_path(package: str, *parts: str) -> str:
    """
    Return a real filesystem path for a resource inside `package`.
    Works even when the package is a .zip (zipimport).
    The underlying temp file is kept alive by the process-global ExitStack.
    """
    ref = files(package).joinpath(*parts)
    path = _resource_stack.enter_context(as_file(ref))
    return str(path)

# def getLumiMaskRun2():


#     golden_json_dir = files("smp_jetmass_run2") / "corrections" / "goldenJsons"
#     golden_json_path_2016 = golden_json_dir / "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
#     golden_json_path_2017 = golden_json_dir / "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
#     golden_json_path_2018 = golden_json_dir / "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"

#     masks = {
#         "2016APV": LumiMask(str(golden_json_path_2016)),
#         "2016": LumiMask(str(golden_json_path_2016)),
#         "2017": LumiMask(str(golden_json_path_2017)),
#         "2018": LumiMask(str(golden_json_path_2018)),
#     }

#     return masks

def getLumiMaskRun2():
    pkg = "smp_jetmass_run2"  # package root

    golden_json_path_2016 = resource_path(pkg, "corrections", "goldenJsons",
        "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
    golden_json_path_2017 = resource_path(pkg, "corrections", "goldenJsons",
        "Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
    golden_json_path_2018 = resource_path(pkg, "corrections", "goldenJsons",
        "Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")

    masks = {
        "2016APV": LumiMask(golden_json_path_2016),
        "2016"   : LumiMask(golden_json_path_2016),
        "2017"   : LumiMask(golden_json_path_2017),
        "2018"   : LumiMask(golden_json_path_2018),
    }
    return masks


def debug_jec_weightset(iov: str = "2018", mode: str = "AK8", is_data: bool = False, run: Optional[str] = None):
    """Small helper to test reading a single JEC text file with coffea's extractor."""
    mode_key = mode.upper()
    if mode_key not in _JME_AK_LABEL:
        raise ValueError(f"Unsupported jet mode '{mode}'. Expected one of {tuple(_JME_AK_LABEL)}.")

    ak_label = _JME_AK_LABEL[mode_key]
    jec_tag, jec_tag_data, _ = _resolve_jec_tags(iov)

    corrections_root = files("smp_jetmass_run2").joinpath("corrections")
    if is_data:
        if run is None:
            raise ValueError("Parameter 'run' must be provided when is_data=True.")
        if run not in jec_tag_data:
            raise KeyError(f"Run '{run}' not available for IOV '{iov}'.")
        tag = jec_tag_data[run]
        target_resource = corrections_root.joinpath("JEC", tag, f"{tag}_L1FastJet_{ak_label}.jec.txt")
    else:
        target_resource = corrections_root.joinpath("JEC", jec_tag, f"{jec_tag}_L1FastJet_{ak_label}.jec.txt")

    exists_in_package = target_resource.is_file()

    with ExitStack() as stack:
        tmp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        target_path = tmp_dir / target_resource.name
        with target_resource.open("rb") as src, target_path.open("wb") as dst:
            dst.write(src.read())
        ext = extractor()
        ext.add_weight_sets([f"* * {target_path.as_posix()}"])
        ext.finalize()
        evaluator = ext.make_evaluator()
        keys = list(evaluator.keys())

    return {
        "path": target_path.as_posix(),
        "resource_exists": exists_in_package,
        "evaluator_keys": keys,
    }


# ARC recommendation (SMP-25-010 / AN-24-162 L777): the JMAR JMS/JMR scale
# factors are derived for the GROOMED (soft-drop) jet mass only. There is no
# dedicated ungroomed calibration, so the ungroomed up/down uncertainty is
# conservatively inflated by this factor (the deviation from the nominal SF is
# scaled by it). Nominal is left unchanged. Default: x2 -> JMS 1%->2%, JMR 2%->4%.
UNGROOMED_JMSJMR_INFLATION = 2.0


def _inflate_ungroomed_sf(nom, var_val):
    """Inflate an up/down JMS/JMR scale factor for the ungroomed jet mass.

    The groomed (soft-drop) mass keeps the JMAR value ``var_val`` directly; the
    ungroomed mass has no dedicated calibration, so its deviation from the
    nominal SF is scaled by ``UNGROOMED_JMSJMR_INFLATION``. For the nominal
    variation (``var_val == nom``) this returns ``nom`` unchanged, so only the
    up/down systematic width is affected.
    """
    return nom + UNGROOMED_JMSJMR_INFLATION * (var_val - nom)


def jmssf(IOV, FatJet,  var = ''):
    # Old per-year JMAR values (pre SoftDropJMSJMRULRun2 recommendation), kept
    # for provenance -- these were still applied in the arc_r2 reskims. The
    # GluonJetMass side switched to unity +- 1% in 15ae99c (2025-07-01); this
    # port brings the zjet lineage in line with it.
    # "2016APV":{"sf": 1.00, "sfup": 1.0094, "sfdown": 0.9906},
    # "2016"   :{"sf": 1.00, "sfup": 1.0094, "sfdown": 0.9906},
    # "2017"   :{"sf": 0.982, "sfup": 0.986, "sfdown": 0.978},
    # "2018"   :{"sf": 0.999, "sfup": 1.001, "sfdown": 0.997}}
    # UL recommendation: SF = 1 with 1% uncertainty (groomed), see
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/SoftDropJMSJMRULRun2
    jmsSF = {

        "2016APV":{"sf": 1.00, "sfup": 1.01, "sfdown": 0.99},

        "2016"   :{"sf": 1.00, "sfup": 1.01, "sfdown": 0.99},

        "2017"   :{"sf": 1.00, "sfup": 1.01, "sfdown": 0.99},

        "2018"   :{"sf": 1.00, "sfup": 1.01, "sfdown": 0.99}}

    nom = jmsSF[IOV]["sf"]
    out = jmsSF[IOV]["sf"+var]

    # groomed (soft-drop): JMAR value applied directly
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * out, 'msoftdrop')
    # ungroomed: no dedicated calibration -> inflate the up/down deviation
    out_ung = _inflate_ungroomed_sf(nom, out)
    FatJet = ak.with_field(FatJet, FatJet.mass * out_ung, 'mass')
    return FatJet

def jmrsf(IOV, FatJet, var = ''):
    # Old per-year JMAR values (pre SoftDropJMSJMRULRun2 recommendation), kept
    # for provenance -- these were still applied in the arc_r2 reskims,
    # including the +-20% placeholder for 2016(APV).
    # "2016APV":{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8},
    # "2016"   :{"sf": 1.00, "sfup": 1.2, "sfdown": 0.8},
    # "2017"   :{"sf": 1.09, "sfup": 1.14, "sfdown": 1.04},
    # "2018"   :{"sf": 1.108, "sfup": 1.142, "sfdown": 1.074}}
    # UL recommendation: SF = 1 with 2% uncertainty (groomed), see
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/SoftDropJMSJMRULRun2
    jmrSF = {

        "2016APV":{"sf": 1.00, "sfup": 1.02, "sfdown": 0.98},
        "2016"   :{"sf": 1.00, "sfup": 1.02, "sfdown": 0.98},

        "2017"   :{"sf": 1.00, "sfup": 1.02, "sfdown": 0.98},

        "2018"   :{"sf": 1.00, "sfup": 1.02, "sfdown": 0.98}}

    nom = jmrSF[IOV]["sf"]
    jmrvalnom = jmrSF[IOV]["sf"+var]
    recomass = FatJet.mass
    genmass = FatJet.matched_gen.mass

    def _jmr_factor(jval):
        # dR-unmatched jets have no gen mass to smear against: leave them
        # unsmeared (factor 1, deltamass -> 0) instead of propagating None --
        # otherwise fakes get an undefined mass and silently drop out of the
        # reco spectra (and the unfolder's fakes-by-subtraction).
        deltamass = ak.fill_none((recomass-genmass)*(jval-1.0), 0.0, axis=-1)
        condition = ((recomass+deltamass)/recomass) > 0
        return ak.where( recomass <= 0.0, 0 , ak.where( condition , (recomass+deltamass)/recomass, 0 ))

    # groomed (soft-drop): JMAR value applied directly
    FatJet = ak.with_field(FatJet, FatJet.msoftdrop * _jmr_factor(jmrvalnom), 'msoftdrop')
    # ungroomed: no dedicated calibration -> inflate the up/down deviation
    jmrval_ung = _inflate_ungroomed_sf(nom, jmrvalnom)
    FatJet = ak.with_field(FatJet, FatJet.mass * _jmr_factor(jmrval_ung), 'mass')
    return FatJet


@lru_cache(maxsize=None)
def _get_jet_factory(IOV, isData, mode, era, uncertainties_key):
    """Build (and cache) the CorrectedJetsFactory for a given correction setup.

    Everything here — parsing the JEC/JER text files, building the evaluator,
    JECStack, and CorrectedJetsFactory — depends only on the correction *setup*
    (IOV, data-vs-MC, AK4/AK8, data era, uncertainty list), never on the actual
    jets. Constructing it parses ~30 text files from disk, so doing it per chunk
    dominated the hadronic runtime. The factory is reusable across chunks: only
    the lazy ``jet_factory.build(jets)`` in GetJetCorrections is per-event work.
    ``uncertainties_key`` is a tuple (or None) so the args stay hashable.
    """
    AK_str = 'AK8PFPuppi'
    if mode == 'AK4':
        AK_str = 'AK4PFPuppi'

    print(f"Using TAG {AK_str}")
    if uncertainties_key is None:
        uncertainty_sources = ["AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF",
"PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF","RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample", "RelativeStatEC","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
    else:
        uncertainty_sources = list(uncertainties_key)
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py
    jec_tag, jec_tag_data, jer_tag = _resolve_jec_tags(IOV)


    #print("extracting corrections from files for " + jec_tag)
    ext = extractor()
    pkg_root = files("smp_jetmass_run2")
    corrections_root = pkg_root.joinpath("corrections")
    jec_dir = corrections_root.joinpath("JEC", jec_tag)
    data_file = jec_dir.joinpath(f"{jec_tag}_L1FastJet_{AK_str}.jec.txt")
    #print("File exists check for JEC: ", str(data_file))
    #print(data_file.is_file())

    with ExitStack() as stack:
        tmp_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        cache = {}

        def resource_file(*parts: str) -> Path:
            traversable = corrections_root.joinpath(*parts)
            if not traversable.is_file():
                raise FileNotFoundError(f"Missing correction resource at {traversable}")
            key = "/".join(parts)
            if key not in cache:
                target_path = tmp_dir / traversable.name
                with traversable.open("rb") as src, target_path.open("wb") as dst:
                    dst.write(src.read())
                cache[key] = target_path
            return cache[key]

        if not isData:
        #For MC
            ext.add_weight_sets([
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_L1FastJet_{AK_str}.jec.txt").as_posix(),
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_L2Relative_{AK_str}.jec.txt").as_posix(),
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_L3Absolute_{AK_str}.jec.txt").as_posix(),
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_UncertaintySources_{AK_str}.junc.txt").as_posix(),
                '* * ' + resource_file("JEC", jec_tag, f"{jec_tag}_Uncertainty_{AK_str}.junc.txt").as_posix(),
            ])
            #### Do AK8PUPPI jer files exist??
            if jer_tag:
                ext.add_weight_sets([
                '* * ' + resource_file("JER", jer_tag, f"{jer_tag}_PtResolution_{AK_str}.jr.txt").as_posix(),
                '* * ' + resource_file("JER", jer_tag, f"{jer_tag}_SF_{AK_str}.jersf.txt").as_posix()])
                # print("JER SF added")
        else:       
            
            #For data, make sure we don't duplicate
            tags_done = []
            print("In the DATA section")
            for run, tag in jec_tag_data.items():
                if not (tag in tags_done):

                    #print("Doing", tag, AK_str)
                    ext.add_weight_sets([
                    '* * ' + resource_file("JEC", tag, f"{tag}_L1FastJet_{AK_str}.jec.txt").as_posix(),
                    '* * ' + resource_file("JEC", tag, f"{tag}_L2Relative_{AK_str}.jec.txt").as_posix(),
                    '* * ' + resource_file("JEC", tag, f"{tag}_L3Absolute_{AK_str}.jec.txt").as_posix(),
                    '* * ' + resource_file("JEC", tag, f"{tag}_L2L3Residual_{AK_str}.jec.txt").as_posix(),
                    ])
                    tags_done += [tag]
                    #print("Done", tag, AK_str)
            print("Added JEC weight sets")

        
        ext.finalize()

        evaluator = ext.make_evaluator()

    if (not isData):
        jec_names = [
            '{0}_L1FastJet_{1}'.format(jec_tag, AK_str),
            '{0}_L2Relative_{1}'.format(jec_tag, AK_str),
            '{0}_L3Absolute_{1}'.format(jec_tag, AK_str)]
        #### if jes in arguments add total uncertainty values for comparison and easy plotting
        if 'jes' in uncertainty_sources:
            jec_names.extend(['{0}_Uncertainty_{1}'.format(jec_tag, AK_str)])
            uncertainty_sources.remove('jes')
        jec_names.extend(['{0}_UncertaintySources_{1}_{2}'.format(jec_tag, AK_str, unc_src) for unc_src in uncertainty_sources])

        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_{1}'.format(jer_tag, AK_str),
                              '{0}_SF_{1}'.format(jer_tag, AK_str)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_{1}'.format(tag, AK_str),
                '{0}_L3Absolute_{1}'.format(tag, AK_str),
                '{0}_L2Relative_{1}'.format(tag, AK_str),
                '{0}_L2L3Residual_{1}'.format(tag, AK_str),]



    if not isData:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_inputs = {name: evaluator[name] for name in jec_names[era]}


    # print("jec_input", jec_inputs)
    jec_stack = JECStack(jec_inputs)

    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['Rho'] = 'jec_rho'

    return CorrectedJetsFactory(name_map, jec_stack)


def GetJetCorrections(FatJets, events, era, IOV, isData=False, uncertainties=None, mode='AK8'):
    #### The expensive evaluator/JECStack/factory build is cached in
    #### _get_jet_factory (it depends only on the correction setup). Here we only
    #### attach the per-event input fields the factory's name_map expects and run
    #### the lazy build, which is the actual per-chunk work.
    uncertainties_key = None if uncertainties is None else tuple(uncertainties)
    jet_factory = _get_jet_factory(IOV, isData, mode, era, uncertainties_key)

    if not isData:
        if mode == 'AK8':
            FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.matched_gen.pt, 0), np.float32)
        if mode == 'AK4':
            SubGenJetAK8 = events.SubGenJetAK8
            SubGenJetAK8['p4'] = ak.with_name(SubGenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJets["p4"] = ak.with_name(FatJets[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.p4.nearest(SubGenJetAK8.p4, threshold=0.4).pt, 0), np.float32)
    if mode == 'AK4':
        FatJets['area'] = ak.full_like( FatJets.pt, 0.503)

    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    FatJets['jec_rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]

    return jet_factory.build(FatJets)

def pad_none_lep(array):
    """Pad an awkward array of leptons to have exactly two entries per event, filling with None."""
    return ak.pad_none(array, 2, axis=1, clip=True)
    
def add_lepton_weights(events_j, twoReco_ee_sel, twoReco_mm_sel, weights, IOV):
    elereco_nom, elereco_up, elereco_down = GetEleSF(IOV, "RecoAbove20", events_j.Electron.eta, events_j.Electron.pt)
    elereco_nom, elereco_up, elereco_down = pad_none_lep(elereco_nom), pad_none_lep(elereco_up), pad_none_lep(elereco_down)       
    elereco_nom = ak.where(twoReco_ee_sel, elereco_nom[:,0] * elereco_nom[:,1], 1.0)
    elereco_up = ak.where(twoReco_ee_sel, elereco_up[:,0] * elereco_up[:,1], 1.0)
    elereco_down = ak.where(twoReco_ee_sel, elereco_down[:,0] * elereco_down[:,1], 1.0)
    weights.add(name = "elereco", weight = elereco_nom, weightUp = elereco_up, weightDown = elereco_down)

    eleid_nom, eleid_up, eleid_down = GetEleSF(IOV, "Tight", events_j.Electron.eta, events_j.Electron.pt)
    eleid_nom, eleid_up, eleid_down = pad_none_lep(eleid_nom), pad_none_lep(eleid_up), pad_none_lep(eleid_down)
    eleid_nom = ak.where(twoReco_ee_sel, eleid_nom[:,0] * eleid_nom[:,1], 1.0)
    eleid_up = ak.where(twoReco_ee_sel, eleid_up[:,0] * eleid_up[:,1], 1.0)
    eleid_down = ak.where(twoReco_ee_sel, eleid_down[:,0] * eleid_down[:,1], 1.0)
    weights.add(name = "eleid", weight = eleid_nom, weightUp = eleid_up, weightDown = eleid_down)

    eletrig_nom, eletrig_up, eletrig_down = GetEleTrigEff(IOV, events_j.Electron.pt, events_j.Electron.eta)
    eletrig_nom, eletrig_up, eletrig_down = pad_none_lep(eletrig_nom), pad_none_lep(eletrig_up), pad_none_lep(eletrig_down)
    eletrig_nom = ak.where(twoReco_ee_sel, eletrig_nom[:,0] * eletrig_nom[:,1], 1.0)
    eletrig_up = ak.where(twoReco_ee_sel, eletrig_up[:,0] * eletrig_up[:,1], 1.0)
    eletrig_down = ak.where(twoReco_ee_sel, eletrig_down[:,0] * eletrig_down[:,1], 1.0)
    weights.add(name = "eletrig", weight = eletrig_nom, weightUp = eletrig_up, weightDown = eletrig_down)


    # Muon SFs in the same style as electrons
    mureco_nom, mureco_up, mureco_down = GetMuonSF(IOV, "RECO", np.abs(events_j.Muon.eta), events_j.Muon.pt)
    mureco_nom, mureco_up, mureco_down = pad_none_lep(mureco_nom), pad_none_lep(mureco_up), pad_none_lep(mureco_down)
    mureco_nom = ak.where(twoReco_mm_sel, mureco_nom[:,0] * mureco_nom[:,1], 1.0)
    mureco_up  = ak.where(twoReco_mm_sel, mureco_up[:,0]  * mureco_up[:,1],  1.0)
    mureco_down= ak.where(twoReco_mm_sel, mureco_down[:,0]* mureco_down[:,1],1.0)
    weights.add(name="mureco", weight=mureco_nom, weightUp=mureco_up, weightDown=mureco_down)

    muid_nom, muid_up, muid_down = GetMuonSF(IOV, "ID", np.abs(events_j.Muon.eta), events_j.Muon.pt)
    muid_nom, muid_up, muid_down = pad_none_lep(muid_nom), pad_none_lep(muid_up), pad_none_lep(muid_down)
    muid_nom = ak.where(twoReco_mm_sel, muid_nom[:,0] * muid_nom[:,1], 1.0)
    muid_up  = ak.where(twoReco_mm_sel, muid_up[:,0]  * muid_up[:,1],  1.0)
    muid_down= ak.where(twoReco_mm_sel, muid_down[:,0]* muid_down[:,1],1.0)
    weights.add(name="muid", weight=muid_nom, weightUp=muid_up, weightDown=muid_down)

    muiso_nom, muiso_up, muiso_down = GetMuonSF(IOV, "ISO", np.abs(events_j.Muon.eta), events_j.Muon.pt)
    muiso_nom, muiso_up, muiso_down = pad_none_lep(muiso_nom), pad_none_lep(muiso_up), pad_none_lep(muiso_down)
    muiso_nom = ak.where(twoReco_mm_sel, muiso_nom[:,0] * muiso_nom[:,1], 1.0)
    muiso_up  = ak.where(twoReco_mm_sel, muiso_up[:,0]  * muiso_up[:,1],  1.0)
    muiso_down= ak.where(twoReco_mm_sel, muiso_down[:,0]* muiso_down[:,1],1.0)
    weights.add(name="muiso", weight=muiso_nom, weightUp=muiso_up, weightDown=muiso_down)

    mutrig_nom, mutrig_up, mutrig_down = GetMuonSF(IOV, "HLT", np.abs(events_j.Muon.eta), events_j.Muon.pt)
    mutrig_nom, mutrig_up, mutrig_down = pad_none_lep(mutrig_nom), pad_none_lep(mutrig_up), pad_none_lep(mutrig_down)
    mutrig_nom = ak.where(twoReco_mm_sel, mutrig_nom[:,0] * mutrig_nom[:,1], 1.0)
    mutrig_up  = ak.where(twoReco_mm_sel, mutrig_up[:,0]  * mutrig_up[:,1],  1.0)
    mutrig_down= ak.where(twoReco_mm_sel, mutrig_down[:,0]* mutrig_down[:,1],1.0)
    weights.add(name="mutrig", weight=mutrig_nom, weightUp=mutrig_up, weightDown=mutrig_down)

def HEMVeto(FatJets, runs, iov):
    ## from https://github.com/laurenhay/GluonJetMass/blob/main/python/corrections.py
    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    
    runid = (runs >= 319077)
    print(runid)
    # print("Fat jet phi ", FatJets.phi)
    # print("Fat jet phi length ", len(FatJets.phi))
    # print("Fat jet eta ", FatJets.eta)
    # print("Fat jet eta length ", len(FatJets.eta))
    detector_region1 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -1.3) & (FatJets.eta > -2.5))
    detector_region2 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -2.5) & (FatJets.eta > -3.0))
    jet_selection    = ((FatJets.jetId > 1) & (FatJets.pt > 15))

    vetoHEMFatJets = ak.any((detector_region1 & jet_selection & runid) ^ (detector_region2 & jet_selection & runid), axis=1)
    #print("Number of hem vetoed jets: ", ak.sum(vetoHEMFatJets))
    vetoHEM = ~(vetoHEMFatJets)
    
    return vetoHEM

def HEMVeto(FatJets, runs, isMC=False, year="2018"):
    """
    HEM veto for data and MC (flat weight approach for MC).
    For data: returns boolean mask (True = event passes veto).
    For MC:   returns per-event weight (1.0 or flat lumi-fraction weight).
    """

    # Luminosity fractions for 2018 (adjust to your golden JSON)
    L_total    = 59.74  # fb-1, total 2018
    L_preHEM   = 21.07  # fb-1, before run 319077 (2018A+B+early C)
    L_postHEM  = 38.67  # fb-1, run >= 319077 (late 2018C + 2018D)
    f_affected = L_postHEM / L_total  # ~0.647

    # Define dead detector regions
    detector_region1 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                        (FatJets.eta < -1.3)  & (FatJets.eta > -2.5))
    detector_region2 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                        (FatJets.eta < -2.5)  & (FatJets.eta > -3.0))
    jet_selection    = ((FatJets.jetId > 1) & (FatJets.pt > 15))

    in_HEM_region = (detector_region1 | detector_region2) & jet_selection
    event_has_HEM_jet = ak.any(in_HEM_region, axis=1)  # True = bad jet present

    if not isMC:
        # ---- DATA: hard veto, only for affected runs ----
        runid = (runs >= 319077)
        vetoHEMFatJets = ak.any(in_HEM_region & runid, axis=1)
        return ~vetoHEMFatJets  # True = event passes

    else:
        # ---- MC (2018 only): flat luminosity-fraction weight ----
        # Events with a HEM jet: they would be vetoed in the affected lumi fraction
        #   → weight = (1 - f_affected) = L_preHEM / L_total
        # Events without a HEM jet: always pass
        #   → weight = 1.0
 
        hem_weight = ak.where(
            event_has_HEM_jet,
            (1.0 - f_affected),   # ~0.353 — only the "safe" lumi counts
            1.0                   # no HEM jet → unaffected
        )
        return hem_weight


# =====================================================================
# MET xy ("phi") correction  (UL, PF Type-1 MET)
# ---------------------------------------------------------------------
# Removes the nPV-dependent MET phi-modulation from the beamspot offset +
# phi-nonuniform detector response, parametrized SEPARATELY for data (per
# era, by run number) and MC (per IOV). Coefficients are the standard
# Louis Thomas UL recipe (XYMETCorrection_withUL17andUL18andUL16.h,
# non-puppi branch): METxcorr = -(a*npv + b), METycorr = -(c*npv + d);
# corrected (x, y) = uncorrected (x, y) + (METxcorr, METycorr). npv is
# capped at 100 as in the reference. Only used for validation MET plots --
# MET is not a selection variable, so this does not touch the measurement.
_MET_XY_MC = {  # (a, b, c, d)
    "2016APV": (-0.188743, 0.136539, 0.0127927, 0.117747),
    "2016":    (-0.153497, -0.231751, 0.00731978, 0.243323),
    "2017":    (-0.300155, 1.90608, 0.300213, -2.02232),
    "2018":    (0.183518, 0.546754, 0.192263, -0.42121),
}
_MET_XY_DATA = {  # era label -> (a, b, c, d)
    "2018A": (0.263733, -1.91115, 0.0431304, -0.112043),
    "2018B": (0.400466, -3.05914, 0.146125, -0.533233),
    "2018C": (0.430911, -1.42865, 0.0620083, -1.46021),
    "2018D": (0.457327, -1.56856, 0.0684071, -0.928372),
    "2017B": (-0.211161, 0.419333, 0.251789, -1.28089),
    "2017C": (-0.185184, -0.164009, 0.200941, -0.56853),
    "2017D": (-0.201606, 0.426502, 0.188208, -0.58313),
    "2017E": (-0.162472, 0.176329, 0.138076, -0.250239),
    "2017F": (-0.210639, 0.72934, 0.198626, 1.028),
    "2016B": (-0.0214894, -0.188255, 0.0876624, 0.812885),
    "2016C": (-0.032209, 0.067288, 0.113917, 0.743906),
    "2016D": (-0.0293663, 0.21106, 0.11331, 0.815787),
    "2016E": (-0.0132046, 0.20073, 0.134809, 0.679068),
    "2016F": (-0.0543566, 0.816597, 0.114225, 1.17266),
    "2016Flate": (0.134616, -0.89965, 0.0397736, 1.0385),
    "2016G": (0.121809, -0.584893, 0.0558974, 0.891234),
    "2016H": (0.0868828, -0.703489, 0.0888774, 0.902632),
}


def _met_xy_data_era_masks(run):
    """Boolean run-number masks for each UL data era (matches the reference
    run ranges, including the 2016F/F-late single-run special cases)."""
    return {
        "2018A": (run >= 315252) & (run <= 316995),
        "2018B": (run >= 316998) & (run <= 319312),
        "2018C": (run >= 319313) & (run <= 320393),
        "2018D": (run >= 320394) & (run <= 325273),
        "2017B": (run >= 297020) & (run <= 299329),
        "2017C": (run >= 299337) & (run <= 302029),
        "2017D": (run >= 302030) & (run <= 303434),
        "2017E": (run >= 303435) & (run <= 304826),
        "2017F": (run >= 304911) & (run <= 306462),
        "2016B": (run >= 272007) & (run <= 275376),
        "2016C": (run >= 275657) & (run <= 276283),
        "2016D": (run >= 276315) & (run <= 276811),
        "2016E": (run >= 276831) & (run <= 277420),
        "2016F": ((run >= 277772) & (run <= 278768)) | (run == 278770),
        "2016Flate": ((run >= 278801) & (run <= 278808)) | (run == 278769),
        "2016G": (run >= 278820) & (run <= 280385),
        "2016H": (run >= 280919) & (run <= 284044),
    }


def METXYCorr(met_pt, met_phi, npv, iov, isMC=False, run=None):
    """UL PF-MET xy correction. Returns (corrected_pt, corrected_phi) as numpy
    arrays. `npv` is the number of primary vertices (events.PV.npvs); `run` is
    required for data (per-era coefficients). Runs/eras that match nothing get
    zero correction (a safe no-op)."""
    npv = np.minimum(np.asarray(ak.to_numpy(npv), dtype=np.float64), 100.0)
    pt = np.asarray(ak.to_numpy(met_pt), dtype=np.float64)
    phi = np.asarray(ak.to_numpy(met_phi), dtype=np.float64)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)

    xcorr = np.zeros_like(npv)
    ycorr = np.zeros_like(npv)
    if isMC:
        if iov not in _MET_XY_MC:
            raise KeyError(f"No MET xy MC coefficients for IOV {iov!r}")
        a, b, c, d = _MET_XY_MC[iov]
        xcorr = -(a * npv + b)
        ycorr = -(c * npv + d)
    else:
        if run is None:
            raise ValueError("METXYCorr needs `run` for data (per-era coefficients).")
        run = np.asarray(ak.to_numpy(run))
        for era, mask in _met_xy_data_era_masks(run).items():
            a, b, c, d = _MET_XY_DATA[era]
            xcorr = np.where(mask, -(a * npv + b), xcorr)
            ycorr = np.where(mask, -(c * npv + d), ycorr)

    cx = px + xcorr
    cy = py + ycorr
    return np.hypot(cx, cy), np.arctan2(cy, cx)


# =====================================================================
# Hadronic (dijet / trijet) ported functions
# ---------------------------------------------------------------------
# Ported from GluonJetMass python/corrections.py and python/utils.py.
# Selections in dijet/trijet are kept byte-for-byte identical, so these
# functions are reproduced verbatim except that file access goes through
# the packaged `corrections/` data (resource_path) instead of a
# filesystem `correctionFiles/` directory.
#
# Per user decisions:
#   - JMS/JMR: since the zjet unity port (54bb33f) jmssf/jmrsf carry the
#     same flat sf=1.0, +-1% / +-2% tables as GluonJetMass, so all three
#     channels share them; applyjmsSF / applyjmrSF are kept as aliases.
#   - q2 kept split as GetQ2muF + GetQ2muR.
#   - HEM uses the weight-based zjet HEMVeto(FatJets, runs, isMC, year)
#     above; gluon's HEMCleaning jet-pt-scaling is intentionally NOT
#     ported (the old 'HEM' jet systematic is replaced by the weight).
# =====================================================================


def _hadronic_corr_path(*parts):
    """Real filesystem path to a packaged correction file (for correctionlib)."""
    return resource_path("smp_jetmass_run2", "corrections", *parts)


## --------------------------------- MET Filters ------------------------------#
## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL
MET_filters = {
    '2016APV': ["goodVertices", "globalSuperTightHalo2016Filter", "HBHENoiseFilter",
                "HBHENoiseIsoFilter", "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter", "BadPFMuonDzFilter", "eeBadScFilter",
                "hfNoisyHitsFilter"],
    '2016':    ["goodVertices", "globalSuperTightHalo2016Filter", "HBHENoiseFilter",
                "HBHENoiseIsoFilter", "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter", "BadPFMuonDzFilter", "eeBadScFilter",
                "hfNoisyHitsFilter"],
    '2017':    ["goodVertices", "globalSuperTightHalo2016Filter", "HBHENoiseFilter",
                "HBHENoiseIsoFilter", "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter", "BadPFMuonDzFilter", "hfNoisyHitsFilter",
                "eeBadScFilter", "ecalBadCalibFilter"],
    '2018':    ["goodVertices", "globalSuperTightHalo2016Filter", "HBHENoiseFilter",
                "HBHENoiseIsoFilter", "EcalDeadCellTriggerPrimitiveFilter",
                "BadPFMuonFilter", "BadPFMuonDzFilter", "hfNoisyHitsFilter",
                "eeBadScFilter", "ecalBadCalibFilter"],
}


# JMS/JMR: identical unity tables in all channels since the zjet UL port
# (54bb33f) -- the hadronic processors import these aliases, the shared
# implementation lives in jmssf/jmrsf above.
applyjmsSF = jmssf
applyjmrSF = jmrsf


def GetPSWeights(df, shower="ISR"):
    """Return nominal, up, down weights for ISR or FSR.

    Pads PSWeight to length 4 with the neutral weight 1.0 (some samples store
    fewer than 4 entries; awkward 2 raises IndexError on overruns).
    """
    ps = ak.fill_none(ak.pad_none(df.PSWeight, 4, axis=1, clip=True), 1.0)
    ones = ak.ones_like(df.event)
    if shower == "ISR":
        return ones, ps[:, 0], ps[:, 2]
    elif shower == "FSR":
        return ones, ps[:, 1], ps[:, 3]


def GetQ2muF(events):
    muF = np.ones(len(events))
    up = np.ones(len(events))
    down = np.ones(len(events))
    if ("LHEScaleWeight" in ak.fields(events)):
        if ak.all(ak.num(events.LHEScaleWeight, axis=1) == 9):
            nom = events.LHEScaleWeight[:, 4]
            up = events.LHEScaleWeight[:, 5] / nom
            down = events.LHEScaleWeight[:, 3] / nom
        elif ak.all(ak.num(events.LHEScaleWeight, axis=1) == 8):
            up = events.LHEScaleWeight[:, 4]
            down = events.LHEScaleWeight[:, 3]
    return muF, up, down


def GetQ2muR(events):
    muR = np.ones(len(events))
    up = np.ones(len(events))
    down = np.ones(len(events))
    if ("LHEScaleWeight" in ak.fields(events)):
        if ak.all(ak.num(events.LHEScaleWeight, axis=1) == 9):
            # pure muR variation at muF=1: nom=[4]=(muR1,muF1),
            # up=[7]=(muR2,muF1), down=[1]=(muR0.5,muF1)
            nom = events.LHEScaleWeight[:, 4]
            up = events.LHEScaleWeight[:, 7] / nom
            down = events.LHEScaleWeight[:, 1] / nom
        elif ak.all(ak.num(events.LHEScaleWeight, axis=1) == 8):
            up = events.LHEScaleWeight[:, 6]
            down = events.LHEScaleWeight[:, 1]
    return muR, up, down


def GetLumiUnc(events, IOV):
    lumi_unc = {"2016": 0.016, "2016APV": 0.016, "2017": 0.016, "2018": 0.016}
    lumi_nom = ak.ones_like(events.L1PreFiringWeight.Nom)
    lumi_up = (1.0 + lumi_unc[IOV]) * lumi_nom
    lumi_dn = (1.0 - lumi_unc[IOV]) * lumi_nom
    return lumi_nom, lumi_up, lumi_dn


def ApplyVetoMap(IOV, jets, mapname='jetvetomap'):
    if IOV == "2016APV":
        IOV = "2016"
    fname = _hadronic_corr_path("jetvetomap", "jetvetomaps_UL" + IOV + ".json.gz")
    hname = {"2016": "Summer19UL16_V1", "2017": "Summer19UL17_V1", "2018": "Summer19UL18_V1"}
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    jetphi = np.where(jets.phi < 3.141592, jets.phi, 3.141592)
    jetphi = np.where(jetphi > -3.141592, jetphi, -3.141592)
    vetoedjets = np.array(evaluator[hname[IOV]].evaluate(mapname, np.array(jets.eta), jetphi), dtype=bool)
    return ~vetoedjets


def getJetFlavors(jet):
    genjet = jet.matched_gen
    jetflavs = {}
    jetflavs["Gluon"]  = jet[np.abs(genjet.partonFlavour) == 21]
    jetflavs["UDS"]    = jet[np.abs(genjet.partonFlavour) < 4]
    jetflavs["Charm"]  = jet[np.abs(genjet.partonFlavour) == 4]
    jetflavs["Bottom"] = jet[np.abs(genjet.partonFlavour) == 5]
    jetflavs["Other"]  = jet[(np.abs(genjet.partonFlavour) > 5) & (np.abs(genjet.partonFlavour) != 21)]
    return jetflavs


def applyBTag(events, btag):
    if (btag == 'bbloose'):
        sel = (events.FatJet[:, 0].btagDeepB >= 0.2027) & (events.FatJet[:, 1].btagDeepB >= 0.2027)
        events = events[sel]
        print('Loose WP CSV V2 B tag applied to leading two jets')
    elif (btag == 'bloose'):
        sel = (events.FatJet[:, 0].btagDeepB >= 0.2027)
        events = events[sel]
        print('Loose WP CSV V2 B tag applied to leading jet only')
    elif (btag == 'bbmed'):
        sel = (events.FatJet[:, 0].btagDeepB >= 0.6001) & (events.FatJet[:, 1].btagDeepB >= 0.6001)
        events = events[sel]
        print('Medium WP CSV V2 B tag applied to first two jets')
    elif (btag == 'bmed'):
        sel = (events.FatJet[:, 0].btagDeepB >= 0.6001)
        events = events[sel]
        print('Medium WP CSV V2 B tag applied to leading jet only')
    else:
        sel = np.ones(len(events), dtype=bool)
        print('no btag applied')
    return events, sel


def get_gen_sd_mass_jet(jet, subjets):
    combs = ak.cartesian((jet, subjets), axis=1)
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    combs = combs[dr_jet_subjets < 0.8]
    total = combs['1'].sum(axis=1)
    return total


def get_dphi(jet0, jet1):
    '''Find dphi between two jets, returning none when the event has < 2 jets.'''
    combs = ak.cartesian((jet0, jet1), axis=1)
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    return ak.firsts(dphi)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out."""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


def getLumiMask(year):
    """Single-IOV lumi mask (gluon-style call), backed by zjet's golden JSONs."""
    return getLumiMaskRun2()[year]


# in fb^-1, from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable
lumi = {'2018': 59.74, '2017': 41.48, '2016APV': 19.5, "2016": 16.8}

xsdb = {
    'QCD_Pt_170to300_TuneCP5_13TeV_pythia8': 104000.0,
    'QCD_Pt_300to470_TuneCP5_13TeV_pythia8': 6806.0,
    'QCD_Pt_470to600_TuneCP5_13TeV_pythia8': 552.0,
    'QCD_Pt_600to800_TuneCP5_13TeV_pythia8': 154.6,
    'QCD_Pt_800to1000_TuneCP5_13TeV_pythia8': 26.15,
    'QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8': 0.03567,
    'QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8': 0.6419,
    'QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8': 0.0877,
    'QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8': 0.005241,
    'QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8': 0.0001346,
    'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 23640000.0,
    'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 1546000.0,
    'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 321600.0,
    'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 30250.0,
    'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7': 6364.0,
    'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 1117.0,
    'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 108.4,
    'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 22.36,
    'QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 23640000.0,
    'QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 1546000.0,
    'QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 321600.0,
    'QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 30250.0,
    'QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 6364.0,
    'QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 1117.0,
    'QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 108.4,
    'QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 22.36,
    'QCD_HT100to200_TuneCP5_13TeV-madgraphMLM-pythia8': 23700000.0,
    'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8': 1552000.0,
    'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8': 321100.0,
    'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8': 30980.0,
    'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8': 6398.0,
    'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8': 1122.0,
    'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8': 109.4,
    'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8': 21.74,
    'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7': 1329000000.0,
    'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8': 53940.0,
    'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 271.3,
    'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8': 72.69,
    'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8': 9.961,
    'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8': 2.425,
    'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8': 1.076,
    'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 0.2474,
    'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8': 0.005609,
    'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 471.7,
}

sumw_qcd_mg = {
    '2016APV': {
        'QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 101566726040.5,
        'QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 18372261702.0,
        'QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 2253983329.5,
        'QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 409397088.68359375,
        'QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 20934451.69921875,
        'QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 1983266.7685546875,
        'QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 196907.1982421875,
    },
    '2016': {
        'QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 70210797200.0,
        'QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 14630450930.0,
        'QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 2254432227.65625,
        'QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 158406238.25,
        'QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 8486996.25390625,
        'QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 1361777.333267212,
        'QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8': 67843.4627532959,
    },
    '2017': {
        'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8': 181026146494.0,
        'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8': 36516294423.5,
        'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8': 3486166624.203125,
        'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8': 563102688.9375,
        'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8': 32239843.666503906,
        'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8': 2451711.3540649414,
        'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8': 248156.31952667236,
    },
    '2018': {
        'QCD_HT200to300_TuneCP5_13TeV-madgraphMLM-pythia8': 181673668336.0,
        'QCD_HT300to500_TuneCP5_13TeV-madgraphMLM-pythia8': 35869948198.75,
        'QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8': 3529868167.8984375,
        'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8': 575521898.7402344,
        'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8': 33323673.349121094,
        'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8': 2520025.5220947266,
        'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8': 260872.52236175537,
    },
}

num_gen_herwig_flat = {
    '2016':    {'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7': 53923986},
    '2016APV': {'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7': 45952213},
    '2017':    {'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7': 99773000},
    '2018':    {'QCD_Pt-15to7000_TuneCH3_Flat_13TeV_herwig7': 99775000},
}

sumw_herwig = {
    "2016APV": {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 141536808960.0,
                'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 8881636224.0,
                'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1006554600.0,
                'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 50850553.5,
                'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7': 11126359.375,
                'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 1957313.9375,
                'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 180116.1953125,
                'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 36778.2041015625},
    "2016":    {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 144014468096.0,
                'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 8579037856.0,
                'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 1083017536.0,
                'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 56391885.375,
                'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7': 11584725.5,
                'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 2099561.78125,
                'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 181317.447265625,
                'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 34087.36328125},
    "2017":    {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 120172825088.0,
                'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 5956648160.0,
                'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 906730648.0,
                'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 44980886.0,
                'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7': 9735645.125,
                'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 1673187.5625,
                'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 155704.818359375,
                'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 29484.88671875},
    "2018":    {'QCD_HT100to200_TuneCH3_13TeV-madgraphMLM-herwig7': 117946849024.0,
                'QCD_HT200to300_TuneCH3_13TeV-madgraphMLM-herwig7': 7728662208.0,
                'QCD_HT300to500_TuneCH3_13TeV-madgraphMLM-herwig7': 754584448.0,
                'QCD_HT500to700_TuneCH3_13TeV-madgraphMLM-herwig7': 37442241.0,
                'QCD_HT700to1000_TuneCH3_13TeV-madgraphMLM-herwig7': 9712919.75,
                'QCD_HT1000to1500_TuneCH3_13TeV-madgraphMLM-herwig7': 1673205.375,
                'QCD_HT1500to2000_TuneCH3_13TeV-madgraphMLM-herwig7': 155008.26953125,
                'QCD_HT2000toInf_TuneCH3_13TeV-madgraphMLM-herwig7': 29513.98876953125},
}

sumw_bg = {
    '2016APV': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 7715405,
        'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8': 7531529,
        'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8': 6770574,
        'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8': 2030858,
        'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8': 703970,
        'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 136393,
        'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8': 111838,
        'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 9238796.376953125},
    '2016': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 7083216,
        'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8': 6814106,
        'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8': 6114046,
        'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8': 1881671,
        'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8': 633500,
        'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 115609,
        'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8': 110461,
        'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 10952247.5546875},
    '2017': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 18948271,
        'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8': 17189820,
        'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8': 13963690,
        'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8': 4418971,
        'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8': 1513585,
        'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 267125,
        'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8': 176201,
        'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8': 9566440293248.0},
    '2018': {
        'ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8': 6299042.111694336,
        'ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8': 1676015.0802001953,
        'ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8': 233325.02221679688,
        'ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8': 13479.285751342773,
        'ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8': 2724.645004272461,
        'ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8': 136.23207068443298,
        'ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8': 2.5148895382881165,
        'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': 9437788.830200195,
        'WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8': 9537355436664.0},
}


def getXSweight(dataset, IOV):
    for year in np.array(list(lumi.keys())):
        if year in IOV:
            lum = lumi[year]
            for process in np.array(list(xsdb.keys())):
                if process in dataset:
                    xs = xsdb[process]
                    if 'herwig' in process:
                        if "madgraphMLM" in process:
                            weight = xs * lum * 1000 / sumw_herwig[year][process]
                        else:
                            weight = xs * lum * 1000 / num_gen_herwig_flat[year][process]
                    else:
                        if "QCD" in process and process in sumw_qcd_mg[year].keys():
                            weight = xs * lum * 1000 / sumw_qcd_mg[year][process]
                        elif process in sumw_bg[year].keys():
                            weight = xs * lum * 1000 / sumw_bg[year][process]
                        else:
                            print("Don't have xs + sumw values for process", process)
                            weight = 1.
                    return weight


turnOnPts_JetHT = {
    '2016': {'AK8PFJet40': 0., 'AK8PFJet60': 140., 'AK8PFJet80': 210., 'AK8PFJet140': 290.,
             'AK8PFJet200': 380., 'AK8PFJet260': 450., 'AK8PFJet320': 550., 'AK8PFJet400': 640.,
             'AK8PFJet450': 690., 'AK8PFJet500': 820.},
    '2016APV': {'AK8PFJet40': 0., 'AK8PFJet60': 140, 'AK8PFJet80': 210., 'AK8PFJet140': 290.,
                'AK8PFJet200': 380., 'AK8PFJet260': 450., 'AK8PFJet320': 550., 'AK8PFJet400': 640.,
                'AK8PFJet450': 730., 'AK8PFJet500': 820.},
    '2017': {'AK8PFJet40': 0., 'AK8PFJet60': 0., 'AK8PFJet80': 160., 'AK8PFJet140': 270.,
             'AK8PFJet200': 310., 'AK8PFJet260': 450., 'AK8PFJet320': 560., 'AK8PFJet400': 640.,
             'AK8PFJet450': 700., 'AK8PFJet500': 760., 'AK8PFJet550': 810.},
    '2018': {'AK8PFJet15': 0., 'AK8PFJet25': 0., 'AK8PFJet40': 0., 'AK8PFJet60': 0.,
             'AK8PFJet80': 160., 'AK8PFJet140': 270., 'AK8PFJet200': 390., 'AK8PFJet260': 470.,
             'AK8PFJet320': 570., 'AK8PFJet400': 650., 'AK8PFJet450': 710., 'AK8PFJet500': 760.,
             'AK8PFJet550': 820.},
}


def applyPrescales(events, year, trigger="AK8PFJet", turnOnPts=turnOnPts_JetHT, data=True):
    if year == '2016' or year == '2016APV':
        trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
        if trigger == "PFJet":
            pseval = correctionlib.CorrectionSet.from_file(_hadronic_corr_path("prescales", "ps_weight_JSON_PFJet2016.json"))
        else:
            pseval = correctionlib.CorrectionSet.from_file(_hadronic_corr_path("prescales", "ps_weight_JSON_2016.json"))
    elif year == '2017':
        trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
        if trigger == "PFJet":
            pseval = correctionlib.CorrectionSet.from_file(_hadronic_corr_path("prescales", "ps_weight_JSON_PFJet" + year + ".json"))
        else:
            pseval = correctionlib.CorrectionSet.from_file(_hadronic_corr_path("prescales", "ps_weight_JSON_" + year + ".json"))
    elif year == '2018':
        trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
        if trigger == "PFJet":
            pseval = correctionlib.CorrectionSet.from_file(_hadronic_corr_path("prescales", "ps_weight_JSON_PFJet" + year + ".json"))
        else:
            pseval = correctionlib.CorrectionSet.from_file(_hadronic_corr_path("prescales", "ps_weight_JSON_" + year + ".json"))
    turnOnPts = np.array(list(turnOnPts[year].values()))
    HLT_paths = [trigger + str(i) for i in trigThresh]
    events_mask = np.full(len(events), False)
    weights = np.ones(len(events))
    HLT_cutflow_initial = {}
    HLT_cutflow_final = {}

    #### allRuns_AK8HLT.csv is the result csv of running 'brilcalc trg --prescale
    #### --hltpath "HLT_AK8PFJet*" --output-style csv' and is used to create the
    #### ps_weight_JSON files. Lumimask and >=1 jet are already applied upstream.
    for i in np.arange(len(HLT_paths))[::-1]:
        path = HLT_paths[i]
        if path in events.HLT.fields:
            HLT_cutflow_initial[path] = ak.sum(events.HLT[path])
            pt0 = ak.firsts(events.FatJet[:, 0:].pt)
            psweights = pseval['prescaleWeight'].evaluate(
                ak.to_numpy(events.run), path,
                ak.to_numpy(ak.values_astype(events.luminosityBlock, np.float32)))
            if (i == (len(HLT_paths) - 1)):
                events_mask = np.where(((pt0 > turnOnPts[i]) & events.HLT[path]), True, events_mask)
                weights = np.where(((pt0 > turnOnPts[i]) & events.HLT[path]), psweights, weights)
                n_pass = ak.sum((pt0 > turnOnPts[i]) & events.HLT[path])
            else:
                events_mask = np.where(((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i + 1]) & events.HLT[path]), True, events_mask)
                weights = np.where(((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i + 1])), psweights, weights)
                n_pass = ak.sum((pt0 > turnOnPts[i]) & (pt0 <= turnOnPts[i + 1]))
            HLT_cutflow_final[path] = n_pass
    return events_mask, weights, HLT_cutflow_initial, HLT_cutflow_final
