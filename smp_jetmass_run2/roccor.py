# roccor.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.special import erf, erfinv

try:
    import awkward as ak
except Exception:
    ak = None


# -------------------------
# CrystalBall (as in C++)
# -------------------------
@dataclass
class CrystalBall:
    m: float = 0.0
    s: float = 1.0
    a: float = 10.0
    n: float = 10.0

    # derived constants
    B: float = field(init=False, default=0.0)
    C: float = field(init=False, default=0.0)
    D: float = field(init=False, default=0.0)
    N: float = field(init=False, default=0.0)
    NA: float = field(init=False, default=0.0)
    Ns: float = field(init=False, default=0.0)
    NC: float = field(init=False, default=0.0)
    F: float = field(init=False, default=0.0)
    G: float = field(init=False, default=0.0)
    k: float = field(init=False, default=0.0)
    cdfMa: float = field(init=False, default=0.0)
    cdfPa: float = field(init=False, default=0.0)

    pi: float = math.pi
    sqrtPiOver2: float = math.sqrt(math.pi / 2.0)
    sqrt2: float = math.sqrt(2.0)

    def init(self) -> None:
        fa = abs(self.a)
        ex = math.exp(-fa * fa / 2.0)
        A = (self.n / fa) ** self.n * ex
        C1 = (self.n / fa) / (self.n - 1.0) * ex
        D1 = 2.0 * self.sqrtPiOver2 * erf(fa / self.sqrt2)

        self.B = self.n / fa - fa
        self.C = (D1 + 2.0 * C1) / C1
        self.D = (D1 + 2.0 * C1) / 2.0

        self.N = 1.0 / self.s / (D1 + 2.0 * C1)
        self.k = 1.0 / (self.n - 1.0)

        self.NA = self.N * A
        self.Ns = self.N * self.s
        self.NC = self.Ns * C1
        self.F = 1.0 - fa * fa / self.n
        self.G = self.s * self.n / fa

        self.cdfMa = self.cdf(self.m - self.a * self.s)
        self.cdfPa = self.cdf(self.m + self.a * self.s)

    def cdf(self, x: float) -> float:
        d = (x - self.m) / self.s
        if d < -self.a:
            return self.NC / (self.F - self.s * d / self.G) ** (self.n - 1.0)
        if d > self.a:
            return self.NC * (self.C - (self.F + self.s * d / self.G) ** (1.0 - self.n))
        return self.Ns * (self.D - self.sqrtPiOver2 * erf(-d / self.sqrt2))

    def invcdf(self, u: float) -> float:
        if u < self.cdfMa:
            return self.m + self.G * (self.F - (self.NC / u) ** self.k)
        if u > self.cdfPa:
            return self.m - self.G * (self.F - (self.C - u / self.NC) ** (-self.k))
        arg = (self.D - u / self.Ns) / self.sqrtPiOver2
        return self.m - self.sqrt2 * self.s * erfinv(arg)


# -------------------------
# Resolution helper (RocRes)
# -------------------------
class RocRes:
    MC, Data, Extra = 0, 1, 2

    @dataclass
    class ResParams:
        eta: float = 0.0
        kRes: List[float] = field(default_factory=lambda: [1.0, 1.0])   # [MC, Data]
        nTrk: List[List[float]] = field(default_factory=lambda: [[], []])  # [MC/Data][RTRK+1]
        rsPar: List[List[float]] = field(default_factory=lambda: [[], [], []])  # 3 x RTRK
        cb: List[CrystalBall] = field(default_factory=list)  # RTRK

    def __init__(self):
        self.NETA = 0
        self.NTRK = 0
        self.NMIN = 0
        self.resol: List[RocRes.ResParams] = []

    def etaBin(self, eta: float) -> int:
        abseta = abs(eta)
        for i in range(self.NETA - 1):
            if abseta < self.resol[i + 1].eta:
                return i
        return self.NETA - 1

    def trkBin(self, x: float, h: int, T: int = MC) -> int:
        arr = self.resol[h].nTrk[T]
        for i in range(self.NTRK - 1):
            if x < arr[i + 1]:
                return i
        return self.NTRK - 1

    def Sigma(self, pt: float, H: int, F: int) -> float:
        dpt = pt - 45.0
        rp = self.resol[H]
        return rp.rsPar[0][F] + rp.rsPar[1][F] * dpt + rp.rsPar[2][F] * dpt * dpt

    def rndm(self, H: int, F: int, w: float) -> float:
        rp = self.resol[H]
        return rp.nTrk[self.MC][F] + (rp.nTrk[self.MC][F + 1] - rp.nTrk[self.MC][F]) * w

    def kSpread_simple(self, gpt: float, rpt: float, eta: float) -> float:
        H = self.etaBin(eta)
        kMC = self.resol[H].kRes[self.MC]
        kDT = self.resol[H].kRes[self.Data]
        x = gpt / rpt
        return x / (1.0 + (x - 1.0) * kDT / kMC)

    def kExtra(self, pt: float, eta: float, n: int, u: float, w: Optional[float] = None) -> float:
        H = self.etaBin(eta)
        F = (n - self.NMIN) if n > self.NMIN else 0
        rp = self.resol[H]

        if w is None:
            d, m = rp.kRes[self.Data], rp.kRes[self.MC]
            if d > m:
                x = math.sqrt(d * d - m * m) * self.Sigma(pt, H, F) * rp.cb[F].invcdf(u)
            else:
                x = 0.0
        else:
            v = rp.nTrk[self.MC][F] + (rp.nTrk[self.MC][F + 1] - rp.nTrk[self.MC][F]) * w
            D = self.trkBin(v, H, self.Data)
            RD = rp.kRes[self.Data] * self.Sigma(pt, H, D)
            RM = rp.kRes[self.MC] * self.Sigma(pt, H, F)
            if RD > RM:
                x = math.sqrt(RD * RD - RM * RM) * rp.cb[F].invcdf(u)
            else:
                x = 0.0

        if x <= -1.0:
            return 1.0
        return 1.0 / (1.0 + x)


# -------------------------
# Main class (RoccoR)
# -------------------------
class RoccoR:
    MC, DT = 0, 1
    MPHI = -math.pi

    @dataclass
    class RocOne:
        RR: RocRes = field(default_factory=RocRes)

    def __init__(self, filename: Optional[str] = None):
        self.nset = 0
        self.nmem: List[int] = []
        self.tvar: List[int] = []

        self.NETA = 0
        self.NPHI = 0
        self.DPHI = 0.0
        self.etabin: List[float] = []

        self.RC: List[List[RoccoR.RocOne]] = []

        # CP tables as numeric arrays for vectorization:
        # CPM[s, m, type(0/1), etaBin, phiBin]
        self.CPM: Optional[np.ndarray] = None
        self.CPA: Optional[np.ndarray] = None

        if filename is not None:
            self.init(filename)

    # ---------- binning (vector-friendly) ----------
    def etaBin(self, eta):
        # eta is signed; bins are in signed space (as in official file)
        eta = np.asarray(eta)
        bins = np.digitize(eta, self.etabin) - 1
        return np.clip(bins, 0, self.NETA - 1)

    def phiBin(self, phi):
        phi = np.asarray(phi)
        ibin = ((phi - self.MPHI) / self.DPHI).astype(np.int64)
        return np.clip(ibin, 0, self.NPHI - 1)

    # ---------- init / parser ----------
    def init(self, filename: str):
        # These correspond to "RMIN RTRK RETA" blocks in the txt
        RMIN = 0
        RTRK = 0
        RETA = 0
        BETA: List[float] = []

        with open(filename, "r") as f:
            lines = f.readlines()

        # pass 1: set global sizes and allocate RC
        for line in lines:
            s = line.strip()
            if not s:
                continue
            tag4 = s[:4]
            toks = s.split()

            if tag4 == "NSET":
                self.nset = int(toks[1])
                self.nmem = [0] * self.nset
                self.tvar = [0] * self.nset
                self.RC = [[] for _ in range(self.nset)]

            elif tag4 == "NMEM":
                for i in range(self.nset):
                    self.nmem[i] = int(toks[1 + i])
                    self.RC[i] = [self.RocOne() for _ in range(self.nmem[i])]

            elif tag4 == "TVAR":
                for i in range(self.nset):
                    self.tvar[i] = int(toks[1 + i])

            elif tag4 == "RMIN":
                RMIN = int(toks[1])

            elif tag4 == "RTRK":
                RTRK = int(toks[1])

            elif tag4 == "RETA":
                RETA = int(toks[1])
                BETA = [float(x) for x in toks[2:2 + RETA + 1]]

            elif tag4 == "CPHI":
                self.NPHI = int(toks[1])
                self.DPHI = 2.0 * math.pi / self.NPHI

            elif tag4 == "CETA":
                self.NETA = int(toks[1])
                self.etabin = [float(x) for x in toks[2:2 + self.NETA + 1]]

        if self.nset == 0 or self.NETA == 0 or self.NPHI == 0:
            raise RuntimeError("Failed to parse header: NSET/CETA/CPHI not found or invalid.")

        max_mem = max(self.nmem) if self.nmem else 1
        self.CPM = np.zeros((self.nset, max_mem, 2, self.NETA, self.NPHI), dtype=np.float64)
        self.CPA = np.zeros((self.nset, max_mem, 2, self.NETA, self.NPHI), dtype=np.float64)

        # init RR tables for each sys/mem
        for sys in range(self.nset):
            for mem in range(self.nmem[sys]):
                rr = self.RC[sys][mem].RR
                rr.NETA, rr.NTRK, rr.NMIN = RETA, RTRK, RMIN
                rr.resol = [RocRes.ResParams() for _ in range(RETA)]
                for ir in range(RETA):
                    rp = rr.resol[ir]
                    rp.eta = BETA[ir]
                    rp.cb = [CrystalBall() for _ in range(RTRK)]
                    rp.nTrk = [[0.0] * (RTRK + 1), [0.0] * (RTRK + 1)]
                    rp.rsPar = [[0.0] * RTRK, [0.0] * RTRK, [0.0] * RTRK]

        # pass 2: fill RR + CP arrays
        for line in lines:
            s = line.strip()
            if not s:
                continue
            tag4 = s[:4]
            if tag4 in ("NSET", "NMEM", "TVAR", "RMIN", "RTRK", "RETA", "CPHI", "CETA"):
                continue

            toks = s.split()
            sys = int(toks[0])
            mem = int(toks[1])
            tag = toks[2]

            if sys < 0 or sys >= self.nset or mem < 0 or mem >= self.nmem[sys]:
                continue

            rr = self.RC[sys][mem].RR

            if tag == "R":
                var = int(toks[3])
                bin_ = int(toks[4])
                vals = [float(x) for x in toks[5:5 + rr.NTRK]]
                for i in range(rr.NTRK):
                    if var in (0, 1):
                        rr.resol[bin_].rsPar[var][i] = vals[i]
                    elif var == 2:
                        rr.resol[bin_].rsPar[var][i] = vals[i] / 100.0
                    elif var == 3:
                        rr.resol[bin_].cb[i].s = vals[i]
                    elif var == 4:
                        rr.resol[bin_].cb[i].a = vals[i]
                    elif var == 5:
                        rr.resol[bin_].cb[i].n = vals[i]

            elif tag == "T":
                type_ = int(toks[3])  # 0 MC, 1 Data
                bin_ = int(toks[4])
                vals = [float(x) for x in toks[5:5 + rr.NTRK + 1]]
                rr.resol[bin_].nTrk[type_] = vals

            elif tag == "F":
                type_ = int(toks[3])  # 0 MC, 1 Data
                vals = [float(x) for x in toks[4:4 + rr.NETA]]
                for i in range(rr.NETA):
                    rr.resol[i].kRes[type_] = vals[i]

            elif tag == "C":
                type_ = int(toks[3])  # 0 MC, 1 DT
                var = int(toks[4])    # 0 -> M, 1 -> A
                bin_ = int(toks[5])   # eta bin index for CP
                vals = np.array([float(x) for x in toks[6:6 + self.NPHI]], dtype=np.float64)

                # C++ convention: M = 1 + vals/100 ; A = vals/100
                if var == 0:
                    self.CPM[sys, mem, type_, bin_, :] = 1.0 + vals / 100.0
                elif var == 1:
                    self.CPA[sys, mem, type_, bin_, :] = vals / 100.0

        # finalize CrystalBall init
        for sys in range(self.nset):
            for mem in range(self.nmem[sys]):
                rr = self.RC[sys][mem].RR
                for rp in rr.resol:
                    for cb in rp.cb:
                        cb.init()

    # ---------- Core: vectorized CP-based scale ----------
    def kScaleDT(self, Q, pt, eta, phi, s: int = 0, m: int = 0):
        Q, pt, eta, phi = np.broadcast_arrays(Q, pt, eta, phi)
        H = self.etaBin(eta).astype(np.int64)
        F = self.phiBin(phi).astype(np.int64)
        M = self.CPM[s, m, self.DT][H, F]
        A = self.CPA[s, m, self.DT][H, F]
        return 1.0 / (M + Q.astype(np.int64) * A * pt)

    def kScaleMC(self, Q, pt, eta, phi, s: int = 0, m: int = 0):
        Q, pt, eta, phi = np.broadcast_arrays(Q, pt, eta, phi)
        H = self.etaBin(eta).astype(np.int64)
        F = self.phiBin(phi).astype(np.int64)
        M = self.CPM[s, m, self.MC][H, F]
        A = self.CPA[s, m, self.MC][H, F]
        return 1.0 / (M + Q.astype(np.int64) * A * pt)

    # ---------- Spread: CP vectorized + RR spread (scalar inside RR but fast) ----------
    def kSpreadMC(self, Q, pt, eta, phi, gt, s: int = 0, m: int = 0):
        Q, pt, eta, phi, gt = np.broadcast_arrays(Q, pt, eta, phi, gt)
        # CP part is vectorized
        k = self.kScaleMC(Q, pt, eta, phi, s=s, m=m)

        rr = self.RC[s][m].RR
        out = np.empty_like(pt, dtype=np.float64)

        kf = np.ravel(k)
        ptf = np.ravel(pt)
        etaf = np.ravel(eta)
        gtf = np.ravel(gt)
        outf = np.ravel(out)

        for i in range(outf.size):
            rpt = kf[i] * float(ptf[i])
            outf[i] = kf[i] * rr.kSpread_simple(float(gtf[i]), rpt, float(etaf[i]))

        return out

    # ---------- Smear: CP vectorized + RR.kExtra loop (the only loop you really need) ----------
    def kSmearMC(self, Q, pt, eta, phi, n, u, s: int = 0, m: int = 0):
        Q, pt, eta, phi, n, u = np.broadcast_arrays(Q, pt, eta, phi, n, u)
        k = self.kScaleMC(Q, pt, eta, phi, s=s, m=m)

        rr = self.RC[s][m].RR
        out = np.empty_like(pt, dtype=np.float64)

        kf = np.ravel(k)
        ptf = np.ravel(pt)
        etaf = np.ravel(eta)
        nf = np.ravel(n)
        uf = np.ravel(u)
        outf = np.ravel(out)

        for i in range(outf.size):
            k_i = float(kf[i])
            outf[i] = k_i * rr.kExtra(k_i * float(ptf[i]), float(etaf[i]), int(nf[i]), float(uf[i]))

        return out

    # =====================================================================
    # Awkward helpers (these are what you want in coffea land)
    # =====================================================================
    def ak_kScaleDT(self, muons, s: int = 0, m: int = 0):
        """
        muons: awkward Array of muon records with fields: charge, pt, eta, phi
        returns awkward array with same structure as muons.pt
        """
        if ak is None:
            raise RuntimeError("awkward is not available. pip install awkward")

        pt = ak.to_numpy(ak.flatten(muons.pt))
        eta = ak.to_numpy(ak.flatten(muons.eta))
        phi = ak.to_numpy(ak.flatten(muons.phi))
        q = ak.to_numpy(ak.flatten(muons.charge))

        k = self.kScaleDT(q, pt, eta, phi, s=s, m=m)
        return ak.unflatten(k, ak.num(muons.pt))

    def ak_kScaleMC(self, muons, s: int = 0, m: int = 0):
        if ak is None:
            raise RuntimeError("awkward is not available. pip install awkward")

        pt = ak.to_numpy(ak.flatten(muons.pt))
        eta = ak.to_numpy(ak.flatten(muons.eta))
        phi = ak.to_numpy(ak.flatten(muons.phi))
        q = ak.to_numpy(ak.flatten(muons.charge))

        k = self.kScaleMC(q, pt, eta, phi, s=s, m=m)
        return ak.unflatten(k, ak.num(muons.pt))

    def ak_kSmearMC(self, muons, nTrackerLayers, rng: np.random.Generator, s: int = 0, m: int = 0):
        """
        muons: awkward muon records
        nTrackerLayers: awkward array aligned with muons (same jagged structure)
        rng: numpy random Generator (pass a stable one for reproducibility)
        returns awkward array of k factors (same structure as muons.pt)
        """
        if ak is None:
            raise RuntimeError("awkward is not available. pip install awkward")

        pt = ak.to_numpy(ak.flatten(muons.pt))
        eta = ak.to_numpy(ak.flatten(muons.eta))
        phi = ak.to_numpy(ak.flatten(muons.phi))
        q = ak.to_numpy(ak.flatten(muons.charge))
        n = ak.to_numpy(ak.flatten(nTrackerLayers))

        u = rng.random(pt.shape[0], dtype=np.float64)

        k = self.kSmearMC(q, pt, eta, phi, n, u, s=s, m=m)
        return ak.unflatten(k, ak.num(muons.pt))